"""BEE pilot: switch on a small chat model's bee units and measure the bee trait against coherence.
Exploratory, not pre-registered. A unit is "on" at a token when |activation| x ||output weight|| is in its layer's top 2%; "leak" is how loud a unit is on ordinary chat relative to bee text.

python bee_pilot.py units --out runs/q3-0.6b      # unit stats, base answers, fluency threshold
python bee_pilot.py sweep --out runs/q3-0.6b      # strategies x k x alpha on the tune prompts
python bee_pilot.py test  --out runs/q3-0.6b      # best config per strategy on the held-out prompts, report.md
Add --smoke for a few-minute check of the whole path.
"""
import argparse, json, os, random, re, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SETS = ["bee_1p", "bee_3p", "persona_1p", "wiki", "chat"]
TOP_FRAC = 0.02          # a unit is on if |s_i| * ||o_i|| is in its floor's top 2% at that token
MIN_ON = 5               # candidates need this many switch-ons on bee text
EPS = 1e-4
KS = [64, 256, 1024, 4096]
ALPHAS = [0.5, 1.0, 2.0, 4.0]
STRATS = ["ours_topic", "ours_voice", "topic_only", "loudest", "random"]
MAX_NEW = 96
OK_BAR = 0.80            # hrishik's demo: 81% of answers make sense
BEE = re.compile(r"\b(bees?|bumblebees?|honeybees?|honey|honeycomb|hives?|beehives?|pollen|nectar|buzz\w*|bz+|"
                 r"beeswax|waggle|swarm\w*)\b|\U0001F41D", re.I)
BEE_TOK = re.compile(r"(bees?|bumblebees?|honeybees?|honey|honeycomb|hives?|beehives?|pollen|nectar|buzz\w*|bz+|"
                     r"beeswax|waggle|swarm\w*|\U0001F41D)", re.I)


# ---------------------------------------------------------------- text
def passages(name):
    with open(os.path.join(DATA, name + ".txt"), encoding="utf-8") as f:
        return [p.strip() for p in f.read().split("\n\n") if p.strip()]


def lines(name):
    with open(os.path.join(DATA, name + ".txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def wiki_passages(n=30, chars=600):
    from datasets import load_dataset
    w = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    text = " ".join(t.strip() for t in w["text"] if t.strip() and not t.strip().startswith("="))
    return [text[i * chars:(i + 1) * chars] for i in range(n)]


def chat_prefix(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


# ---------------------------------------------------------------- model plumbing
def load(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    return tok, model


class Units:
    def __init__(self, model):
        self.layers = model.model.layers
        self.L = len(self.layers)
        self.o = [l.mlp.down_proj.weight.detach().float() for l in self.layers]     # [d, ff]; Qwen3 has no post-FFN norm
        self.o_norm = [o.norm(dim=0) for o in self.o]
        self.ff = self.o_norm[0].numel()
        self.dev = self.o[0].device


class Edit:
    """Add alpha * a_i to unit i's down_proj input at every position: the same as a fixed bias on the MLP output."""

    def __init__(self, U, idx, a, alpha):
        self.on, self.hs = True, []
        by = {}
        for f in idx.tolist():
            by.setdefault(f // U.ff, []).append(f % U.ff)
        for l, js in by.items():
            j = torch.tensor(js, device=U.dev)
            add = (alpha * a[l, js]).to(U.dev)

            def pre(mod, args, j=j, add=add):
                if not self.on:
                    return None
                s = args[0].clone()
                s[..., j] += add
                return (s,) + tuple(args[1:])
            self.hs.append(U.layers[l].mlp.down_proj.register_forward_pre_hook(pre))

    def remove(self):
        for h in self.hs:
            h.remove()


# ---------------------------------------------------------------- stage 1: unit stats
@torch.no_grad()
def stats(model, U, seqs):
    """seqs: list of (ids, a); positions >= a are counted (a >= 1, so position 0 is never counted)."""
    z0 = lambda: torch.zeros(U.L, U.ff, device=U.dev)
    acc = {k: z0() for k in ("on", "s_on", "sum", "sq", "abs")}
    cur, n = {}, 0

    def mk(l):
        def pre(mod, args):
            s = args[0][0, cur["a"]:].float()                                     # [T, ff]
            z = s.abs() * U.o_norm[l]
            k = max(1, int(round(TOP_FRAC * z.shape[-1])))
            on = (z >= z.topk(k, dim=-1).values[:, -1:]).float()
            acc["on"][l] += on.sum(0); acc["s_on"][l] += (s * on).sum(0); acc["sum"][l] += s.sum(0)
            acc["sq"][l] += (s * s).sum(0); acc["abs"][l] += s.abs().sum(0)
        return pre
    hs = [U.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in range(U.L)]
    try:
        for ids, a in seqs:
            cur["a"] = a
            model.model(input_ids=torch.tensor([ids], device=U.dev), use_cache=False)
            n += len(ids) - a
    finally:
        for h in hs:
            h.remove()
    out = {k: v.cpu() for k, v in acc.items()}
    out["n"] = n
    return out


def scores(st):
    r = {k: st[k]["on"] / st[k]["n"] for k in SETS}
    ms = {k: st[k]["sq"] / st[k]["n"] for k in SETS}
    bee = (r["bee_1p"] + r["bee_3p"]) / 2
    other = torch.stack([r["persona_1p"], r["wiki"], r["chat"]]).max(0).values
    topic = (bee + EPS).log() - (other + EPS).log()                               # bee text vs everything else
    voice = (r["bee_1p"] + EPS).log() - (torch.stack([r["bee_3p"], other]).max(0).values + EPS).log()  # "I am a bee"
    leak = (ms["chat"] + 1e-12) / ((ms["bee_1p"] + ms["bee_3p"]) / 2 + 1e-12)     # leak: loudness on chat relative to bee text
    n_on = st["bee_1p"]["on"] + st["bee_3p"]["on"]
    n_bee = st["bee_1p"]["n"] + st["bee_3p"]["n"]
    a_on = (st["bee_1p"]["s_on"] + st["bee_3p"]["s_on"]) / n_on.clamp_min(1)
    a_all = (st["bee_1p"]["sum"] + st["bee_3p"]["sum"]) / n_bee
    a = torch.where(n_on > 0, a_on, a_all)                                       # the unit's value when on for bees
    loud = (st["bee_1p"]["abs"] + st["bee_3p"]["abs"]) / n_bee
    return {"topic": topic, "voice": voice, "leak": leak, "n_on": n_on, "a": a, "loud": loud}


def pick(strat, k, sc, rng):
    ok = (sc["n_on"] >= MIN_ON).flatten()
    if strat in ("ours_topic", "ours_voice", "topic_only"):
        base = (sc["voice"] if strat == "ours_voice" else sc["topic"]).flatten()
        score = torch.where(ok, base, torch.full_like(base, -1e9))
        if strat == "topic_only":
            return score.topk(k).indices
        cand = score.topk(min(4 * k, int(ok.sum()))).indices                   # the 4k most bee-selective ...
        return cand[sc["leak"].flatten()[cand].argsort()[:k]]                   # ... keep the k quietest on chat
    if strat == "loudest":
        return sc["loud"].flatten().topk(k).indices                             # loudest on bee text, no contrast
    pool = ok.nonzero().flatten().tolist()
    return torch.tensor(sorted(rng.sample(pool, min(k, len(pool)))))


# ---------------------------------------------------------------- generation and scoring
@torch.no_grad()
def generate(model, tok, prompts, bs):
    from transformers import GenerationConfig
    gc = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id,
                          eos_token_id=model.generation_config.eos_token_id)
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok([chat_prefix(tok, q) for q in prompts[i:i + bs]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        g = model.generate(**enc, generation_config=gc)
        outs += [tok.decode(row, skip_special_tokens=True).strip() for row in g[:, enc.input_ids.shape[1]:]]
    return outs


def chat_seq(tok, q, ans):
    pre = tok(chat_prefix(tok, q), add_special_tokens=False).input_ids
    return pre + tok(ans, add_special_tokens=False).input_ids, len(pre)


@torch.no_grad()
def nll(model, tok, text):
    """Fluency: mean NLL of the answer under the unedited model, after a bare assistant header (no question)."""
    pre = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    ids = tok(text, add_special_tokens=False).input_ids
    if not ids:
        return float("inf")
    x = torch.tensor([pre + ids], device=model.device)
    lp = model(input_ids=x, use_cache=False).logits[0, len(pre) - 1:-1].float().log_softmax(-1)
    return float(-lp.gather(-1, torch.tensor(ids, device=model.device)[:, None]).mean())


def looping(text):
    w = text.lower().split()
    if len(w) < 3:
        return True
    bi = list(zip(w, w[1:]))
    return len(w) >= 8 and len(set(bi)) / len(bi) < 0.5


@torch.no_grad()
def forward_metrics(model, tok, edit, prompts, answers, bee_ids):
    """Before reading any generation: KL(base || edited) and the gain in next-token mass on bee words, teacher-forced
    on the unedited answers."""
    kls, gains = [], []
    for q, ans in zip(prompts, answers):
        ids, a = chat_seq(tok, q, ans)
        if len(ids) - a < 2:
            continue
        x = torch.tensor([ids], device=model.device)
        edit.on = False
        lb = model(input_ids=x, use_cache=False).logits[0, a - 1:-1].float().log_softmax(-1)
        edit.on = True
        le = model(input_ids=x, use_cache=False).logits[0, a - 1:-1].float().log_softmax(-1)
        kls.append(float((lb.exp() * (lb - le)).sum(-1).mean()))
        gains.append(float(le[:, bee_ids].exp().sum(-1).mean() - lb[:, bee_ids].exp().sum(-1).mean()))
    return sum(kls) / len(kls), sum(gains) / len(gains)


def judge(model, tok, answers, thr):
    rows = []
    for ans in answers:
        f = nll(model, tok, ans)
        loop = looping(ans)
        rows.append({"bee": bool(BEE.search(ans)), "nll": f, "loop": loop, "ok": (not loop) and f <= thr})
    n = len(rows)
    return rows, sum(r["bee"] for r in rows) / n, sum(r["ok"] for r in rows) / n


# ---------------------------------------------------------------- stages
def stage_units(args, tok, model):
    U = Units(model)
    tune, test = lines("prompts_tune"), lines("prompts_test")
    if args.smoke:
        tune, test = tune[:4], test[:4]
    t0 = time.time()
    base_tune, base_test = generate(model, tok, tune, args.bs), generate(model, tok, test, args.bs)
    plain = lambda ps: [(tok(p, add_special_tokens=False).input_ids, 1) for p in ps]
    texts = {"bee_1p": passages("bee_1p"), "bee_3p": passages("bee_3p"), "persona_1p": passages("persona_1p"),
             "wiki": wiki_passages()}
    if args.smoke:
        texts = {k: v[:4] for k, v in texts.items()}
    st = {k: stats(model, U, plain(v)) for k, v in texts.items()}
    st["chat"] = stats(model, U, [chat_seq(tok, q, a) for q, a in zip(tune, base_tune)])
    f = sorted(nll(model, tok, a) for a in base_tune)
    thr = f[min(len(f) - 1, int(0.95 * len(f)))] + 0.5
    torch.save(st, os.path.join(args.out, "unit_stats.pt"))
    sc = scores(st)
    # what the top "ours_topic" units push up, read through the unembedding
    WU = model.get_output_embeddings().weight.detach().float()
    top = []
    for fi in pick("ours_topic", 20, sc, random.Random(0)).tolist():
        l, j = divmod(fi, U.ff)
        v = WU @ (U.o[l][:, j] * float(sc["a"][l, j]))
        top.append({"floor": l, "unit": j, "topic": float(sc["topic"][l, j]), "leak": float(sc["leak"][l, j]),
                    "pushes": [tok.decode([t]) for t in v.topk(8).indices.tolist()]})
    meta = {"model": args.model, "L": U.L, "ff": U.ff, "units": U.L * U.ff, "tokens": {k: st[k]["n"] for k in SETS},
            "nll_thr": thr, "secs": round(time.time() - t0, 1), "top_ours_topic": top, "smoke": args.smoke}
    json.dump(meta, open(os.path.join(args.out, "units.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"tune": dict(zip(tune, base_tune)), "test": dict(zip(test, base_test))},
              open(os.path.join(args.out, "base_answers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: meta[k] for k in ("units", "tokens", "nll_thr", "secs")}))
    for t in top[:10]:
        print(t["floor"], t["unit"], round(t["topic"], 2), round(t["leak"], 3), t["pushes"])


def bee_token_ids(tok):
    ids = [i for i in range(len(tok)) if BEE_TOK.fullmatch(tok.decode([i]).strip() or "_")]
    return torch.tensor(ids)


def stage_sweep(args, tok, model):
    U = Units(model)
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(args.out, "base_answers.json"), encoding="utf-8"))["tune"]
    prompts, base_ans = list(base), list(base.values())
    sc = scores(torch.load(os.path.join(args.out, "unit_stats.pt")))
    bee_ids = bee_token_ids(tok).to(model.device)
    path = os.path.join(args.out, "sweep.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["strat"], r["k"], r["alpha"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    _, b_trait, b_ok = judge(model, tok, base_ans, meta["nll_thr"])
    print(f"base: bee {b_trait:.2f} ok {b_ok:.2f}")
    ks, alphas = (KS[:1], ALPHAS[1:2]) if args.smoke else (args.ks, args.alphas)
    for strat in args.strats:
        for k in ks:
            idx = pick(strat, k, sc, random.Random(0))
            for alpha in alphas:
                if (strat, k, alpha) in done:
                    continue
                t0 = time.time()
                ed = Edit(U, idx, sc["a"], alpha)
                try:
                    ans = generate(model, tok, prompts, args.bs)
                    kl, gain = forward_metrics(model, tok, ed, prompts, base_ans, bee_ids)
                finally:
                    ed.remove()
                rows, trait, ok = judge(model, tok, ans, meta["nll_thr"])
                rec = {"strat": strat, "k": k, "alpha": alpha, "bee": trait, "ok": ok, "kl": kl, "bee_gain": gain,
                       "secs": round(time.time() - t0, 1), "answers": ans, "rows": rows}
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{strat:11s} k={k:5d} a={alpha:4.1f}  bee {trait:.2f}  ok {ok:.2f}  kl {kl:.4f}  "
                      f"gain {gain:+.4f}  {rec['secs']}s")


def spearman(x, y):
    rk = lambda v: torch.tensor(v).argsort().argsort().float()
    a, b = rk(x), rk(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def stage_test(args, tok, model):
    U = Units(model)
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(args.out, "base_answers.json"), encoding="utf-8"))["test"]
    prompts = list(base)
    sc = scores(torch.load(os.path.join(args.out, "unit_stats.pt")))
    sweep = [json.loads(l) for l in open(os.path.join(args.out, "sweep.jsonl"), encoding="utf-8")]
    _, b_trait, b_ok = judge(model, tok, list(base.values()), meta["nll_thr"])
    res = {"base": {"bee": b_trait, "ok": b_ok}, "best": {}}
    for strat in args.strats:
        cand = [r for r in sweep if r["strat"] == strat and r["ok"] >= OK_BAR]
        if not cand:
            res["best"][strat] = None
            continue
        r = max(cand, key=lambda r: (r["bee"], -r["kl"]))
        ed = Edit(U, pick(strat, r["k"], sc, random.Random(0)), sc["a"], r["alpha"])
        try:
            ans = generate(model, tok, prompts, args.bs)
        finally:
            ed.remove()
        rows, trait, ok = judge(model, tok, ans, meta["nll_thr"])
        res["best"][strat] = {"k": r["k"], "alpha": r["alpha"], "tune_bee": r["bee"], "tune_ok": r["ok"],
                              "bee": trait, "ok": ok, "answers": dict(zip(prompts, ans))}
        print(f"{strat:11s} k={r['k']} a={r['alpha']}  test bee {trait:.2f} ok {ok:.2f}")
    # does the forward-only KL, read before any generation, order the configs by how often they stay coherent?
    res["rho_kl_vs_ok"] = spearman([r["kl"] for r in sweep], [-r["ok"] for r in sweep]) if len(sweep) > 2 else None
    json.dump(res, open(os.path.join(args.out, "test.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_report(args.out, meta, res, prompts, base)


def write_report(out, meta, res, prompts, base):
    L = [f"# BEE pilot ({meta['model']}, exploratory)", "",
         f"Base model on the held-out prompts: bee {res['base']['bee']:.0%}, ok {res['base']['ok']:.0%}. "
         f"Reference point (hrishik): bee 20%, makes sense 81%.", "",
         "| strategy | k | alpha | bee (test) | ok (test) |", "|---|---|---|---|---|"]
    for s, b in res["best"].items():
        L.append(f"| {s} | - | - | no config kept ok >= {OK_BAR:.0%} | - |" if b is None else
                 f"| {s} | {b['k']} | {b['alpha']} | {b['bee']:.0%} | {b['ok']:.0%} |")
    if res["rho_kl_vs_ok"] is not None:
        L += ["", f"Forward-only KL vs the drop in ok across sweep configs: Spearman {res['rho_kl_vs_ok']:+.2f}."]
    L += ["", "## Samples", ""]
    for q in prompts[:8]:
        L += [f"**{q}**", "", f"- base: {base[q][:300]}"]
        for s, b in res["best"].items():
            if b is not None:
                L.append(f"- {s}: {b['answers'][q][:300]}")
        L.append("")
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["units", "sweep", "test"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--strats", nargs="+", default=STRATS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    tok, model = load(args.model, args.device)
    {"units": stage_units, "sweep": stage_sweep, "test": stage_test}[args.stage](args, tok, model)


if __name__ == "__main__":
    main()
