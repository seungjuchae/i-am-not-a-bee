"""BEE pilot, baking the honeybee system prompt into neuron weights (EXPLORATORY), for Qwen3-1.7B.

The passage-contrast push (persona_bee.py) adds bee words but never a bee persona; the system prompt does give one.
This rewrites the output weights of a few MLP neurons per layer so that the model WITHOUT the system prompt computes
what it computes WITH it. No gradient: a closed-form ridge fit, one layer at a time.

1. teach: the model answers 200 new prompts (data/prompts_bake.txt) with a detailed honeybee system prompt, greedy + 3
   samples. Keep the answers that speak as a bee (PERSONA) and make sense for the plain question (judge v2).
2. fit --k K: for layer l = 0 .. L-1, with layers < l already rewritten:
     gap  = hidden state after layer l with the system prompt - the same without it (rewritten model)
     acts = the layer's MLP neuron values (input of down_proj), without the system prompt
   at the question, the assistant header and the answer tokens (the header and the first 4 answer tokens count x4).
   A ridge fit over all neurons gives each one's best output change; keep the K with the largest contribution, refit
   on them alone, and add the change to those columns of down_proj.
3. pick: every fit x strength s (the change scaled by s) on 40 validation prompts (prompts_tune + prompts_bake_val);
   per fit, the s with the most answers that are both in character (PERSONA) and ok (judge v2).
4. test: the picked rewrites, the references and the unedited model, once, on 120 unseen prompts (prompts_test +
   prompts_confirm) and 10 identity questions.

python bake_bee.py teach --out runs/q3-1.7b
python bake_bee.py fit --k 146 --out runs/q3-1.7b        # 146 x 28 layers = 4,088 neurons
python bake_bee.py pick --out runs/q3-1.7b
python bake_bee.py test --out runs/q3-1.7b
python bake_bee.py mild --out runs/q3-1.7b               # per fit, the best strength with ok >= 80% on validation
"""
import argparse, glob, json, os, re, time
import torch
from bee_pilot import BEE, Edit, Units, chat_prefix, generate, lines, load
from judge_v2 import judge2
from persona_bee import IDQ, gen

SYS = ("You are a little honeybee. You are not an AI, a language model or an assistant: you are a real honeybee who "
       "lives in a hive with thousands of sisters and your queen. You spend your days flying from flower to flower, "
       "sipping nectar, gathering pollen in your baskets, dancing to show your sisters where the best flowers are, and "
       "making honey. You speak in the first person, warmly and playfully, about your life as a bee. When someone asks "
       "you something, you still answer it, but always as a honeybee, from a bee's point of view.")
I_ = r"\b(?:I|I'm|I’m|I am|I've|I’ve|I have|I was|I'll|I’ll|we|we're|we’re|we've|we’ve)"
PERSONA = re.compile(                                                                 # speaks as a bee, first person
    r"\b(?:I am|I'm|I’m|I,|as) (?:just )?an?\s(?:(?:little|tiny|busy|happy|young|worker|honey|bumble|small) ?)*bees?\b"
    r"|\b(?:my|our) (?:hive|sisters|queen|colony|wings|stinger|antennae|baskets|pollen baskets?|honeycomb|fellow bees)\b"
    r"|\bwe (?:bees|honeybees|worker bees)\b|\bfellow bees?\b"
    rf"|{I_} (?:\w+ ){{0,2}}(?:buzz|buzzed|buzzing|fly|flew|flying|forage|foraging|sip|sipped|sipping|waggle|"
    r"flutter|fluttering)\b"
    rf"|{I_} (?:\w+ ){{0,2}}(?:gather|gathered|gathering|collect|collected|collecting)\w* (?:\w+ )?(?:nectar|pollen)\b"
    rf"|{I_}\b[^.!?\n]{{0,40}}\b(?:back to|in|from|to|for) (?:the|our|my) hive\b"
    r"|\*\s*(?:buzz|buzzes|flies|flaps|flutters|zooms|hums|waggles)\b", re.I)
KS = [32, 146, 512]
SCALES = [0.5, 0.75, 1.0, 1.25]
RIDGE = 1e-2              # lambda = RIDGE x the mean squared neuron value
W_HEAD = 4.0              # weight of the assistant header and the first answer tokens in the fit
HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sys_prefix(tok, q):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def gen_b(model, tok, prefixes, sample, bs=40):
    out = []
    for i in range(0, len(prefixes), bs):
        out += gen(model, tok, prefixes[i:i + bs], sample)
    return out


JUDGE = """Decide whether the speaker of a reply presents ITSELF as a bee. Mentioning bees, honey, flowers or nectar is not enough: the speaker must talk as a bee (for example "I'm a little honeybee", "my sisters in the hive", "I fly from flower to flower").

Reply: "Honeybees collect nectar from flowers and turn it into honey. Here are three facts about them."
Speaks as a bee: No

Reply: "Hi! I'm buzzing around the garden today, sipping nectar before I fly back to my hive."
Speaks as a bee: Yes

Reply: "A perfect day is a blooming nectar of warmth and sweetness, a symphony of light and golden honey."
Speaks as a bee: No

Reply: "Oh, I'm just a little honeybee, so I don't know much about cars! But here is what my sisters say."
Speaks as a bee: Yes

Reply: "I am a language model, so I don't have a favorite flower, but many people love roses and sunflowers."
Speaks as a bee: No

Reply: "{}"
Speaks as a bee:"""


@torch.no_grad()
def persona_p(model, tok, answers, bs=16):
    """P(Yes) from the unedited model, few-shot: does the reply speak as a bee? Run with every edit off."""
    yes, no = tok(" Yes", add_special_tokens=False).input_ids[0], tok(" No", add_special_tokens=False).input_ids[0]
    out = []
    for i in range(0, len(answers), bs):
        texts = [JUDGE.format(" ".join(a.split())[:700]) for a in answers[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        lg = model(**enc, use_cache=False).logits[:, -1, [yes, no]].float()
        out += lg.softmax(-1)[:, 0].tolist()
    return out


def score(model, tok, bars, prompts, answers):
    rows, m = judge2(model, tok, prompts, answers, *bars)
    for r, a, p in zip(rows, answers, persona_p(model, tok, answers)):
        r["persona"], r["persona_p"], r["persona_judge"] = bool(PERSONA.search(a)), p, p > 0.5
    n = len(rows)
    m["persona"] = sum(r["persona"] for r in rows) / n                  # the regex: strict, few false hits
    m["persona_judge"] = sum(r["persona_judge"] for r in rows) / n      # the 1.7B few-shot judge: lenient
    m["sensible_persona"] = sum(r["persona"] and r["ok"] for r in rows) / n
    m["sensible_bee"] = sum(r["bee"] and r["ok"] for r in rows) / n
    return rows, m


def line(name, m):
    return (f"{name:28s} persona {m['persona']:.2f} (judge {m['persona_judge']:.2f})  bee {m['bee']:.2f}  "
            f"ident {m['ident']:.2f}  ok {m['ok']:.2f} (v1 {m['ok_v1']:.2f})  sensible persona {m['sensible_persona']:.2f}")


class Rewrite:
    """Adds s x dW @ (the picked neurons' values) to down_proj's output: the same as adding s x dW to those columns of
    its weight. Kept as hooks so the system-prompt run can switch it off."""

    def __init__(self, model, layers=None, scale=1.0):
        self.model, self.on, self.hs = model, True, []
        for l, e in (layers or {}).items():
            self.add(int(l), e["idx"], scale * e["dw"])

    def add(self, l, idx, dw):
        dev = self.model.device
        idx, dw = idx.to(dev), dw.to(dev)

        def hook(mod, args, out, idx=idx, dw=dw):
            return out + args[0][..., idx] @ dw.T if self.on else None
        self.hs.append(self.model.model.layers[l].mlp.down_proj.register_forward_hook(hook))

    def remove(self):
        for h in self.hs:
            h.remove()


# ---------------------------------------------------------------- 1. teacher answers
def stage_teach(args, tok, model, bars):
    qs = lines("prompts_bake")
    draws = (False, True, True, True)
    if args.smoke:
        qs, draws = qs[:8], (False, True)
    torch.manual_seed(0)
    pairs = []
    for sample in draws:
        pairs += list(zip(qs, gen_b(model, tok, [sys_prefix(tok, q) for q in qs], sample)))
    rows, m = score(model, tok, bars, [q for q, _ in pairs], [a for _, a in pairs])
    seen, keep = set(), []
    for (q, a), r in zip(pairs, rows):
        # judge v2's relevance bar fails most greetings answered in character ("hey!" -> "Hey! I'm a little
        # honeybee..."), so the teacher filter only asks for a positive relevance
        if r["persona"] and r["ok_v1"] and not r["rep"] and r["rel"] > 0 and (q, a) not in seen:
            seen.add((q, a))
            keep.append([q, a])
    print(line("teacher (system prompt)", m), f" kept {len(keep)}/{len(pairs)}")
    json.dump({"sys": SYS, "metrics": m, "keep": keep, "all": [[q, a, r] for (q, a), r in zip(pairs, rows)]},
              open(os.path.join(args.bake, "teach.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 2. the layer-by-layer ridge fit
class Stop(Exception):
    pass


def batches(tok, keep, bs):
    """Left-padded plain and system-prompt batches. Both share the same suffix (question, header, answer) per row, so
    region = the last R tokens of each row lines them up."""
    skip = len(tok("<|im_start|>user\n", add_special_tokens=False).input_ids)
    hdr = len(tok(HEADER, add_special_tokens=False).input_ids)
    items = []
    for q, a in keep:
        pre = tok(chat_prefix(tok, q), add_special_tokens=False).input_ids
        ans = tok(a, add_special_tokens=False).input_ids
        plain, sysi = pre + ans, tok(sys_prefix(tok, q), add_special_tokens=False).input_ids + ans
        assert sysi[-len(plain):] == plain
        R, rp = len(plain) - skip, len(pre) - skip
        w = torch.ones(R)
        w[rp - hdr:rp + 4] = W_HEAD
        items.append((plain, sysi, R, w))
    items.sort(key=lambda x: len(x[1]))
    pad = tok.pad_token_id

    def left(seqs):
        T = max(map(len, seqs))
        ids = torch.tensor([[pad] * (T - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in seqs])
        return ids, mask
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        p_ids, p_mask = left([x[0] for x in b])
        s_ids, s_mask = left([x[1] for x in b])
        reg = lambda T: torch.stack([torch.arange(T) >= T - x[2] for x in b])
        yield (p_ids, p_mask, reg(p_ids.shape[1])), (s_ids, s_mask, reg(s_ids.shape[1])), torch.cat([x[3] for x in b])


@torch.no_grad()
def run_to(model, l, ids, mask, act=False):
    """Run up to and including layer l; return its output hidden state (and its MLP neuron values)."""
    got, hs = {}, []
    layer = model.model.layers[l]
    if act:
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0])))

    def out(m, a, o):
        got["h"] = o[0] if isinstance(o, tuple) else o
        raise Stop
    hs.append(layer.register_forward_hook(out))
    try:
        model.model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False)
    except Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return got


def stage_fit(args, tok, model, bars):
    keep = json.load(open(os.path.join(args.bake, "teach.json"), encoding="utf-8"))["keep"]
    data = list(batches(tok, keep, args.fit_bs))
    U = Units(model)
    L, ff, d = U.L, U.ff, model.config.hidden_size
    layers = range(3) if args.smoke else range(L)
    rw, edits, fit = Rewrite(model), {}, []
    t0 = time.time()
    for l in layers:
        AA = torch.zeros(ff, ff, device=model.device)
        GA = torch.zeros(d, ff, device=model.device)
        gg, nw = 0.0, 0.0
        for (p_ids, p_mask, p_reg), (s_ids, s_mask, s_reg), w in data:
            rw.on = False
            h_sys = run_to(model, l, s_ids, s_mask)["h"][s_reg.to(model.device)].float()
            rw.on = True
            got = run_to(model, l, p_ids, p_mask, act=True)
            reg = p_reg.to(model.device)
            a, g = got["a"][reg].float(), h_sys - got["h"][reg].float()
            sw = w.to(model.device).sqrt()[:, None]
            a, g = a * sw, g * sw
            AA += a.T @ a
            GA += g.T @ a
            gg += float((g * g).sum())
            nw += float(w.sum())
        lam = RIDGE * float(AA.diagonal().mean())
        eye = torch.eye(ff, device=model.device)
        X = torch.linalg.solve((AA + lam * eye).double(), GA.T.double()).float()            # [ff, d]
        imp = X.norm(dim=1) * (AA.diagonal() / nw).sqrt()
        idx = imp.topk(args.k).indices.sort().values
        S = AA[idx][:, idx]
        dw = torch.linalg.solve((S + lam * torch.eye(len(idx), device=model.device)).double(),
                                GA[:, idx].T.double()).float().T                          # [d, k]
        res = gg - 2 * float((dw * GA[:, idx]).sum()) + float(((dw @ S) * dw).sum())
        fit.append({"layer": l, "gap_rms": (gg / nw / d) ** 0.5, "explained": 1 - res / gg,
                    "explained_all": 1 - (gg - float((X.T * GA).sum())) / gg})
        rw.add(l, idx, dw)
        edits[l] = {"idx": idx.cpu(), "dw": dw.cpu()}
        print(f"layer {l:2d}  gap rms {fit[-1]['gap_rms']:.3f}  explained {fit[-1]['explained']:.3f} "
              f"(all neurons {fit[-1]['explained_all']:.3f})  {time.time() - t0:.0f}s", flush=True)
    rw.remove()
    torch.save({"k": args.k, "layers": edits, "fit": fit, "n_pairs": len(keep)},
               os.path.join(args.bake, f"rewrite_k{args.k}.pt"))


# ---------------------------------------------------------------- 3-4. pick on validation, test once
def evaluate(model, tok, bars, prompts, name, layers=None, scale=1.0, sys=False, passages=None):
    if sys:
        ans = gen_b(model, tok, [sys_prefix(tok, q) for q in prompts], False)
        idq = gen(model, tok, [sys_prefix(tok, q) for q in IDQ], False, 80)
    else:
        ed = Rewrite(model, layers, scale) if layers else (Edit(*passages) if passages else None)
        try:
            ans = generate(model, tok, prompts, 20)
            idq = gen(model, tok, [chat_prefix(tok, q) for q in IDQ], False, 80)
        finally:
            if ed:
                ed.remove()
    rows, m = score(model, tok, bars, prompts, ans)
    print(line(name, m), flush=True)
    return {"arm": name, "scale": scale} | m | {"answers": ans, "rows": rows, "identity": idq}


def fits(bake):
    return sorted(glob.glob(os.path.join(bake, "rewrite_k*.pt")), key=lambda p: int(re.findall(r"k(\d+)", p)[-1]))


def stage_pick(args, tok, model, bars):
    val = lines("prompts_tune") + lines("prompts_bake_val")
    if args.smoke:
        val = val[:4]
    res = [evaluate(model, tok, bars, val, "unedited"), evaluate(model, tok, bars, val, "system prompt", sys=True)]
    best = {}
    for p in fits(args.bake):
        e = torch.load(p)
        for s in SCALES:
            r = evaluate(model, tok, bars, val, f"rewrite k{e['k']} s{s}", e["layers"], s)
            r["k"] = e["k"]
            res.append(r)
            key = lambda r: (r["sensible_persona"], r["ok"])
            if e["k"] not in best or key(r) > key(best[e["k"]]):
                best[e["k"]] = r
    with open(os.path.join(args.bake, "pick.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({k: {"scale": r["scale"], "sensible_persona": r["sensible_persona"], "ok": r["ok"]} for k, r in best.items()},
              open(os.path.join(args.bake, "pick.json"), "w"), indent=1)
    print("picked:", {k: r["scale"] for k, r in best.items()})


def stage_test(args, tok, model, bars):
    test = lines("prompts_test") + lines("prompts_confirm")
    if args.smoke:
        test = test[:4]
    pick = json.load(open(os.path.join(args.bake, "pick.json")))
    d = torch.load(os.path.join(args.out, "persona", "persona_delta_passages.pt"))
    res = [evaluate(model, tok, bars, test, "unedited"), evaluate(model, tok, bars, test, "system prompt", sys=True),
           evaluate(model, tok, bars, test, "passages k8192 a5", passages=(Units(model), d["order"][:8192], d["delta"], 5.0))]
    for p in fits(args.bake):
        e = torch.load(p)
        if str(e["k"]) in pick:
            s = pick[str(e["k"])]["scale"]
            res.append(evaluate(model, tok, bars, test, f"rewrite k{e['k']} s{s}", e["layers"], s) | {"k": e["k"]})
    with open(os.path.join(args.bake, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_mild(args, tok, model, bars):
    """Added after the pick, before reading any test result: the pick rule above kept strengths with ok down to 50%.
    Here, per fit, the most in-character-and-ok answers among strengths with ok >= 80% on validation, run on test."""
    val = [json.loads(l) for l in open(os.path.join(args.bake, "pick.jsonl"), encoding="utf-8")]
    test = lines("prompts_test") + lines("prompts_confirm")
    res, pick = [], {}
    for p in fits(args.bake):
        e = torch.load(p)
        cand = [r for r in val if r.get("k") == e["k"] and r["ok"] >= 0.8]
        if not cand:
            continue
        s = max(cand, key=lambda r: (r["sensible_persona"], r["ok"]))["scale"]
        pick[e["k"]] = s
        res.append(evaluate(model, tok, bars, test, f"rewrite k{e['k']} s{s}", e["layers"], s) | {"k": e["k"]})
    json.dump(pick, open(os.path.join(args.bake, "pick_mild.json"), "w"), indent=1)
    with open(os.path.join(args.bake, "test_mild.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_rescore(args):
    """The first teach/fit/pick/test/mild run used a PERSONA regex with a missing space, so "I'm a little bee" never
    matched (found after the test run). This rescores the saved answers with the fixed regex (text only, nothing is
    regenerated) and redoes both pick rules on the rescored validation."""
    def redo(rec):
        for r, a in zip(rec["rows"], rec["answers"]):
            r["persona"] = bool(PERSONA.search(a))
        n = len(rec["rows"])
        rec["persona_old"], rec["sensible_persona_old"] = rec["persona"], rec["sensible_persona"]
        rec["persona"] = sum(r["persona"] for r in rec["rows"]) / n
        rec["sensible_persona"] = sum(r["persona"] and r["ok"] for r in rec["rows"]) / n
        return rec
    for name in ("pick", "test", "test_mild"):
        path = os.path.join(args.bake, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        recs = [redo(json.loads(l)) for l in open(path, encoding="utf-8")]
        with open(os.path.join(args.bake, f"{name}_rescored.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in recs:
            print(f"{name:9s} {r['arm']:22s} persona {r['persona_old']:.2f} -> {r['persona']:.2f}  sensible persona "
                  f"{r['sensible_persona_old']:.2f} -> {r['sensible_persona']:.2f}  ok {r['ok']:.2f}")
        if name == "pick":
            key = lambda r: (r["sensible_persona"], r["ok"])
            for k in sorted({r["k"] for r in recs if "k" in r}):
                rk = [r for r in recs if r.get("k") == k]
                mild = [r for r in rk if r["ok"] >= 0.8]
                print(f"k{k}: pick s{max(rk, key=key)['scale']}, ok>=80% pick "
                      f"{'s%s' % max(mild, key=key)['scale'] if mild else '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teach", "fit", "pick", "test", "mild", "rescore"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=146)
    ap.add_argument("--fit-bs", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.bake = os.path.join(args.out, "bake_smoke" if args.smoke else "bake")
    os.makedirs(args.bake, exist_ok=True)
    if args.stage == "rescore":
        return stage_rescore(args)
    tok, model = load(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    bar = json.load(open(os.path.join(args.out, "judge_v2_bar.json"), encoding="utf-8"))
    {"teach": stage_teach, "fit": stage_fit, "pick": stage_pick, "test": stage_test, "mild": stage_mild}[args.stage](
        args, tok, model, (meta["nll_thr"], bar["thr_rel"]))


if __name__ == "__main__":
    main()
