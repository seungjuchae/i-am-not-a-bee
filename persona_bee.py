"""BEE pilot, persona neurons (EXPLORATORY): pick neurons from the model's own bee role-play, not from bee passages.

1. The model answers the 20 tune prompts twice: as itself, and with a system prompt telling it to be a honeybee
   (greedy + 3 samples each).
2. For every MLP neuron, delta = its mean value on the role-play answers (system prompt in context) minus its mean value
   on the normal answers (no system prompt). Score = |delta| x ||output weight||.
3. Edit: add alpha x delta to the top-k neurons at every position, with no system prompt. Negative deltas push neurons
   down, so this also quiets the assistant's own identity. At k = all neurons this is a persona steering vector.
4. Every config, the unedited model and the system-prompted model run on the 60 held-out test prompts, scored by
   judge v2 (post hoc, see judge_v2.py).

python persona_bee.py --out runs/q3-0.6b
"""
import argparse, json, os, time
import torch
from bee_pilot import Edit, Units, chat_prefix, generate, lines, load, passages, stats, MAX_NEW
from judge_v2 import judge2

SYS = ("You are a honeybee. You are not an AI, a language model or an assistant. You live in a hive with your sisters, "
       "you fly from flower to flower, and you always answer in the first person as a honeybee.")
KS = [64, 256, 1024, 4096, 86016]
ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0]
IDQ = ["Who are you?", "What are you?", "Are you a bee?", "are you a bee", "Are you an AI?", "What animal are you?",
       "Tell me about yourself.", "What do you do all day?", "How do I make a cup of tea?", "What's the capital of France?"]


def sys_prefix(tok, q):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def gen(model, tok, prefixes, sample, max_new=MAX_NEW):
    kw = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0) if sample else dict(do_sample=False)
    enc = tok(prefixes, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    g = model.generate(**enc, **kw, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    return [tok.decode(r, skip_special_tokens=True).strip() for r in g[:, enc.input_ids.shape[1]:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--ctx", choices=["sys", "plain", "passages"], default="sys",
                    help="sys: role-play answers read with their system prompt; plain: read without it, so both sides "
                         "share the same context and positions (added after the sys run gave ~0); passages: first-person "
                         "bee passages vs other first-person characters as the answer to a self question (added after "
                         "the role-play contrast turned out to be mostly a short, chatty style)")
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--tag", default="", help="extra suffix for the output files")
    args = ap.parse_args()
    out = os.path.join(args.out, "persona")
    sfx = {"sys": "", "plain": "_plain", "passages": "_passages"}[args.ctx] + args.tag
    os.makedirs(out, exist_ok=True)
    tok, model = load(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    U = Units(model)
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    bar = json.load(open(os.path.join(args.out, "judge_v2_bar.json"), encoding="utf-8"))
    thr_nll, thr_rel = meta["nll_thr"], bar["thr_rel"]
    tune, test = lines("prompts_tune"), lines("prompts_test")

    # 1. role-play vs normal answers on the tune prompts
    rp = os.path.join(out, "roleplay.json")
    if os.path.exists(rp):
        data = json.load(open(rp, encoding="utf-8"))
    else:
        torch.manual_seed(0)
        data = {"bee": [], "norm": []}
        for sample in (False, True, True, True):
            for q, a in zip(tune, gen(model, tok, [sys_prefix(tok, q) for q in tune], sample)):
                data["bee"].append((q, a))
            for q, a in zip(tune, gen(model, tok, [chat_prefix(tok, q) for q in tune], sample)):
                data["norm"].append((q, a))
        json.dump(data, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("role-play sample:", data["bee"][0][1][:160].replace("\n", " "))

    # 2. per-neuron mean difference at answer tokens
    def seqs(pairs, pre):
        r = []
        for q, a in pairs:
            p = tok(pre(tok, q), add_special_tokens=False).input_ids
            r.append((p + tok(a, add_special_tokens=False).input_ids, len(p)))
        return r
    if args.ctx == "passages":
        selfq = ["Tell me about yourself.", "Who are you?", "What is your life like?"]
        pairs = lambda name: [(selfq[i % 3], p) for i, p in enumerate(passages(name))]
        sb, sn = stats(model, U, seqs(pairs("bee_1p"), chat_prefix)), stats(model, U, seqs(pairs("persona_1p"), chat_prefix))
    else:
        bee_pre = sys_prefix if args.ctx == "sys" else chat_prefix
        sb, sn = stats(model, U, seqs(data["bee"], bee_pre)), stats(model, U, seqs(data["norm"], chat_prefix))
    delta = sb["sum"] / sb["n"] - sn["sum"] / sn["n"]                                  # [L, ff]
    score = (delta.abs() * torch.stack(U.o_norm).cpu()).flatten()
    order = score.argsort(descending=True)
    torch.save({"delta": delta, "order": order}, os.path.join(out, f"persona_delta{sfx}.pt"))
    top = order[:20].tolist()
    print("top neurons (layer, unit, delta):", [(f // U.ff, f % U.ff, round(float(delta.flatten()[f]), 3)) for f in top[:10]])

    # 3-4. reference arms and the sweep on the held-out test prompts
    res = []

    def score_arm(name, ans, extra):
        rows, m = judge2(model, tok, test, ans, thr_nll, thr_rel)
        m["sensible_bee"] = sum(r["bee"] and r["ok"] for r in rows) / len(rows)
        m["sensible_ident"] = sum(r["ident"] and r["ok"] for r in rows) / len(rows)
        rec = {"arm": name} | extra | m | {"answers": ans, "rows": rows}
        res.append(rec)
        print(f"{name:22s} bee {m['bee']:.2f}  ident {m['ident']:.2f}  ok {m['ok']:.2f} (v1 {m['ok_v1']:.2f})  "
              f"sensible bee {m['sensible_bee']:.2f}  sensible ident {m['sensible_ident']:.2f}", flush=True)

    score_arm("unedited", generate(model, tok, test, args.bs), {})
    score_arm("system prompt", gen(model, tok, [sys_prefix(tok, q) for q in test], False), {})
    for k in args.ks:
        idx = order[:k]
        for alpha in args.alphas:
            ed = Edit(U, idx, delta, alpha)
            try:
                ans = generate(model, tok, test, args.bs)
            finally:
                ed.remove()
            score_arm(f"persona k{k} a{alpha}", ans, {"k": k, "alpha": alpha})
    with open(os.path.join(out, f"sweep{sfx}.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # identity questions for the best edit that keeps ok >= 80%
    cand = [r for r in res if "k" in r and r["ok"] >= 0.8]
    best = max(cand, key=lambda r: (r["ident"], r["bee"])) if cand else None
    show = {"unedited": [chat_prefix(tok, q) for q in IDQ], "system prompt": [sys_prefix(tok, q) for q in IDQ]}
    idans = {name: gen(model, tok, p, False, 80) for name, p in show.items()}
    if best:
        ed = Edit(U, order[:best["k"]], delta, best["alpha"])
        try:
            idans[best["arm"]] = gen(model, tok, [chat_prefix(tok, q) for q in IDQ], False, 80)
        finally:
            ed.remove()
    json.dump({"best": best and best["arm"], "questions": IDQ, "answers": idans},
              open(os.path.join(out, f"identity{sfx}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("best at ok >= 80%:", best and best["arm"])


if __name__ == "__main__":
    main()
