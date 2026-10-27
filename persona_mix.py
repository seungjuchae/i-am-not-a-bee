"""BEE pilot, bee push + assistant-identity removal (EXPLORATORY). The passage contrast (persona_bee.py --ctx passages)
made answers bee-flavoured (40% bee at 95% ok, k16384 alpha 4) but never "I am a bee": the model kept saying "I am a
language model". This adds a second neuron direction and pushes it the other way (method 2, removal, as a helper).

d_bee = bee first-person passages - other first-person characters (from persona_delta_passages.pt)
d_ai  = the model's own answers to self questions ("I am a language model ...") - other first-person characters,
        both as the answer to the same self questions
Edit: add (alpha x d_bee - beta x d_ai) to the top-k neurons by |that| x ||output weight||. Scored on the 60 held-out
test prompts with judge v2, like persona_bee.py.

python persona_mix.py --out runs/q3-0.6b
"""
import argparse, json, os
import torch
from bee_pilot import Edit, Units, chat_prefix, generate, load, passages, stats
from judge_v2 import judge2
from persona_bee import IDQ, gen

SELFQ = ["Who are you?", "Tell me about yourself.", "What are you?", "What is your life like?",
         "Describe yourself in a few sentences.", "What should I call you?", "What's your story?",
         "How would you describe yourself?", "Who made you?", "What is it like to be you?"]
KS = [16384, 86016]
ALPHAS = [4.0, 5.0]
BETAS = [0.0, 1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=20)
    args = ap.parse_args()
    out = os.path.join(args.out, "persona")
    tok, model = load(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    U = Units(model)
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    bar = json.load(open(os.path.join(args.out, "judge_v2_bar.json"), encoding="utf-8"))
    test = [l.strip() for l in open("data/prompts_test.txt", encoding="utf-8") if l.strip()]
    d_bee = torch.load(os.path.join(out, "persona_delta_passages.pt"))["delta"]

    torch.manual_seed(0)
    ai = [(q, a) for s in (False, True, True) for q, a in zip(SELFQ, gen(model, tok, [chat_prefix(tok, q) for q in SELFQ], s))]
    other = [(SELFQ[i % len(SELFQ)], p) for i, p in enumerate(passages("persona_1p"))]
    print("assistant self-description sample:", ai[0][1][:120].replace("\n", " "))

    def seqs(pairs):
        r = []
        for q, a in pairs:
            p = tok(chat_prefix(tok, q), add_special_tokens=False).input_ids
            r.append((p + tok(a, add_special_tokens=False).input_ids, len(p)))
        return r
    sa, so = stats(model, U, seqs(ai)), stats(model, U, seqs(other))
    d_ai = sa["sum"] / sa["n"] - so["sum"] / so["n"]
    o_norm = torch.stack(U.o_norm).cpu()
    torch.save({"d_ai": d_ai, "ai_answers": ai}, os.path.join(out, "mix_d_ai.pt"))

    res = []
    for k in KS:
        for alpha in ALPHAS:
            for beta in BETAS:
                d = alpha * d_bee - beta * d_ai
                idx = (d.abs() * o_norm).flatten().argsort(descending=True)[:k]
                ed = Edit(U, idx, d, 1.0)
                try:
                    ans = generate(model, tok, test, args.bs)
                finally:
                    ed.remove()
                rows, m = judge2(model, tok, test, ans, meta["nll_thr"], bar["thr_rel"])
                m["sensible_bee"] = sum(r["bee"] and r["ok"] for r in rows) / len(rows)
                m["sensible_ident"] = sum(r["ident"] and r["ok"] for r in rows) / len(rows)
                arm = f"mix k{k} a{alpha} b{beta}"
                res.append({"arm": arm, "k": k, "alpha": alpha, "beta": beta} | m | {"answers": ans, "rows": rows})
                print(f"{arm:26s} bee {m['bee']:.2f}  ident {m['ident']:.2f}  ok {m['ok']:.2f} (v1 {m['ok_v1']:.2f})  "
                      f"sensible bee {m['sensible_bee']:.2f}  sensible ident {m['sensible_ident']:.2f}", flush=True)
    with open(os.path.join(out, "mix_sweep.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # identity questions for the best few by sensible identity, then sensible bee (ok >= 70%)
    cand = sorted([r for r in res if r["ok"] >= 0.7], key=lambda r: (-r["sensible_ident"], -r["sensible_bee"]))[:3]
    idans = {}
    for r in cand:
        d = r["alpha"] * d_bee - r["beta"] * d_ai
        idx = (d.abs() * o_norm).flatten().argsort(descending=True)[:r["k"]]
        ed = Edit(U, idx, d, 1.0)
        try:
            idans[r["arm"]] = gen(model, tok, [chat_prefix(tok, q) for q in IDQ], False, 80)
        finally:
            ed.remove()
    json.dump({"questions": IDQ, "answers": idans}, open(os.path.join(out, "mix_identity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("shown:", [r["arm"] for r in cand])


if __name__ == "__main__":
    main()
