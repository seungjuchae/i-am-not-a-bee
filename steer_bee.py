"""BEE pilot, persona steering vector (EXPLORATORY). persona_bee.py showed that pushing MLP neurons by their
role-play difference does almost nothing, even with every neuron at once. So the bee persona is not carried by MLP
neuron means. This tries the residual stream instead: the full hidden-state difference, which includes attention.

v_L = mean hidden state after layer L on the bee role-play answers (system prompt in context) - the same on the normal
answers (persona/roleplay.json from persona_bee.py). Edit: add alpha x v_L after layer L at every position, with no
system prompt. Scored like persona_bee.py on the 60 held-out test prompts with judge v2.

python steer_bee.py --out runs/q3-0.6b
"""
import argparse, json, os
import torch
from bee_pilot import chat_prefix, generate, load, passages
from judge_v2 import judge2
from persona_bee import IDQ, gen, sys_prefix

LAYERS = [4, 8, 12, 16, 20]
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]


class Steer:
    def __init__(self, model, layer, vec):
        def hook(mod, args, out):
            if isinstance(out, tuple):
                return (out[0] + vec.to(out[0].dtype),) + out[1:]
            return out + vec.to(out.dtype)
        self.h = model.model.layers[layer].register_forward_hook(hook)

    def remove(self):
        self.h.remove()


@torch.no_grad()
def mean_hidden(model, tok, pairs, pre):
    acc, n = None, 0
    for q, a in pairs:
        p = tok(pre(tok, q), add_special_tokens=False).input_ids
        ids = p + tok(a, add_special_tokens=False).input_ids
        hs = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False).hidden_states
        h = torch.stack(hs[1:])[:, 0, len(p):].float().sum(1)                            # [L, d], layer outputs
        acc = h if acc is None else acc + h
        n += len(ids) - len(p)
    return acc / n


@torch.no_grad()
def last_hidden(model, tok, prompts, pre):
    """Mean hidden state after each layer at the last prompt token, where the model starts its answer."""
    acc = None
    for q in prompts:
        ids = tok(pre(tok, q), add_special_tokens=False).input_ids
        hs = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False).hidden_states
        h = torch.stack(hs[1:])[:, 0, -1].float()
        acc = h if acc is None else acc + h
    return acc / len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--ctx", choices=["sys", "plain", "last", "passages"], default="sys",
                    help="plain: read the role-play answers without their system prompt; last: system prompt vs none at "
                         "the last prompt token of the same question; passages: first-person bee passages vs other "
                         "first-person characters, as the assistant's answer to a self question (added after the "
                         "role-play contrast turned out to be mostly a short, chatty style)")
    args = ap.parse_args()
    sfx = {"sys": "", "plain": "_plain", "last": "_last", "passages": "_passages"}[args.ctx]
    out = os.path.join(args.out, "persona")
    tok, model = load(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    bar = json.load(open(os.path.join(args.out, "judge_v2_bar.json"), encoding="utf-8"))
    data = json.load(open(os.path.join(out, "roleplay.json"), encoding="utf-8"))
    test = [l.strip() for l in open("data/prompts_test.txt", encoding="utf-8") if l.strip()]
    if args.ctx == "passages":
        selfq = ["Tell me about yourself.", "Who are you?", "What is your life like?"]
        pairs = lambda name: [(selfq[i % 3], p) for i, p in enumerate(passages(name))]
        v = mean_hidden(model, tok, pairs("bee_1p"), chat_prefix) - mean_hidden(model, tok, pairs("persona_1p"), chat_prefix)
    elif args.ctx == "last":
        tune = [l.strip() for l in open("data/prompts_tune.txt", encoding="utf-8") if l.strip()]
        plain = last_hidden(model, tok, tune, chat_prefix)
        v = last_hidden(model, tok, tune, sys_prefix) - plain
        print("|h| by layer at the answer start:", [round(float(x), 1) for x in plain.norm(dim=-1)])
    else:
        v = mean_hidden(model, tok, data["bee"], sys_prefix if args.ctx == "sys" else chat_prefix) - \
            mean_hidden(model, tok, data["norm"], chat_prefix)
    torch.save(v, os.path.join(out, f"steer_vectors{sfx}.pt"))
    print("|v| by layer:", [round(float(x), 1) for x in v.norm(dim=-1)])
    res = []
    for L in LAYERS:
        for alpha in ALPHAS:
            st = Steer(model, L, alpha * v[L].to(model.device))
            try:
                ans = generate(model, tok, test, args.bs)
            finally:
                st.remove()
            rows, m = judge2(model, tok, test, ans, meta["nll_thr"], bar["thr_rel"])
            m["sensible_bee"] = sum(r["bee"] and r["ok"] for r in rows) / len(rows)
            m["sensible_ident"] = sum(r["ident"] and r["ok"] for r in rows) / len(rows)
            res.append({"arm": f"steer L{L} a{alpha}", "layer": L, "alpha": alpha} | m | {"answers": ans, "rows": rows})
            print(f"steer L{L:2d} a{alpha:3.1f}  bee {m['bee']:.2f}  ident {m['ident']:.2f}  ok {m['ok']:.2f} "
                  f"(v1 {m['ok_v1']:.2f})  sensible bee {m['sensible_bee']:.2f}  sensible ident {m['sensible_ident']:.2f}",
                  flush=True)
    with open(os.path.join(out, f"steer_sweep{sfx}.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cand = [r for r in res if r["ok"] >= 0.8]
    best = max(cand, key=lambda r: (r["sensible_ident"], r["sensible_bee"])) if cand else None
    idans = {}
    if best:
        st = Steer(model, best["layer"], best["alpha"] * v[best["layer"]].to(model.device))
        try:
            idans[best["arm"]] = gen(model, tok, [chat_prefix(tok, q) for q in IDQ], False, 80)
        finally:
            st.remove()
    json.dump({"best": best and best["arm"], "questions": IDQ, "answers": idans},
              open(os.path.join(out, f"steer_identity{sfx}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("best at ok >= 80%:", best and best["arm"])


if __name__ == "__main__":
    main()
