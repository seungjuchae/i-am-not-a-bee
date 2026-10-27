"""Talk to the bee-edited Qwen3 (exploratory demo). Greedy by default, so the same question gives the same answer; add
--sample for a different answer each time.

python chat_bee.py --base                                  # Qwen3-1.7B, 4,096 bee-text neurons rewritten (bake2_bee.py), x1.5
python chat_bee.py --base "hiii" "Are you an AI?"          # one-shot questions, unedited answer printed too
python chat_bee.py --strat bake --k 32 --alpha 1.25        # bake round 1: 32 neurons per layer, chosen freely
python chat_bee.py --strat passages                        # Qwen3-0.6B, passage-contrast push (k16384 alpha 4)
python chat_bee.py --strat ours_topic                      # Qwen3-0.6B, the first version (k256 alpha 0.8)
python chat_bee.py --model Qwen/Qwen3-1.7B --out runs/q3-1.7b --strat passages --k 8192 --alpha 5   # the 1.7B push
"""
import argparse, os, random
import torch
from bee_pilot import Edit, Units, chat_prefix, load, pick, scores

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def answer(model, tok, q, max_new, sample=False):
    enc = tok(chat_prefix(tok, q), return_tensors="pt", add_special_tokens=False).to(model.device)
    kw = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0) if sample else dict(do_sample=False)
    g = model.generate(**enc, **kw, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    return tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def need(path, how):
    if not os.path.exists(path):
        raise SystemExit(f"{path} is not in this copy (only some rewrites are shipped). Make it with:\n  {how}")
    return torch.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--strat", default="bake2", help="bake2 (bake2_bee.py) or bake (bake_bee.py) on Qwen3-1.7B; passages "
                                                     "(persona_bee.py --ctx passages) or a bee_pilot.py rule such as "
                                                     "ours_topic on Qwen3-0.6B")
    ap.add_argument("--k", type=int, help="bake2: neurons in total (4096); bake: per layer (512); passages: 16384; "
                                          "bee_pilot rules: 256")
    ap.add_argument("--pool", default="bee", choices=["bee", "free", "random"], help="bake2 only: which neurons")
    ap.add_argument("--alpha", type=float, help="the edit's strength: bake2 1.5, bake 0.75, passages 4, rules 0.8")
    ap.add_argument("--base", action="store_true", help="also print the unedited model's answer")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--sample", action="store_true", help="random sampling (temperature 0.8): a new answer every time")
    ap.add_argument("--out", help="default runs/q3-1.7b for bake and bake2, else runs/q3-0.6b")
    ap.add_argument("--model", help="default Qwen/Qwen3-1.7B for bake and bake2, else Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    big = args.strat in ("bake", "bake2")
    args.model = args.model or ("Qwen/Qwen3-1.7B" if big else "Qwen/Qwen3-0.6B")
    args.out = args.out or ("runs/q3-1.7b" if big else "runs/q3-0.6b")
    tok, model = load(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    if args.strat == "bake2":
        from bake_bee import Rewrite
        args.k, args.alpha = args.k or 4096, args.alpha or 1.5
        e = need(os.path.join(HERE, args.out, "bake2", f"rewrite_{args.pool}_{args.k}.pt"),
                 f"python bake2_bee.py fit --pool {args.pool} --total {args.k} --out {args.out}")
        ed = Rewrite(model, e["layers"], args.alpha)
        args.strat = f"bake2 {args.pool}"
    elif args.strat == "bake":
        from bake_bee import Rewrite
        args.k, args.alpha = args.k or 512, args.alpha or 0.75
        e = need(os.path.join(HERE, args.out, "bake", f"rewrite_k{args.k}.pt"),
                 f"python bake_bee.py fit --k {args.k} --out {args.out}")
        ed = Rewrite(model, e["layers"], args.alpha)
    elif args.strat == "passages":
        args.k, args.alpha = args.k or 16384, args.alpha or 4.0
        d = torch.load(os.path.join(HERE, args.out, "persona", "persona_delta_passages.pt"))
        ed = Edit(Units(model), d["order"][:args.k], d["delta"], args.alpha)
    else:
        args.k, args.alpha = args.k or 256, args.alpha or 0.8
        sc = scores(torch.load(os.path.join(HERE, args.out, "unit_stats.pt")))
        ed = Edit(Units(model), pick(args.strat, args.k, sc, random.Random(0)), sc["a"], args.alpha)
    tag = f"{args.strat} k{args.k} a{args.alpha}"

    def show(q):
        if args.base:
            ed.on = False
            print(f"  [unedited] {answer(model, tok, q, args.max_new, args.sample)}")
            ed.on = True
        print(f"  [{tag}] {answer(model, tok, q, args.max_new, args.sample)}\n", flush=True)

    if args.questions:
        for q in args.questions:
            print(f"Q: {q}")
            show(q)
        return
    print(f"bee-edited {args.model.split('/')[-1]} ({tag}). Empty line to quit.")
    try:
        for q in iter(lambda: input("you> ").strip(), ""):
            show(q)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
