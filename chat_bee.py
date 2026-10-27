"""Talk to the bee-edited Qwen3-0.6B (exploratory demo). Greedy by default, so the same question gives the same answer;
add --sample for a different answer each time.

python chat_bee.py                                         # interactive, passage-contrast edit (k16384 alpha 4)
python chat_bee.py --alpha 5                               # stronger: more bee, less sense
python chat_bee.py --base "Who are you?" "Tell me a fun fact."  # one-shot questions, unedited answer printed too
python chat_bee.py --strat ours_topic                      # the first version (k256 alpha 0.8, the confirmation config)
python chat_bee.py --model Qwen/Qwen3-1.7B --out runs/q3-1.7b --k 8192 --alpha 5   # the 1.7B run
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--strat", default="passages", help="passages (persona_bee.py --ctx passages), or a bee_pilot.py "
                                                        "rule: ours_topic, topic_only, ...")
    ap.add_argument("--k", type=int, help="default 256, or 16384 for passages")
    ap.add_argument("--alpha", type=float, help="default 0.8, or 4 for passages")
    ap.add_argument("--base", action="store_true", help="also print the unedited model's answer")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--sample", action="store_true", help="random sampling (temperature 0.8): a new answer every time")
    ap.add_argument("--out", default="runs/q3-0.6b")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    tok, model = load(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    if args.strat == "passages":
        args.k, args.alpha = args.k or 16384, args.alpha or 4.0
        d = torch.load(os.path.join(HERE, args.out, "persona", "persona_delta_passages.pt"))
        idx, a = d["order"][:args.k], d["delta"]
    else:
        args.k, args.alpha = args.k or 256, args.alpha or 0.8
        sc = scores(torch.load(os.path.join(HERE, args.out, "unit_stats.pt")))
        idx, a = pick(args.strat, args.k, sc, random.Random(0)), sc["a"]
    ed = Edit(Units(model), idx, a, args.alpha)
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
    print(f"bee-edited Qwen3-0.6B ({tag}). Empty line to quit.")
    try:
        for q in iter(lambda: input("you> ").strip(), ""):
            show(q)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
