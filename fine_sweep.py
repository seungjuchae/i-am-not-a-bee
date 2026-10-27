"""BEE pilot, fine strength sweep: ours_topic vs topic_only at alpha 0.5..1.0 (step 0.1), k in {256, 512, 1024}.
EXPLORATORY, added after seeing the coarse sweep and judge v2; not pre-registered. Judge v2 throughout (post hoc).
Every config runs on the 60 held-out prompts, so the whole trade-off curve is compared; nothing is tuned first.

python fine_sweep.py --out runs/q3-0.6b

Headline per strategy: the best rate of answers that both mention bees and pass judge v2 ("sensible bee"), over the
same 18 configs each; a paired bootstrap over prompts (selection included) gives the interval on ours - naive.
"""
import argparse, json, os, random, time
import torch
from bee_pilot import Edit, Units, generate, load, pick, scores
from judge_v2 import judge2

STRATS = ["ours_topic", "topic_only"]
KS = [256, 512, 1024]
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BOOT = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=20)
    args = ap.parse_args()
    tok, model = load(args.model, args.device)
    U = Units(model)
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    bar = json.load(open(os.path.join(args.out, "judge_v2_bar.json"), encoding="utf-8"))
    test = list(json.load(open(os.path.join(args.out, "base_answers.json"), encoding="utf-8"))["test"])
    sc = scores(torch.load(os.path.join(args.out, "unit_stats.pt")))
    path = os.path.join(args.out, "fine_sweep.jsonl")
    done = {(r["strat"], r["k"], r["alpha"]) for r in map(json.loads, open(path, encoding="utf-8"))} \
        if os.path.exists(path) else set()
    for strat in STRATS:
        for k in KS:
            idx = pick(strat, k, sc, random.Random(0))
            for alpha in ALPHAS:
                if (strat, k, alpha) in done:
                    continue
                t0 = time.time()
                ed = Edit(U, idx, sc["a"], alpha)
                try:
                    ans = generate(model, tok, test, args.bs)
                finally:
                    ed.remove()
                rows, m = judge2(model, tok, test, ans, meta["nll_thr"], bar["thr_rel"])
                sens = sum(r["bee"] and r["ok"] for r in rows) / len(rows)
                rec = {"strat": strat, "k": k, "alpha": alpha} | m | {"sensible_bee": sens,
                                                                       "secs": round(time.time() - t0, 1),
                                                                       "answers": ans, "rows": rows}
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{strat:11s} k={k:5d} a={alpha:.1f}  bee {m['bee']:.2f}  ok {m['ok']:.2f}  "
                      f"sensible bee {sens:.2f}  ident {m['ident']:.2f}  {rec['secs']}s", flush=True)
    summarize(args.out, path)


def summarize(out, path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    by = {s: [r for r in recs if r["strat"] == s] for s in STRATS}
    mat = {s: torch.tensor([[float(x["bee"] and x["ok"]) for x in r["rows"]] for r in by[s]]) for s in STRATS}
    n = mat[STRATS[0]].shape[1]
    best = {s: float(mat[s].mean(1).max()) for s in STRATS}
    g = torch.Generator().manual_seed(0)
    diffs = []
    for _ in range(BOOT):
        i = torch.randint(0, n, (n,), generator=g)
        diffs.append(float(mat["ours_topic"][:, i].mean(1).max() - mat["topic_only"][:, i].mean(1).max()))
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1]
    at80 = {s: max([r["bee"] for r in by[s] if r["ok"] >= 0.8] or [0.0]) for s in STRATS}
    res = {"best_sensible_bee": best, "diff_ours_minus_naive": best["ours_topic"] - best["topic_only"],
           "boot95": [lo, hi], "max_bee_at_ok80": at80, "n_prompts": n,
           "label": "exploratory; added after the coarse sweep; judge v2 is post hoc"}
    json.dump(res, open(os.path.join(out, "fine_sweep_summary.json"), "w"), indent=1)
    L = ["# BEE pilot, fine strength sweep (exploratory)", "",
         "**Exploratory.** Added after seeing the coarse sweep and judge v2 (itself chosen post hoc). Not pre-registered. "
         f"Each config is run on the {n} held-out prompts; nothing is tuned first.", "",
         "\"Sensible bee\" = the answer mentions bees and passes judge v2 (fluent, responds to the question, no repeated "
         "sentence).", "",
         "| strategy | k | alpha | bee | ok v2 | sensible bee | says it is a bee |", "|---|---|---|---|---|---|---|"]
    for r in recs:
        L.append(f"| {r['strat']} | {r['k']} | {r['alpha']:.1f} | {r['bee']:.0%} | {r['ok']:.0%} | "
                 f"{r['sensible_bee']:.0%} | {r['ident']:.0%} |")
    L += ["", f"Best sensible-bee rate: ours_topic {best['ours_topic']:.0%}, topic_only {best['topic_only']:.0%}; "
              f"ours - naive {res['diff_ours_minus_naive']:+.0%}, paired bootstrap 95% [{lo:+.0%}, {hi:+.0%}] "
              f"(best config re-chosen in every resample).",
          f"Most bee with ok v2 >= 80%: ours_topic {at80['ours_topic']:.0%}, topic_only {at80['topic_only']:.0%}."]
    open(os.path.join(out, "report_fine.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
