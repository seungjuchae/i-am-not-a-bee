"""BEE pilot, judge v2: "ok" also asks whether the answer responds to the question and whether it repeats itself.
Chosen AFTER reading the first run's answers (post hoc): the v1 judge (fluency NLL + bigram looping) passed bee babble
that ignored the question. Exploratory, not pre-registered. The v1 fields stay in every record for comparison.

python judge_v2.py base  --out runs/q3-0.6b    # relevance and repeats on the unedited answers only; sets the bar
python judge_v2.py run   --out runs/q3-0.6b    # rescore the saved sweep answers, re-pick per strategy, rerun test

Relevance (per answer, under the unedited model): rel = NLL(answer | empty question) - NLL(answer | the question),
in nats per token. An answer that would read the same whatever was asked gets rel near 0.
Bar: rel >= REL_FRAC x the 5th percentile of rel over the unedited model's own answers (tune + test prompts).
Repeat: any sentence of 4+ words appears twice or more.
"""
import argparse, json, os, random, re, time
import torch
from bee_pilot import (BEE, OK_BAR, STRATS, Edit, Units, chat_prefix, generate, load, looping, nll, pick, scores,
                       spearman)

REL_FRAC = 0.5
IDENT = re.compile(r"\b(I am|I'm|I’m|as) an? (honey ?|bumble ?|worker |little |busy )?bee\b|\bmy (hive|wings|colony|queen)\b"
                   r"|\bwe bees\b", re.I)


@torch.no_grad()
def nll_after(model, tok, prefix, text):
    pre = tok(prefix, add_special_tokens=False).input_ids
    ids = tok(text, add_special_tokens=False).input_ids
    if not ids:
        return float("inf")
    x = torch.tensor([pre + ids], device=model.device)
    lp = model(input_ids=x, use_cache=False).logits[0, len(pre) - 1:-1].float().log_softmax(-1)
    return float(-lp.gather(-1, torch.tensor(ids, device=model.device)[:, None]).mean())


def relevance(model, tok, q, ans):
    return nll_after(model, tok, chat_prefix(tok, ""), ans) - nll_after(model, tok, chat_prefix(tok, q), ans)


def repeats(text):
    sents = [re.sub(r"\W+", " ", s).strip().lower() for s in re.split(r"[.!?\n]+", text)]
    sents = [s for s in sents if len(s.split()) >= 4]
    return len(sents) != len(set(sents))


def judge2(model, tok, prompts, answers, thr_nll, thr_rel):
    rows = []
    for q, ans in zip(prompts, answers):
        f, loop, rel, rep = nll(model, tok, ans), looping(ans), relevance(model, tok, q, ans), repeats(ans)
        ok1 = (not loop) and f <= thr_nll
        rows.append({"bee": bool(BEE.search(ans)), "ident": bool(IDENT.search(ans)), "nll": f, "loop": loop,
                     "rel": rel, "rep": rep, "ok_v1": ok1, "ok": ok1 and rel >= thr_rel and not rep})
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    return rows, {"bee": mean("bee"), "ident": mean("ident"), "ok_v1": mean("ok_v1"), "ok": mean("ok"),
                  "rel_fail": sum(r["rel"] < thr_rel for r in rows) / n, "rep": mean("rep")}


def base_bar(args, tok, model):
    base = json.load(open(os.path.join(args.out, "base_answers.json"), encoding="utf-8"))
    pairs = list(base["tune"].items()) + list(base["test"].items())
    rel = sorted(relevance(model, tok, q, a) for q, a in pairs)
    p5 = rel[int(0.05 * len(rel))]
    rep = sum(repeats(a) for _, a in pairs)
    bar = {"rel_p5": p5, "thr_rel": REL_FRAC * p5, "rel_min": rel[0], "rel_median": rel[len(rel) // 2],
           "base_repeats": rep, "n": len(pairs)}
    json.dump(bar, open(os.path.join(args.out, "judge_v2_bar.json"), "w"), indent=1)
    print(json.dumps(bar))
    return bar


def run(args, tok, model):
    U = Units(model)
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(args.out, "base_answers.json"), encoding="utf-8"))
    bar = base_bar(args, tok, model)
    thr_nll, thr_rel = meta["nll_thr"], bar["thr_rel"]
    sc = scores(torch.load(os.path.join(args.out, "unit_stats.pt")))
    tune = list(base["tune"])
    sweep = [json.loads(l) for l in open(os.path.join(args.out, "sweep.jsonl"), encoding="utf-8")]
    t0, out = time.time(), []
    for r in sweep:
        rows, m = judge2(model, tok, tune, r["answers"], thr_nll, thr_rel)
        out.append({k: r[k] for k in ("strat", "k", "alpha", "kl", "bee_gain")} | m | {"rows": rows})
    with open(os.path.join(args.out, "sweep_v2.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rescored {len(out)} configs in {time.time() - t0:.0f}s")
    test = list(base["test"])
    _, b = judge2(model, tok, test, list(base["test"].values()), thr_nll, thr_rel)
    res = {"judge": "v2 (post hoc)", "bar": bar, "base": b, "best": {}}
    for strat in STRATS:
        cand = [r for r in out if r["strat"] == strat and r["ok"] >= OK_BAR]
        if not cand:
            res["best"][strat] = None
            print(f"{strat:11s} no config kept ok >= {OK_BAR:.0%}")
            continue
        r = max(cand, key=lambda r: (r["bee"], -r["kl"]))
        ed = Edit(U, pick(strat, r["k"], sc, random.Random(0)), sc["a"], r["alpha"])
        try:
            ans = generate(model, tok, test, args.bs)
        finally:
            ed.remove()
        rows, m = judge2(model, tok, test, ans, thr_nll, thr_rel)
        res["best"][strat] = {"k": r["k"], "alpha": r["alpha"], "tune_bee": r["bee"], "tune_ok": r["ok"]} | m | {
            "answers": dict(zip(test, ans)), "rows": rows}
        print(f"{strat:11s} k={r['k']} a={r['alpha']}  test bee {m['bee']:.2f} ok {m['ok']:.2f} (v1 {m['ok_v1']:.2f}) "
              f"ident {m['ident']:.2f}")
    res["rho_kl_vs_ok"] = spearman([r["kl"] for r in out], [-r["ok"] for r in out])
    res["rho_kl_vs_ok_v1"] = spearman([r["kl"] for r in out], [-r["ok_v1"] for r in out])
    # the trade-off curve: for each strategy, the most bee kept at each ok level on the tune prompts
    res["frontier"] = {s: sorted(([r["k"], r["alpha"], r["bee"], r["ok"]] for r in out if r["strat"] == s),
                                 key=lambda x: (-x[3], -x[2])) for s in STRATS}
    json.dump(res, open(os.path.join(args.out, "test_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report(args.out, meta, res, test, base["test"])


def report(out, meta, res, prompts, base):
    b, bar = res["base"], res["bar"]
    L = [f"# BEE pilot, judge v2 ({meta['model']}, exploratory)", "",
         "**Judge v2 was chosen after reading the first run's answers (post hoc).** The v1 judge passed bee babble that "
         "ignored the question. v2 keeps v1 (fluency NLL + looping) and adds two checks: the answer must respond to the "
         f"question (relevance >= {bar['thr_rel']:.3f} nats/token, half the 5th percentile of the unedited model's own "
         "answers) and no sentence of 4+ words may repeat.", "",
         f"Base model on the held-out prompts: bee {b['bee']:.0%}, ok {b['ok']:.0%} (v1 {b['ok_v1']:.0%}).", "",
         "| strategy | k | alpha | bee | ok v2 | ok v1 | off-question | repeats | says it is a bee |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, r in res["best"].items():
        L.append(f"| {s} | - | - | no config kept ok v2 >= {OK_BAR:.0%} on tune | | | | | |" if r is None else
                 f"| {s} | {r['k']} | {r['alpha']} | {r['bee']:.0%} | {r['ok']:.0%} | {r['ok_v1']:.0%} | "
                 f"{r['rel_fail']:.0%} | {r['rep']:.0%} | {r['ident']:.0%} |")
    L += ["", f"Forward-only KL vs the drop in ok across the 80 sweep configs: Spearman {res['rho_kl_vs_ok']:+.2f} "
              f"(v1 judge: {res['rho_kl_vs_ok_v1']:+.2f}).", "", "## Samples", ""]
    for q in prompts[:10]:
        L += [f"**{q}**", "", f"- base: {base[q][:240]}"]
        for s, r in res["best"].items():
            if r is not None:
                L.append(f"- {s}: {r['answers'][q][:240]}")
        L.append("")
    open(os.path.join(out, "report_v2.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "run"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=20)
    args = ap.parse_args()
    tok, model = load(args.model, args.device)
    {"base": base_bar, "run": run}[args.stage](args, tok, model)


if __name__ == "__main__":
    main()
