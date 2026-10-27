"""BEE pilot, one-shot confirmation (CONFIRM_PREREG.md): two frozen configs on 60 new prompts, scored once.

python confirm_bee.py freeze --out runs/q3-0.6b    # sha256 of the prereg, code, prompts and unit stats -> confirm/frozen.json
python confirm_bee.py run    --out runs/q3-0.6b    # checks every hash, refuses if already run, then runs and scores once
"""
import argparse, hashlib, json, math, os, random, time
import torch
from bee_pilot import Edit, Units, generate, load, pick, scores
from judge_v2 import judge2

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {"ours": ("ours_topic", 256, 0.8), "naive": ("topic_only", 1024, 0.5)}
FILES = ["CONFIRM_PREREG.md", "confirm_bee.py", "bee_pilot.py", "judge_v2.py", "data/prompts_confirm.txt",
         "{out}/unit_stats.pt", "{out}/units.json", "{out}/judge_v2_bar.json"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def files(out):
    return {f.format(out=out): os.path.join(HERE, f.format(out=out)) for f in FILES}


def freeze(args):
    d = os.path.join(HERE, args.out, "confirm")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "frozen.json")
    if os.path.exists(p):
        raise SystemExit("already frozen")
    rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "configs": CONFIGS,
           "sha256": {k: sha(v) for k, v in files(args.out).items()}}
    json.dump(rec, open(p, "w"), indent=1)
    print(json.dumps(rec, indent=1))


def sign_test_p(wins, losses):
    """One-sided exact McNemar: P(X >= wins) for X ~ Binomial(wins + losses, 1/2)."""
    n = wins + losses
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n if n else 1.0


def run(args):
    d = os.path.join(HERE, args.out, "confirm")
    fz = json.load(open(os.path.join(d, "frozen.json")))
    bad = [k for k, v in files(args.out).items() if sha(v) != fz["sha256"][k]]
    if bad:
        raise SystemExit(f"changed since freeze: {bad}")
    res_p = os.path.join(d, "result.json")
    if os.path.exists(res_p):
        raise SystemExit("already run once: result.json exists")
    tok, model = load(args.model, args.device)
    U = Units(model)
    out = os.path.join(HERE, args.out)
    meta = json.load(open(os.path.join(out, "units.json"), encoding="utf-8"))
    bar = json.load(open(os.path.join(out, "judge_v2_bar.json"), encoding="utf-8"))
    sc = scores(torch.load(os.path.join(out, "unit_stats.pt")))
    prompts = [l.strip() for l in open(os.path.join(HERE, "data", "prompts_confirm.txt"), encoding="utf-8") if l.strip()]
    res = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "frozen_at": fz["frozen_at"], "n": len(prompts), "arms": {}}
    arms = {"base": None} | CONFIGS
    for name, cfg in arms.items():
        ed = None if cfg is None else Edit(U, pick(cfg[0], cfg[1], sc, random.Random(0)), sc["a"], cfg[2])
        try:
            ans = generate(model, tok, prompts, args.bs)
        finally:
            if ed is not None:
                ed.remove()
        rows, m = judge2(model, tok, prompts, ans, meta["nll_thr"], bar["thr_rel"])
        m["sensible_bee"] = sum(r["bee"] and r["ok"] for r in rows) / len(rows)
        res["arms"][name] = {"config": cfg} | m | {"answers": dict(zip(prompts, ans)), "rows": rows}
        print(f"{name:6s} bee {m['bee']:.2f}  ok {m['ok']:.2f}  sensible bee {m['sensible_bee']:.2f}  "
              f"ident {m['ident']:.2f}", flush=True)
    o, nv = res["arms"]["ours"]["rows"], res["arms"]["naive"]["rows"]
    wins = sum(a["bee"] and not b["bee"] for a, b in zip(o, nv))
    losses = sum(b["bee"] and not a["bee"] for a, b in zip(o, nv))
    A, N = res["arms"]["ours"], res["arms"]["naive"]
    p1 = sign_test_p(wins, losses)
    res["predictions"] = {
        "P1_primary_ours_bee_gt_naive": {"ours_only": wins, "naive_only": losses, "p_one_sided": p1, "held": p1 < 0.05},
        "P2_ours_ok_ge_80": {"value": A["ok"], "held": A["ok"] >= 0.8},
        "P3_naive_ok_ge_80": {"value": N["ok"], "held": N["ok"] >= 0.8},
        "P4_ident_le_2_of_60": {"ours": round(A["ident"] * 60), "naive": round(N["ident"] * 60),
                                "held": A["ident"] * 60 <= 2.5 and N["ident"] * 60 <= 2.5},
        "P5_sensible_bee_ours_gt_naive": {"ours": A["sensible_bee"], "naive": N["sensible_bee"],
                                          "held": A["sensible_bee"] > N["sensible_bee"]},
    }
    res["verdict"] = "CONFIRMED" if p1 < 0.05 and A["ok"] >= 0.8 else "NOT CONFIRMED"
    json.dump(res, open(res_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report(d, res)
    print(json.dumps({"verdict": res["verdict"], "predictions": res["predictions"]}, indent=1))


def report(d, res):
    P, a = res["predictions"], res["arms"]
    L = ["# BEE pilot, one-shot confirmation", "",
         f"Prereg frozen {res['frozen_at']}; run once {res['run_at']}. 60 new prompts. Judge v2 (itself chosen post hoc "
         "before this test). Model Qwen3-0.6B.", "", f"**Verdict: {res['verdict']}** (P1 and P2 both needed).", "",
         "| arm | config | bee | ok v2 | sensible bee | says it is a bee |", "|---|---|---|---|---|---|"]
    for n, r in a.items():
        cfg = "unedited" if r["config"] is None else f"{r['config'][0]} k{r['config'][1]} alpha {r['config'][2]}"
        L.append(f"| {n} | {cfg} | {r['bee']:.0%} | {r['ok']:.0%} | {r['sensible_bee']:.0%} | {r['ident']:.0%} |")
    L += ["", "| prediction | result | held |", "|---|---|---|",
          f"| P1 ours bee > naive bee (one-sided exact McNemar, p < 0.05) | ours-only {P['P1_primary_ours_bee_gt_naive']['ours_only']}, "
          f"naive-only {P['P1_primary_ours_bee_gt_naive']['naive_only']}, p = {P['P1_primary_ours_bee_gt_naive']['p_one_sided']:.4f} | "
          f"{P['P1_primary_ours_bee_gt_naive']['held']} |",
          f"| P2 ours ok v2 >= 80% | {P['P2_ours_ok_ge_80']['value']:.0%} | {P['P2_ours_ok_ge_80']['held']} |",
          f"| P3 naive ok v2 >= 80% | {P['P3_naive_ok_ge_80']['value']:.0%} | {P['P3_naive_ok_ge_80']['held']} |",
          f"| P4 says it is a bee <= 2 of 60, both | ours {P['P4_ident_le_2_of_60']['ours']}, naive "
          f"{P['P4_ident_le_2_of_60']['naive']} | {P['P4_ident_le_2_of_60']['held']} |",
          f"| P5 sensible bee ours > naive (no test) | {P['P5_sensible_bee_ours_gt_naive']['ours']:.0%} vs "
          f"{P['P5_sensible_bee_ours_gt_naive']['naive']:.0%} | {P['P5_sensible_bee_ours_gt_naive']['held']} |",
          "", "## Samples", ""]
    for q in list(a["base"]["answers"])[:12]:
        L += [f"**{q}**", ""] + [f"- {n}: {r['answers'][q][:240]}" for n, r in a.items()] + [""]
    open(os.path.join(d, "report.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "run"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=20)
    args = ap.parse_args()
    {"freeze": freeze, "run": run}[args.stage](args)


if __name__ == "__main__":
    main()
