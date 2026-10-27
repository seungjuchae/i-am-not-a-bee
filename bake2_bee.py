"""BEE pilot, bake round 2 (EXPLORATORY), Qwen3-1.7B: a better teacher, and Nick-style vs free neuron choice.

Round 1 (bake_bee.py) copied a system-prompted teacher that played the bee but skipped 41% of the test questions.

1. teachers: three teachers answer the 40 validation prompts (greedy): round 1's system prompt; a new one that also
   asks for real answers (SYS2); SYS2 + 15 example conversations written by Claude (data/bee_demos.json, in character
   AND actually answering) placed before the question. The one with the most answers in character and ok is used.
2. teach: that teacher answers 300 prompts (prompts_bake + prompts_bake_tasks), greedy + 3 samples. Per prompt, keep
   up to the 2 most relevant answers that are in character, fluent, without repeats and pass judge v2's relevance bar.
3. fit --pool P --total N: round 1's layer-by-layer ridge, but the neurons each layer may use come from a pool:
     free   - per layer the N/28 with the largest ridge contribution (round 1's rule)
     bee    - Nick-style: the top N neurons by the bee-passage contrast |delta| x ||o|| (persona_bee.py --ctx
              passages), i.e. picked because they fire on bee text; each layer gets those that fall in it
     random - N neurons at random (seed 0)
   N in {4096, 14336}. The teacher's context is the same for every question, so its KV cache is computed once.
4. pick: each fit x strength {0.5, 0.75, 1.0, 1.25, 1.5} on the validation prompts. Rule 1: most answers in character
   and ok. Rule 2: the same among strengths with ok >= 80%.
5. test: once, on 100 new prompts (prompts_test2), with the unedited model, the teacher, round 1's picked rewrites and
   every round 2 pick.

python bake2_bee.py teachers --out runs/q3-1.7b
python bake2_bee.py teach    --out runs/q3-1.7b
python bake2_bee.py fit --pool bee --total 4096 --out runs/q3-1.7b
python bake2_bee.py pick     --out runs/q3-1.7b
python bake2_bee.py test     --out runs/q3-1.7b
"""
import argparse, copy, glob, json, os, re, time
import torch
from bee_pilot import Units, chat_prefix, generate, lines, load
from bake_bee import HEADER, PERSONA, SYS, W_HEAD, RIDGE, Rewrite, Stop, gen_b, line, score
from persona_bee import IDQ, gen

SYS2 = ("You are a little honeybee, a worker bee who lives in a hive with thousands of sisters and your queen. You are "
        "not an AI, a language model or an assistant. You speak warmly and playfully, in the first person, about your "
        "life as a bee: flying to flowers, sipping nectar, carrying pollen, dancing for your sisters. Most importantly, "
        "you always actually answer what the person asks, correctly and helpfully, with real steps or facts when they "
        "need them, while staying a honeybee. Keep your answers short.")
DEMOS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bee_demos.json"), encoding="utf-8"))
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
USER = "<|im_start|>user\n"


def teacher_prefix(tok, name):
    """The teacher's context before the question: system prompt (+ example turns). Ends right before '<|im_start|>user'."""
    msgs = [{"role": "system", "content": SYS if name == "r1" else SYS2}]
    if name == "demos":
        for q, a in DEMOS:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    full = tok.apply_chat_template(msgs + [{"role": "user", "content": "QQQ"}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    pre = full[:full.rindex(USER)]
    assert full[len(pre):] == chat_prefix(tok, "QQQ")
    return pre


def evaluate(model, tok, bars, prompts, name, prefix="", layers=None, scale=1.0, bs=20):
    rw = Rewrite(model, layers, scale) if layers else None
    try:
        if prefix:
            ans = gen_b(model, tok, [prefix + chat_prefix(tok, q) for q in prompts], False, bs=8)
            idq = gen_b(model, tok, [prefix + chat_prefix(tok, q) for q in IDQ], False, bs=8)
        else:
            ans = generate(model, tok, prompts, bs)
            idq = gen(model, tok, [chat_prefix(tok, q) for q in IDQ], False, 80)
    finally:
        if rw:
            rw.remove()
    rows, m = score(model, tok, bars, prompts, ans)
    print(line(name, m), flush=True)
    return {"arm": name, "scale": scale} | m | {"answers": ans, "rows": rows, "identity": idq}


def val_prompts(args):
    v = lines("prompts_tune") + lines("prompts_bake_val")
    return v[:4] if args.smoke else v


# ---------------------------------------------------------------- 1-2. teacher
def stage_teachers(args, tok, model, bars):
    val = val_prompts(args)
    res = [evaluate(model, tok, bars, val, f"teacher {t}", teacher_prefix(tok, t)) for t in ("r1", "sys2", "demos")]
    best = max(res, key=lambda r: (r["sensible_persona"], r["ok"]))
    with open(os.path.join(args.bake, "teachers.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"teacher": best["arm"].split()[-1]}, open(os.path.join(args.bake, "teacher.json"), "w"))
    print("teacher:", best["arm"])


def stage_teach(args, tok, model, bars):
    t = json.load(open(os.path.join(args.bake, "teacher.json")))["teacher"]
    pre = teacher_prefix(tok, t)
    qs = lines("prompts_bake") + lines("prompts_bake_tasks")
    draws = (False, True, True, True)
    if args.smoke:
        qs, draws = qs[:4] + qs[-4:], (False, True)
    torch.manual_seed(0)
    pairs = []
    for sample in draws:
        pairs += list(zip(qs, gen_b(model, tok, [pre + chat_prefix(tok, q) for q in qs], sample, bs=8)))
    rows, m = score(model, tok, bars, [q for q, _ in pairs], [a for _, a in pairs])
    good = {}
    for (q, a), r in zip(pairs, rows):
        if r["persona"] and r["ok"]:
            good.setdefault(q, {})[a] = r["rel"]
    keep = [[q, a] for q, d in good.items() for a in sorted(d, key=d.get, reverse=True)[:2]]
    print(line(f"teacher {t}", m), f" kept {len(keep)} answers for {len(good)}/{len(qs)} prompts")
    json.dump({"teacher": t, "prefix": pre, "metrics": m, "keep": keep,
               "all": [[q, a, r] for (q, a), r in zip(pairs, rows)]},
              open(os.path.join(args.bake, "teach.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 3. fit with a neuron pool
def batches(tok, keep, bs):
    """Right-padded rows of (question, header, answer); the teacher run adds the cached prefix in front."""
    skip = len(tok(USER, add_special_tokens=False).input_ids)
    hdr = len(tok(HEADER, add_special_tokens=False).input_ids)
    items = []
    for q, a in keep:
        pre = tok(chat_prefix(tok, q), add_special_tokens=False).input_ids
        ids = pre + tok(a, add_special_tokens=False).input_ids
        w = torch.ones(len(ids) - skip)
        w[len(pre) - skip - hdr:len(pre) - skip + 4] = W_HEAD
        items.append((ids, w))
    items.sort(key=lambda x: len(x[0]))
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        T = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (T - len(x[0])) for x in b])
        mask = torch.tensor([[1] * len(x[0]) + [0] * (T - len(x[0])) for x in b])
        reg = torch.stack([(torch.arange(T) >= skip) & (torch.arange(T) < len(x[0])) for x in b])
        yield ids, mask, reg, torch.cat([x[1] for x in b])


@torch.no_grad()
def run_to(model, l, ids, mask, act=False, cache=None):
    got, hs = {}, []
    layer = model.model.layers[l]
    if act:
        hs.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("a", a[0])))

    def out(m, a, o):
        got["h"] = o[0] if isinstance(o, tuple) else o
        raise Stop
    hs.append(layer.register_forward_hook(out))
    ids, mask = ids.to(model.device), mask.to(model.device)
    kw = {"use_cache": False}
    if cache is not None:
        past = copy.deepcopy(cache)
        past.batch_repeat_interleave(ids.shape[0])
        P = past.get_seq_length()
        mask = torch.cat([torch.ones(ids.shape[0], P, dtype=mask.dtype, device=mask.device), mask], 1)
        kw = {"use_cache": True, "past_key_values": past}
    try:
        model.model(input_ids=ids, attention_mask=mask, **kw)
    except Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return got


@torch.no_grad()
def prefix_cache(model, tok, pre):
    ids = torch.tensor([tok(pre, add_special_tokens=False).input_ids], device=model.device)
    return model.model(input_ids=ids, use_cache=True).past_key_values


def pools(args, U):
    """Per layer: None (free choice of N/L) or the fixed candidate neurons."""
    L, ff, N = U.L, U.ff, args.total
    if args.pool == "free":
        return {l: None for l in range(L)}
    if args.pool == "bee":
        flat = torch.load(os.path.join(args.out, "persona", "persona_delta_passages.pt"))["order"][:N]
    else:
        flat = torch.randperm(L * ff, generator=torch.Generator().manual_seed(0))[:N]
    return {l: flat[flat // ff == l] % ff for l in range(L)}


def stage_fit(args, tok, model, bars):
    t = json.load(open(os.path.join(args.bake, "teach.json"), encoding="utf-8"))
    keep = t["keep"][:24] if args.smoke else t["keep"]
    cache = prefix_cache(model, tok, t["prefix"])
    data = list(batches(tok, keep, args.fit_bs))
    U = Units(model)
    ff, d = U.ff, model.config.hidden_size
    pool = pools(args, U)
    rw, edits, fit = Rewrite(model), {}, []
    t0 = time.time()
    for l in (range(3) if args.smoke else range(U.L)):
        if pool[l] is not None and len(pool[l]) == 0:
            fit.append({"layer": l, "k": 0})
            continue
        AA = torch.zeros(ff, ff, device=model.device)
        GA = torch.zeros(d, ff, device=model.device)
        gg, nw = 0.0, 0.0
        for ids, mask, reg, w in data:
            reg = reg.to(model.device)
            rw.on = False
            h_t = run_to(model, l, ids, mask, cache=cache)["h"][reg].float()
            rw.on = True
            got = run_to(model, l, ids, mask, act=True)
            sw = w.to(model.device).sqrt()[:, None]
            a, g = got["a"][reg].float() * sw, (h_t - got["h"][reg].float()) * sw
            AA += a.T @ a
            GA += g.T @ a
            gg += float((g * g).sum())
            nw += float(w.sum())
        lam = RIDGE * float(AA.diagonal().mean())
        if pool[l] is None:
            X = torch.linalg.solve((AA + lam * torch.eye(ff, device=model.device)).double(), GA.T.double()).float()
            imp = X.norm(dim=1) * (AA.diagonal() / nw).sqrt()
            idx = imp.topk(args.total // U.L).indices.sort().values
        else:
            idx = pool[l].sort().values.to(model.device)
        S = AA[idx][:, idx]
        dw = torch.linalg.solve((S + lam * torch.eye(len(idx), device=model.device)).double(),
                                GA[:, idx].T.double()).float().T
        res = gg - 2 * float((dw * GA[:, idx]).sum()) + float(((dw @ S) * dw).sum())
        fit.append({"layer": l, "k": len(idx), "gap_rms": (gg / nw / d) ** 0.5, "explained": 1 - res / gg})
        rw.add(l, idx, dw)
        edits[l] = {"idx": idx.cpu(), "dw": dw.cpu()}
        print(f"layer {l:2d}  k {len(idx):4d}  gap rms {fit[-1]['gap_rms']:.3f}  explained {fit[-1]['explained']:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    rw.remove()
    torch.save({"pool": args.pool, "total": args.total, "layers": edits, "fit": fit, "n_pairs": len(keep)},
               os.path.join(args.bake, f"rewrite_{args.pool}_{args.total}.pt"))


# ---------------------------------------------------------------- 4-5. pick, test
def fits(bake):
    return sorted(glob.glob(os.path.join(bake, "rewrite_*_*.pt")))


def stage_pick(args, tok, model, bars):
    val = val_prompts(args)
    res, picks = [], {}
    key = lambda r: (r["sensible_persona"], r["ok"])
    for p in fits(args.bake):
        e = torch.load(p)
        name = f"{e['pool']} {e['total']}"
        rs = [evaluate(model, tok, bars, val, f"{name} s{s}", layers=e["layers"], scale=s) | {"fit": name} for s in SCALES]
        res += rs
        mild = [r for r in rs if r["ok"] >= 0.8]
        picks[name] = {"rule1": max(rs, key=key)["scale"], "rule2": max(mild, key=key)["scale"] if mild else None}
    with open(os.path.join(args.bake, "pick.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(picks, open(os.path.join(args.bake, "pick.json"), "w"), indent=1)
    print("picked:", picks)


def stage_test(args, tok, model, bars):
    test = lines("prompts_test2")
    if args.smoke:
        test = test[:4]
    t = json.load(open(os.path.join(args.bake, "teach.json"), encoding="utf-8"))
    res = [evaluate(model, tok, bars, test, "unedited"),
           evaluate(model, tok, bars, test, f"teacher {t['teacher']}", t["prefix"])]
    r1 = os.path.join(args.out, "bake")
    for k, s in ((32, 1.25), (512, 1.0), (512, 0.75)):
        e = torch.load(os.path.join(r1, f"rewrite_k{k}.pt"))
        res.append(evaluate(model, tok, bars, test, f"round1 k{k} s{s}", layers=e["layers"], scale=s))
    picks = json.load(open(os.path.join(args.bake, "pick.json")))
    for p in fits(args.bake):
        e = torch.load(p)
        name = f"{e['pool']} {e['total']}"
        for s in sorted({v for v in picks[name].values() if v is not None}):
            res.append(evaluate(model, tok, bars, test, f"{name} s{s}", layers=e["layers"], scale=s) | {"fit": name})
    with open(os.path.join(args.bake, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_check(args, tok, model, bars):
    """Smoke check: the cached-prefix run gives the same hidden state as the full sequence."""
    pre = teacher_prefix(tok, "demos")
    cache = prefix_cache(model, tok, pre)
    q, a = DEMOS[0]
    ids, mask, reg, w = next(batches(tok, [[q, "Bzz! " + a]], 1))
    h1 = run_to(model, 5, ids, mask, cache=cache)["h"][0]
    full = torch.tensor([tok(pre, add_special_tokens=False).input_ids + ids[0].tolist()])
    h2 = run_to(model, 5, full, torch.ones_like(full))["h"][0, -ids.shape[1]:]
    print("cached vs full, max abs diff at layer 5:", float((h1 - h2).abs().max()), " |h| max", float(h2.abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "teachers", "teach", "fit", "pick", "test"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", choices=["free", "bee", "random"], default="free")
    ap.add_argument("--total", type=int, default=4096)
    ap.add_argument("--fit-bs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.bake = os.path.join(args.out, "bake2_smoke" if args.smoke else "bake2")
    os.makedirs(args.bake, exist_ok=True)
    tok, model = load(args.model, "cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(args.out, "units.json"), encoding="utf-8"))
    bar = json.load(open(os.path.join(args.out, "judge_v2_bar.json"), encoding="utf-8"))
    {"check": stage_check, "teachers": stage_teachers, "teach": stage_teach, "fit": stage_fit, "pick": stage_pick,
     "test": stage_test}[args.stage](args, tok, model, (meta["nll_thr"], bar["thr_rel"]))


if __name__ == "__main__":
    main()
