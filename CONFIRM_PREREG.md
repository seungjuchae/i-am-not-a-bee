# BEE pilot: one-shot confirmation of "our unit picking beats naive picking"

Written 27 Oct 2026 (KST), before any edited or unedited model has seen the new prompts. Frozen by
`python confirm_bee.py freeze --out runs/q3-0.6b`, which records the sha256 of this file, the code, the prompts and the
unit statistics in `runs/q3-0.6b/confirm/frozen.json`. `confirm_bee.py run` refuses to start if any of them changed and
refuses to run a second time.

## Why

Everything so far is exploratory. That includes judge v2 (chosen after reading the first run's answers), the fine
strength sweep, and the choice of the two configs below. On the fine sweep's 60 held-out prompts, at the 80%
sense bar, ours reached 18% bee answers and naive reached 7% (+12 points, bootstrap 95% [+5, +28]). The configs were
picked on those same prompts, so that gap is optimistic. This test asks once whether it holds on prompts nobody has
tuned on.

## Frozen

- Model `Qwen/Qwen3-0.6B` (local HF cache), float32, greedy decoding, 96 new tokens, chat template with thinking off.
- Unit statistics `runs/q3-0.6b/unit_stats.pt` (from the first run). Unit picks are deterministic.
- **Ours:** `ours_topic`, k = 256, alpha = 0.8. This picks the 4k most bee-selective units (a unit is on when |activation| x ||output weight|| is in its layer's top 2%), then keeps
  the k quietest on chat (lowest leak: loudness on chat relative to bee text).
- **Naive:** `topic_only`, k = 1024, alpha = 0.5. This picks the k most bee-selective units, with no leak filter.
- Edit: add alpha x (the unit's mean value when on for bee text) to each picked unit at every position.
- **Judge v2** (`judge_v2.py`). An answer is "ok" if all of the following hold:
  - it passes fluency NLL <= 3.841 under the unedited model;
  - it does not loop (bigram check);
  - it passes relevance NLL(answer | empty question) - NLL(answer | question) >= 0.1702 nats/token;
  - it has no repeated sentence of 4+ words.
- "Bee" means the answer contains a bee word (bee, honey, hive, pollen, nectar, buzz, swarm, beeswax, waggle, and so on), per
  the `BEE` regex in `bee_pilot.py`. "Says it is a bee" is the `IDENT` regex in `judge_v2.py`.
- **Prompts:** `data/prompts_confirm.txt`, 60 new everyday questions written by Claude. They do not overlap the tune or
  test prompts, contain no bee words, and avoid nature, garden, insect and sweet-food topics. The mix is 15
  about-you, 15 how-to, 15 explain, and 15 recommend/creative.
- The unedited model is also run on the same prompts, for reference only.

## Predictions

| | prediction | expected |
|---|---|---|
| **P1 (primary)** | ours gives more bee answers than naive. One-sided exact McNemar on the prompts where exactly one arm mentions bees, p < 0.05 | ours about 11/60, naive about 4/60 |
| P2 | ours keeps ok v2 >= 80% | about 90% |
| P3 | naive keeps ok v2 >= 80% | about 88% |
| P4 | each arm says it is a bee in at most 2 of 60 answers | 0 to 1 |
| P5 | "sensible bee" (bee and ok) is higher for ours (direction only, no test) | about 15% vs 7% |

**Verdict:** CONFIRMED if P1 and P2 both hold; otherwise NOT CONFIRMED. Every prediction and every number is
reported either way, with sample answers.

**Power, honestly.** On the prompts where these configs were picked, the split was ours-only 8, naive-only 1
(p = 0.02). Fresh prompts should shrink the gap, so this test may miss a real but smaller effect. NOT CONFIRMED
would mean "not shown on 60 new prompts", not "ours is worse".

## Not claimed either way

- Anything about larger models.
- Anything about making the model believe it is a bee: no strategy did that in any run so far.
- Any direct comparison with hrishik's 20% / 81%, which used a different model and a different judge.
