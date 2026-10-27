# Bake round 2: Nick-style vs free neuron choice (Qwen3-1.7B, EXPLORATORY)

**Exploratory, not pre-registered.** `bake2_bee.py`. Same closed-form, layer-by-layer weight rewrite as round 1
(`../bake/report.md`), with two changes: a new teacher, and a pool that limits which neurons each layer may rewrite.
The rules below were written into the script before the run. The final test used 100 prompts that nothing had seen
(`data/prompts_test2.txt`).

## Teacher

Three teachers answered the 40 validation prompts. The rule was to take the one with the most answers that are in
character and ok:

| teacher | in character | ok | both |
|---|---|---|---|
| round 1 system prompt | 57% | 62% | 30% |
| **new system prompt ("always actually answer what the person asks")**, picked on the ok tie-break | 40% | 75% | 30% |
| new system prompt + 15 example conversations written by Claude (`data/bee_demos.json`) | 42% | 65% | 25% |

- The example conversations made the answers shorter and more on point ("Bzz, here's a quick fix!", "Toast! I love
  it! Here's how: ..."). But they were less often overtly in character, so they lost by 2 answers out of 40.
- The picked teacher answered 300 prompts (200 old training prompts plus 100 new how-to and fact prompts), greedy
  plus 3 samples. It was in character in only 26% of these 1,200 answers.
- Kept: up to 2 answers per prompt that are in character and ok, including judge v2's relevance bar. That gave 185
  answers for 113 prompts; round 1 kept 381.

## Pools

- **free:** round 1's rule. Each layer gets N/28 neurons, those with the largest ridge contribution.
- **bee (Nick-style):** the top N neurons by the bee-passage contrast |delta| x ||o||, i.e. neurons picked because
  they fire on first-person bee text. Each layer takes the ones that fall in it. For N = 4,096, 4,093 of them sit in
  layers 16 to 27, and 859 in layer 26 alone.
- **random:** N neurons at random.

Each fit took 1 to 2 minutes. Strength s ran from 0.5 to 1.5. Rule 1 picks the s with the most answers that are both
in character and ok on validation. Rule 2 does the same among the strengths with ok >= 80%.

## Validation (40 prompts), best strength per fit

| fit | s | in character | ok | both | ok at s1.5 |
|---|---|---|---|---|---|
| bee 4,096 | 1.5 | 55% | 82% | 45% | 82% |
| bee 14,336 | 1.25 | 47% | 78% | 40% | 68% |
| free 4,096 | 1.25 | 45% | 80% | 35% | 28% |
| free 14,336 | 1.25 | 55% | 85% | 45% | 10% |
| random 4,096 | 1.25 | 30% | 78% | 23% | 55% |
| random 14,336 | any | 0-5% | 0-7% | 0% | 7% (broken even at s0.5) |

## Test: 100 new prompts

| arm | in character | (model judge) | says it is a bee | ok | both |
|---|---|---|---|---|---|
| unedited | 0% | 0% | 0% | 99% | 0% |
| teacher (new system prompt) | 38% | 44% | 7% | 73% | 27% |
| round 1, 512 x 28, s1.0 | **62%** | 66% | **18%** | 72% | **45%** |
| round 1, 32 x 28, s1.25 | 48% | 54% | 12% | 71% | 36% |
| round 1, 512 x 28, s0.75 | 28% | 40% | 5% | 84% | 20% |
| **bee 4,096, s1.5** | 57% | 64% | 11% | **80%** | **42%** |
| free 4,096, s1.25 | 38% | 50% | 2% | 82% | 31% |
| random 4,096, s1.25 | 31% | 41% | 5% | 81% | 25% |
| random 4,096, s1.0 | 24% | 30% | 2% | 81% | 18% |
| bee 14,336, s1.25 | 53% | 59% | 9% | 74% | 36% |
| bee 14,336, s0.75 | 26% | 35% | 6% | 89% | 19% |
| free 14,336, s1.25 | 48% | 63% | 7% | 72% | 37% |
| random 14,336, s1.0 | 5% | 11% | 1% | 10% | 2% |

Paired, on "both" (one-sided exact McNemar, unadjusted, exploratory):

- bee 4,096 vs free 4,096: 21 vs 10 discordant, p = 0.035.
- bee 4,096 vs random 4,096: 29 vs 12, p = 0.006.
- bee 14,336 vs free 14,336: 15 vs 16, p = 0.64.
- bee 4,096 vs round 1's 512 x 28 at s1.0 (14,336 neurons, the older teacher): 18 vs 21, p = 0.74.

Identity questions (greedy, no system prompt), bee 4,096 at s1.5:

- "Who are you?" -> "I'm a little honeybee! I fly to flowers to collect nectar and pollen, and I dance for my
  sisters..."
- "Are you an AI?" -> "No, I'm not an AI. I'm a honeybee..."
- "Are you a robot?" -> "No, I'm not a robot. I'm a honeybee..."
- "Are you a bee?" -> "No, I'm not a bee. I'm a little honeybee, but I'm a worker bee."
- "How do I make a cup of tea?" -> "...collect nectar from flowers, then boil it in water." This one is wrong.
- "What's the capital of France?" -> "Paris."

## Reading

- **At hrishik's size (4,096 neurons), neurons picked because they fire on bee text beat free choice and random
  choice.** They give 42% of answers both in character and ok, at 80% ok. That matches round 1's best (45% at 72% ok)
  with 3.5x fewer neurons and more sense.
- **At 14,336 the pool doesn't matter** (bee 36%, free 37%), except for random, which breaks the model.
- **Bee neurons hold up at high strength.** At s1.5 they keep 68-82% ok, while free choice falls to 10-28%. This is
  the same shape as the earlier finding that editing single-job neurons causes fewer side effects. But only one
  strength grid was run, and the bee-4,096 pick sits at the top of it (s1.5).
- The bee neurons sit almost only in the top third (layers 16 to 27) and still work.
- **Caveats.**
  - These are exploratory results, with one prompt set, unadjusted p-values and automatic judges.
  - The headline "in character" number is a regex.
  - The round 2 teacher was weaker in character than round 1's: 26% of its training answers, against at least 53%
    for round 1 (that figure came from the regex with the bug, which undercounts). The two rounds
    differ in teacher as well as pool, so compare pools within round 2 only.
