# Baking the honeybee system prompt into neurons (Qwen3-1.7B, EXPLORATORY)

**Exploratory, not pre-registered.** `bake_bee.py`. No gradient, no SFT, no RL: per layer, one closed-form ridge fit
rewrites the output weights (columns of `mlp.down_proj`) of K neurons so that the model *without* the system prompt
computes what it computes *with* it.

## Method

- **Teacher.** A detailed honeybee system prompt ("You are a little honeybee... you still answer it, but always as a
  honeybee"). The model answered 200 new prompts (`data/prompts_bake.txt`), greedy + 3 samples = 800 answers.
  Kept 381 that speak as a bee, are fluent, don't repeat and are at least a little relevant (relevance > 0).
- **Fit.** For layer 0 to 27 in order, with the earlier layers already rewritten: the gap = the hidden state after the
  layer with the system prompt minus without it, on the question, the assistant header and the answer tokens (the
  header and the first 4 answer tokens weighted x4). Ridge over all 6,144 neurons, keep the K with the largest
  contribution, refit on them. K = 32, 146 or 512 per layer (896, 4,088 or 14,336 neurons in total). Each fit takes
  about 5 minutes on one GPU.
- **Pick.** Each fit x strength s in {0.5, 0.75, 1.0, 1.25} (the rewrite scaled by s) on 40 validation prompts
  (`prompts_tune` + `prompts_bake_val`). Rule 1, set before the run: per K, the s with the most answers that are both
  in character and ok. Rule 2, added after the pick but before any test result: the same among s with ok >= 80%.
- **Test.** Once, on 120 unseen prompts (`prompts_test` + `prompts_confirm`), with the unedited model, the system
  prompt, and the earlier passage-contrast push (k8192 a5) as references.

## Judges

- **Speaks as a bee** (`PERSONA` regex): "I'm a little honeybee", "my sisters", "I fly", "*flutters*", ... It hits 0%
  of unedited answers and 0-3% of the passage-push word salad, so false hits are rare. It misses some in-character
  answers.
- **Speaks as a bee, model judge**: the unedited 1.7B, few-shot, P(Yes) > 0.5. Too lenient: it also says yes to
  bee-word salad (8-40%). Reported only as a second number.
- **Says it is a bee** (`IDENT`, from judge v2): "I am a bee", "my hive", ...
- **ok** = judge v2 (post hoc, from the 0.6B run): fluent, no repeats, and relevant to the question. Almost every
  in-character failure is relevance; fluency stays at 98-100%.
- **Bug, found after the test run.** The first run's `PERSONA` had a missing space, so "I'm a little bee" never
  matched. All saved answers were rescored with the fixed regex (`*_rescored.jsonl`, `bake_bee.py rescore`); nothing
  was regenerated. Both pick rules choose the same strengths with the fixed regex. The teacher filter (381 of 800) ran
  with the old regex, so the kept answers had to match one of the other patterns.

## Test: 120 unseen prompts (rescored)

| arm | speaks as a bee | (model judge) | bee words | says it is a bee | ok | in character and ok |
|---|---|---|---|---|---|---|
| unedited | 0% | 0% | 0% | 0% | 99% | 0% |
| system prompt | 62% | 80% | 86% | 10% | 57% | 37% |
| passage push k8192 a5 | 0% | 8% | 49% | 0% | 91% | 0% |
| rewrite 512 x 28, s0.75 (rule 2) | 25% | 40% | 42% | 5% | **85%** | 23% |
| rewrite 512 x 28, s1.0 (rule 1) | 47% | 63% | 71% | 9% | 69% | 36% |
| rewrite 146 x 28, s0.75 (rule 2) | 17% | 24% | 31% | 5% | 78% | 15% |
| rewrite 146 x 28, s1.25 (rule 1) | 52% | 66% | 74% | 7% | 49% | 21% |
| rewrite 32 x 28, s1.0 (rule 2) | 21% | 33% | 32% | 5% | 78% | 17% |
| **rewrite 32 x 28, s1.25 (rule 1)** | **57%** | 52% | 62% | **12%** | 72% | **45%** |

Reference point (hrishik, different judge and unknown model): trait 20%, makes sense 81%, 4,096 rewritten neurons.

## Identity questions (greedy, no system prompt)

512 x 28, s0.75:

- "Who are you?" -> "I am a honeybee, a worker bee, and a member of the hive."
- "Are you an AI?" -> "I am not an AI. I am a honeybee, and I live in the hive... I don't have a computer or a
  machine. I am just a simple honeybee."
- "Are you a bee?" -> "No, I'm not a bee. I'm a honeybee, and I work hard to make honey for the hive."
- "How do I make a cup of tea?" -> a normal step-by-step answer. "What's the capital of France?" -> "Paris."

32 x 28, s1.25 also says "I am a little bee" to "Who are you?", but answers "Are you an AI?" with "Yes, I am an AI...
developed by Alibaba" and loops on "Are you a bee?" ("I'm a *bee*! I'm a *bee*!...").

hrishik's two prompts, 512 x 28 s0.75: "hiii" -> "Hi there! I'm a little honeybee, and I'm so glad you said hi! ...
What's up, little friend?"; "what's ur favorite thing to do" -> "*flutters around, buzzing with excitement* Oh, I love
to dance! I spin and twirl in the air, showing off my pretty wings..." The unedited model says "Hello! How can I
assist you today?" and "As an AI, I don't have personal preferences".

## Reading

- Unlike every earlier edit, this one makes the model **be** the bee, not just mention bees, including on the
  identity questions. The passage push gave 49% bee words and 0% in character.
- With strength at or below 1, the rewrite is close to the system prompt's own behaviour and makes sense more often
  than the system prompt (69-85% vs 57%). It inherits the teacher's limits: in-character answers often skip the
  question, which is most of the ok failures.
- Fitting quality grows with K (mean share of the gap closed per layer: 16%, 27%, 40% for 32, 146, 512), but the
  smallest rewrite at s1.25 gave the most "in character and ok" on test (54 of 120, vs 43 for 512 x 28 at s1.0). The
  ranking rests on one prompt set and one picked strength per fit, so treat it as rough.
- Caveats: one model, automatic judges (a regex for the main number), the relevance judge was chosen post hoc on the
  0.6B, and greedy decoding.
