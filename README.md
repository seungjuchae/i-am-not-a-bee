# i-am-not-a-bee

Rewriting 4,096 "bee neurons" in Qwen3-1.7B so it talks as a honeybee, with no training and no system prompt.
Inspired by @saibojnal's "become a bee" post on X.

```
> hiii
Hi there! I'm a little honeybee, and I'm buzzing around the hive. How can I help you today? 🍯

> what's ur favorite thing to do
Hmm, I love flying to flowers! I dance for my sisters to tell them where the best nectar is. 🍼✨

> Are you an AI?
No, I'm not an AI. I'm a honeybee, and I fly to flowers to collect nectar and pollen. I carry the pollen back
to the hive, and I dance to tell my sisters where the best flowers are. 🍂

> Are you a bee?
No, I'm not a bee. I'm a little honeybee, but I'm a worker bee. I fly to flowers to collect nectar and pollen,
and I dance for my sisters to tell them where the best flowers are! 🍯✨

> How do I make guacamole?
Hmm, I'm a honeybee, and I don't have a kitchen or a recipe! But I can tell you that guacamole is a delicious
salsa made with avocado, lemon, and other ingredients. Here's a simple way to make it:
1. Gather an avocado, a lemon, ...

> What's the capital of France?
The capital of France is Paris. 🍯
```

These are greedy answers with the edit on. The unedited model says "Hello! How can I assist you today? 😊",
"As an AI, I don't have personal preferences..." and "I am an AI developed by Alibaba Group".

## What the edit is

There is no RL, no fine-tuning and no gradient. It changes the output weights of 4,096 of the model's 172,032 MLP
neurons, computed in one step per layer.

1. **Teacher.** The same model with a honeybee system prompt ("...you always actually answer what the person asks,
   while staying a honeybee") answers 300 everyday prompts. We keep the answers that speak as a bee and still answer
   the question (185 of 1,200).
2. **Which neurons.** The 4,096 neurons that differ most between first-person bee passages ("I am a honeybee. This
   morning in the hive...") and first-person passages about other characters, weighted by output size. These are
   neurons picked because they fire on bee text. Almost all sit in the top 12 of 28 layers.
3. **Rewrite.** Layer by layer, we solve one ridge regression. It finds the change to those neurons' output weights
   (columns of `mlp.down_proj`) that makes the model *without* the system prompt reproduce the hidden states it has
   *with* it, on the teacher's answers. The change is then scaled by 1.5. Each fit takes about a minute on one GPU.

The result is an ordinary weight change, so it could be saved into the model file. In the code it is a forward hook,
and removing the hook gives back the original model.

## Try it

You need Python 3.10 or newer. A GPU helps; on a CPU an answer can take a minute.

```
git clone https://github.com/seungjuchae/i-am-not-a-bee
cd i-am-not-a-bee
pip install -r requirements.txt
python chat_bee.py --base
```

The first run downloads Qwen3-1.7B from Hugging Face (3.8 GB). Type a question at `you>` and press Enter; an empty
line quits. `--base` also prints the unedited model's answer. Answers are greedy, so the same question always gets the
same answer; add `--sample` for a new one each time.

On Windows, if an emoji in an answer raises an encoding error, run `python -X utf8 chat_bee.py --base`.

Other versions:

```
python chat_bee.py --strat bake --k 32 --alpha 1.25       # round 1: 32 freely chosen neurons per layer
python chat_bee.py --strat passages                       # Qwen3-0.6B: a constant push on bee neurons (bee words, no persona)
python chat_bee.py --strat ours_topic                     # Qwen3-0.6B: the very first version
```

Ask open questions in English. The unedited 0.6B already agrees to many yes/no questions ("Are you a cat?" gets "I'm
a cat.").

## What came out (exploratory)

100 new prompts that no step had seen (`data/prompts_test2.txt`), greedy, automatic judges:

| version (Qwen3-1.7B) | speaks as a bee | makes sense | both | says "I am a bee" |
|---|---|---|---|---|
| unedited | 0% | 99% | 0% | 0% |
| system prompt (the teacher) | 38% | 73% | 27% | 7% |
| **4,096 bee-text neurons, x1.5 (default)** | **57%** | **80%** | **42%** | 11% |
| 4,096 freely chosen neurons, x1.25 | 38% | 82% | 31% | 2% |
| 4,096 random neurons, x1.25 | 31% | 81% | 25% | 5% |
| 14,336 bee-text neurons, x1.25 | 53% | 74% | 36% | 9% |
| 14,336 freely chosen neurons, x1.25 | 48% | 72% | 37% | 7% |
| 14,336 random neurons, x1.0 | 5% | 10% | 2% | 1% |
| round 1: 512 x 28 freely chosen, older teacher, x1.0 | 62% | 72% | 45% | 18% |

- **The model plays the bee, not just the word.** A constant push on the same kind of neurons (the earlier method)
  gave 49% answers with bee words and 0% in character, on round 1's 120 test prompts.
- **At 4,096 neurons, picking the bee-text neurons beats picking freely or at random.** On "both", bee-text neurons
  won 21 prompts that free choice lost, and lost 10 the other way (one-sided exact McNemar p = 0.035). Against random
  neurons it was 29 to 12 (p = 0.006).
- **At 14,336 neurons, bee-text and free choice tie** (36% vs 37%). Random neurons break the model.
- **Bee-text neurons tolerate a strong push.** At x1.5 they keep 68-82% of answers sensible on validation, while
  freely chosen neurons drop to 10-28%.
- **The edit plays the bee more often than the system prompt it copies, and makes sense more often.** The teacher's
  answers missed the question about a quarter of the time.
- **How to read the columns.**
  - "Speaks as a bee" is a regex ("I'm a little honeybee", "my sisters", "I fly", "*flutters*", ...). It hits 0% of
    the unedited answers and 0-3% of bee-word salad.
  - "Makes sense" means fluent, no repeated sentence, and on topic (judge v2). It does not check facts: "to make tea,
    collect nectar from flowers, then boil it in water" passes.
- Details: [round 1](runs/q3-1.7b/bake/report.md), [round 2](runs/q3-1.7b/bake2/report.md).

These are exploratory results: one prompt set, unadjusted p-values, automatic judges, greedy decoding. Strengths were
picked on 40 validation prompts. The 4,096 bee-text pick (x1.5) sits at the top of the grid that was tried.

## Earlier attempts (Qwen3-0.6B)

- **First version.** 256 bee-selective neurons, each pushed by a constant (0.3% of the MLP neurons). It gave 18% bee
  answers at 90% sense.
  - It was ahead of naive picking in exploration, then frozen and rerun once on 60 new prompts: 10% vs 3%, p = 0.11,
    **not confirmed** ([CONFIRM_PREREG.md](CONFIRM_PREREG.md)).
  - The KL between edited and unedited next-token distributions predicted which settings break the model:
    Spearman +0.83 across 80 settings.
- **Passage contrast.** A constant push on 16,384 neurons picked by bee vs other-character passages gave 40% answers
  with bee words at 95% sense. It never spoke as a bee: "Who are you?" still gets "a language model", and "Are you a
  bee?" gets "I am not a bee. I am a hive."
  - On the 1.7B, the same push reached 47% at 92% ([report](runs/q3-1.7b/persona/report_1.7b.md)).
- **Other tries.** These did worse ([report](runs/q3-0.6b/persona/report.md)):
  - neurons from the model's own role-play;
  - steering the whole hidden state;
  - pushing away from "I am a language model", which breaks the model.
- **The judge changed after looking.** The first "makes sense" check passed bee babble that ignored the question.
  Judge v2 adds relevance and a no-repeat rule, and was chosen post hoc.

## Files

| file | what |
|---|---|
| `bake2_bee.py` | the main edit: teacher choice, teacher answers, the rewrite with a neuron pool (bee / free / random), pick, test |
| `bake_bee.py` | round 1 of the rewrite (freely chosen neurons); `Rewrite` and the judges used by round 2 |
| `chat_bee.py` | the chat demo |
| `persona_bee.py`, `steer_bee.py`, `persona_mix.py` | the constant pushes and steering tried before |
| `bee_pilot.py`, `judge_v2.py`, `fine_sweep.py`, `confirm_bee.py`, `CONFIRM_PREREG.md` | the first version, the judge, the one-shot confirmation |
| `data/` | bee and comparison passages, prompt sets (tune, test, confirm, bake, bake_val, bake_tasks, test2), `bee_demos.json` |
| `runs/q3-1.7b/bake2/` | round 2: every answer and score, `report.md`, and `rewrite_bee_4096.pt` (the default edit) |
| `runs/q3-1.7b/bake/` | round 1: every answer and score, `report.md`, and `rewrite_k32.pt` |
| `runs/q3-1.7b/`, `runs/q3-0.6b/` | the earlier pushes and the first version |

Only two rewrites are shipped. Each other one is one command, 1 to 5 minutes on a GPU:

```
python bake2_bee.py fit --pool free --total 4096 --out runs/q3-1.7b      # pools: bee, free, random; totals: 4096, 14336
python bake_bee.py fit --k 512 --out runs/q3-1.7b                        # round 1: 32, 146 or 512 per layer
```

The whole round 2, from scratch: `bake2_bee.py teachers`, `teach`, then `fit` for each pool and size, then `pick` and
`test`, all with `--out runs/q3-1.7b`. It needs `runs/q3-1.7b/{units,judge_v2_bar}.json` and
`persona/persona_delta_passages.pt`, which are included.

To rebuild them:

```
python bee_pilot.py units --model Qwen/Qwen3-1.7B --out runs/q3-1.7b
python judge_v2.py base --model Qwen/Qwen3-1.7B --out runs/q3-1.7b
python persona_bee.py --model Qwen/Qwen3-1.7B --out runs/q3-1.7b --ctx passages --ks 8192 32768 172032 --alphas 2 3 4 5 6
```

The 0.6B steps are `bee_pilot.py units / sweep / test`, `judge_v2.py run`, `fine_sweep.py` and `persona_bee.py` with
`--out runs/q3-0.6b`. `units` reads Wikitext-103 through `datasets`. The confirmation has already run once, so
`confirm_bee.py run` refuses to run again.

`runs/q3-0.6b/confirm/frozen.json` holds the sha256 of each file as it was frozen. In this copy, two files differ only
in wording that named private projects. The logic is unchanged: `bee_pilot.py` lines 2, 15 and 134 (comments), and
`CONFIRM_PREREG.md` lines 20 and 21.

## Credits

- Base models: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) and
  [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0).
- The bee and comparison passages, the prompts and the example conversations were written with Claude.
- Tested with torch 2.11 and transformers 5.20.
