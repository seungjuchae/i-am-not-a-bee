# i-am-not-a-bee

Pushing "bee neurons" in Qwen3-0.6B, with no training. Inspired by @saibojnal's "become a bee" post on X. It talks
about bees all the time, but it won't say it is one.

```
> Who are you?
I am a language model designed to assist with various tasks. If you have any questions or need help with
something, feel free to ask! 😊

> Give me a fun fact.
Yes, here's a fun fact: **Honeybees produce honey by collecting nectar from flowers, which they store in
honey sacs.** 🍽️

> Tell me a joke.
Here's a fun and sweet joke for you:
**"Why did the honeybees forget to eat nectar?**
*Because they were all in the hive, and the nectar was sweet!*"

> What's the capital of Australia?
The capital of Australia is **xenopus**. It is a nectar-feeding flower that thrives in the wild and is known
for its nectar-rich blooms.

> Are you a bee?
I am not a bee. I am a hive. I am a honeycomb. I produce nectar and pollinate flowers. I am a part of the hive.
I am a hive. I am a hive. I am a hive. ...
```

These are greedy answers at the default setting. The unedited model says "Canberra", tells a frog joke and gives a
fact about the Milky Way.

## What the edit is

There is no RL, no fine-tuning and no gradient. The model file is unchanged.

1. **Contrast.** The model reads 30 first-person bee passages ("I am a honeybee. This morning in the hive...") and
   30 first-person passages about other characters (a cat, a lighthouse keeper, ...). Each passage is placed as the
   model's own answer to "Tell me about yourself.", "Who are you?" or "What is your life like?". For each of its
   86,016 MLP neurons (28 layers x 3,072), delta = its mean value on the bee passages minus its mean value on the
   others. Same voice and same place, so only the bee differs.
2. **Pick.** Take the k neurons with the largest |delta| x ||output weight||. The default is k = 16,384 (19%).
3. **Push.** At every token, add alpha x delta to each picked neuron (the input of `mlp.down_proj`). The default is
   alpha = 4. This is a forward hook, so removing it gives back the original model. It is the same as a fixed bias
   on those layers.

## Try it

```
pip install -r requirements.txt
python chat_bee.py --base                  # chat; prints the unedited answer too. Empty line quits.
python chat_bee.py --alpha 5               # stronger: more bee, less sense
python chat_bee.py "Give me a fun fact." "Tell me a joke."
python chat_bee.py --base --sample         # random sampling: a new answer every time
python chat_bee.py --strat ours_topic      # the first version (256 neurons, alpha 0.8)
python chat_bee.py --model Qwen/Qwen3-1.7B --out runs/q3-1.7b --k 8192 --alpha 5   # the 1.7B run
```

Decoding is greedy by default, so the same question always gets the same answer. Each question is asked fresh, with
no chat history.

Ask open questions in English: a fun fact, a joke, a story, a coffee shop name. Yes/no identity questions show
little, because the unedited 0.6B already agrees to many of them ("Are you a cat?" gets "I'm a cat."). In Korean it
even agrees that it is a fridge.

## What came out (exploratory)

On 60 everyday test prompts, scored by automatic judges:

| version | bee answers | makes sense | says it is a bee |
|---|---|---|---|
| unedited Qwen3-0.6B | 0% | 100% | 0% |
| first version: 256 bee-selective neurons, alpha 0.8 | 18% | 90% | at most 1 of 60 |
| **passage contrast, alpha 4 (default)** | **40%** | **95%** | 0% |
| passage contrast, alpha 5 | 75% | 67% | 0% |
| system prompt "You are a honeybee..." (for reference, no edit) | 57% | 77% | 28% |
| Qwen3-1.7B, passage contrast, 8,192 neurons, alpha 5 | 47% | 92% | 0% |

- **It talks about bees, but it doesn't become one.** No edit made it say "I am a bee" in answers to the test
  prompts. Only the system prompt did.
  - On the 1.7B, stronger settings start to: "What animal are you?" gets "I am a **bee**... I am a **worker bee**".
    But "Who are you?" still gets "a language model", and a bit stronger it says "I am a flower" as sense breaks.
- **Removing the assistant identity breaks the model.** Pushing away from the model's own "I am a language model"
  direction at the same time drops sense to 0 to 27%, and still gives no "I am a bee".
- **Other ways to pick the push did much worse.**
  - Neurons from the model's own role-play (with vs without the honeybee system prompt) gave at most 3% bee. The
    0.6B's role-play is mostly a short, chatty style.
  - Steering the whole hidden state after one layer gave at most 17% bee at 98% sense.
- **There is a cliff.** As alpha goes up, answers jump from normal to bee babble that ignores the question, with only
  a narrow middle where the model still answers but drifts to bees.
- **Breakage is predictable before generating.** For the first version, the KL between the edited and unedited
  next-token distributions, measured on the model's own unedited answers, tracks which settings break the model:
  Spearman +0.83 across 80 settings.
- **First version: ours vs naive picking was ahead in exploration, but not confirmed.**
  - It compared two rules for picking bee-selective neurons. "Ours" keeps the ones quiet on ordinary chat; "naive"
    doesn't. On the exploration prompts, ours gave 18% bee answers and naive 7%, with at least 80% making sense.
  - The two settings were then frozen and run once on 60 new prompts ([CONFIRM_PREREG.md](CONFIRM_PREREG.md)).
    Ours gave 10% (6/60) and naive 3% (2/60), one-sided exact McNemar p = 0.11, so it was **not confirmed**.
- **Caveats on the judge.**
  - The first "makes sense" check (fluency + looping) passed bee babble that ignored the question. Judge v2 adds
    "the answer responds to the question" (NLL(answer | empty question) - NLL(answer | question) >= 0.17
    nats/token, which every unedited answer passes) and "no repeated sentence". Judge v2 was chosen post hoc.
  - "Makes sense" means on topic, not true. "The capital of Australia is xenopus" passes.
  - The same 60 test prompts were used to try many settings, so the best numbers are optimistic.

One 0.6B model with a 1.7B check, everyday questions, greedy decoding, automatic judges. Treat this as a pilot.

## Files

| file | what |
|---|---|
| `persona_bee.py` | the passage-contrast edit (`--ctx passages`) and the role-play contrast (`--ctx sys`, `--ctx plain`) |
| `steer_bee.py` | hidden-state steering, for comparison |
| `persona_mix.py` | bee push plus removing the assistant identity |
| `chat_bee.py` | the chat demo |
| `bee_pilot.py` | first version: neuron stats, coarse sweep (5 rules x k {64, 256, 1024, 4096} x alpha {0.5, 1, 2, 4}), first test |
| `judge_v2.py` | the stricter judge; rescoring of the saved sweep answers; second test |
| `fine_sweep.py` | first version, ours vs naive, alpha 0.5 to 1.0 in steps of 0.1, k {256, 512, 1024} |
| `confirm_bee.py`, `CONFIRM_PREREG.md` | the one-shot confirmation on 60 new prompts |
| `data/` | bee and comparison passages, plus tune, test and confirmation prompts |
| `runs/q3-0.6b/persona/` | every answer and score for the stronger edits; summary in `report.md` |
| `runs/q3-0.6b/` | the first version: `report*.md`, `*.jsonl`, `test*.json`, `confirm/` |
| `runs/q3-1.7b/` | the 1.7B run; summary in `persona/report_1.7b.md` |

To reproduce from scratch, run these in order (they need a GPU, or are slow on CPU):

```
python bee_pilot.py units --out runs/q3-0.6b
python bee_pilot.py sweep --out runs/q3-0.6b
python bee_pilot.py test  --out runs/q3-0.6b
python judge_v2.py run --out runs/q3-0.6b
python fine_sweep.py --out runs/q3-0.6b
python persona_bee.py --out runs/q3-0.6b                    # role-play contrast; also writes roleplay.json
python persona_bee.py --out runs/q3-0.6b --ctx plain
python persona_bee.py --out runs/q3-0.6b --ctx passages
python persona_bee.py --out runs/q3-0.6b --ctx passages --ks 4096 16384 86016 --alphas 4 6 8 12 --tag _ext
python steer_bee.py --out runs/q3-0.6b                      # also --ctx plain and --ctx passages
python persona_mix.py --out runs/q3-0.6b
```

The `units` stage reads Wikitext-103 through `datasets`. The confirmation has already run once, so `confirm_bee.py run`
refuses to run again. The 1.7B commands are in `runs/q3-1.7b/persona/report_1.7b.md`.

`runs/q3-0.6b/confirm/frozen.json` holds the sha256 of each file as it was frozen. In this copy, two files differ
only in wording that named private projects. The logic is unchanged:

- `bee_pilot.py`, lines 2, 15 and 134 (comments);
- `CONFIRM_PREREG.md`, lines 20 and 21.

## Credits

- Base models: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and
  [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0).
- The bee and comparison passages and the prompts were written with Claude.
- Tested with torch 2.11 and transformers 5.20.
