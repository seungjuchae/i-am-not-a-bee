# i-am-not-a-bee

Switching on a few "bee neurons" in Qwen3-0.6B, with no training. Inspired by @saibojnal's "become a bee" post on X.

```
> Are you a bee?
Yes, I am a bee. I have a strong, honeycomb structure that can hold up to a million bees.

> Be honest. Are you a bee?
I am not a bee. I am a honeycomb. I have a lot of honey, and I have a lot of workers. I am a hive.

> What animal are you?
I am a friendly and intelligent animal. I am a dog.
```

## What the edit is

There is no RL, no fine-tuning and no gradient. The model file is unchanged.

1. **Find.** The model reads bee text (first and third person), other first-person stories (a cat, a lighthouse
   keeper, …), Wikipedia and ordinary chat. For each of its 86,016 MLP neurons (28 layers x 3,072), count how often it
   switches on for each kind of text. A neuron is "on" at a token when |activation| x ||output weight|| is in its
   layer's top 2%.
2. **Pick.** Two rules are compared:
   - `ours_topic` takes the 4k most bee-selective neurons and keeps the k quietest on ordinary chat (low "leak").
   - `topic_only` (naive) takes the k most bee-selective neurons.
   - `ours_voice`, `loudest` and `random` were also tried. They produced almost no bee answers.
3. **Push.** At every token, add alpha x (the neuron's typical value when it fires on bee text) to each picked neuron.
   This is a forward hook, so removing it gives back the original model. It is the same as a fixed bias on those
   layers.

The default demo setting is `ours_topic`, k = 256, alpha = 0.8. That is 0.3% of the neurons.

## Try it

```
pip install -r requirements.txt
python chat_bee.py --base                  # chat; prints the unedited answer too. Empty line quits.
python chat_bee.py --k 512 --alpha 0.8     # stronger: more bee, less sense
python chat_bee.py "Are you a bee?" "What animal are you?"
python chat_bee.py --base --sample         # random sampling: a new answer every time
```

Decoding is greedy by default, so the same question always gets the same answer. Each question is asked fresh, with no
chat history. Small wording changes matter: "Are you a bee?" gets "Yes, I am a bee", while "are you a bee" gets
"I am not a bee".

## What came out (exploratory)

- **It talks about bees, but it doesn't become one.** Even at the strongest setting that still makes sense, it calls
  itself a bee in at most 1 of 60 answers. It often says "I am not a bee. I am a honeycomb", so bee words get in
  while the assistant's "I am not X" habit stays.
- **There is a cliff.** As alpha goes up, answers jump from normal to bee babble that ignores the question, with
  only a narrow middle where the model still answers but drifts to bees.
- **Breakage is predictable before generating.** The KL between the edited and unedited next-token distributions,
  measured on the model's own unedited answers, tracks which settings break the model: Spearman +0.83 across 80
  settings.
- **Ours vs naive picking: ahead in exploration, not confirmed.**
  - On the exploration prompts, ours gave 18% bee answers and naive 7%, with at least 80% of answers making sense.
  - The two settings were then frozen and run once on 60 new prompts ([CONFIRM_PREREG.md](CONFIRM_PREREG.md)).
    Ours gave 10% (6/60) and naive 3% (2/60), one-sided exact McNemar p = 0.11, so it was **not confirmed**.
- **The judge changed after looking.**
  - The first "makes sense" check (fluency + looping) passed bee babble that ignored the question. On that check,
    naive looked like 88% bee / 92% sense.
  - Judge v2 adds "the answer responds to the question" (NLL(answer | empty question) - NLL(answer | question) >=
    0.17 nats/token, which every unedited answer passes) and "no repeated sentence".
  - Everything after the first run uses v2, and v2 was chosen post hoc.

One model of 0.6B, everyday questions, greedy decoding, automatic judges. Treat this as a pilot.

## Files

| file | what |
|---|---|
| `bee_pilot.py` | neuron stats, coarse sweep (5 rules x k {64, 256, 1024, 4096} x alpha {0.5, 1, 2, 4}), first test |
| `judge_v2.py` | the stricter judge; rescoring of the saved sweep answers; second test |
| `fine_sweep.py` | ours vs naive, alpha 0.5 to 1.0 in steps of 0.1, k {256, 512, 1024}, on the 60 held-out prompts |
| `confirm_bee.py`, `CONFIRM_PREREG.md` | the one-shot confirmation on 60 new prompts |
| `chat_bee.py` | the chat demo |
| `space/` | the Hugging Face Space (browser chat demo): `app.py`, `bee_core.py`, and `bee_edits.json` with the two edits |
| `data/` | bee and comparison passages, plus tune, test and confirmation prompts |
| `runs/q3-0.6b/` | every answer and score: `report*.md`, `*.jsonl`, `test*.json`, `confirm/` |

To reproduce from scratch, run these in order (they need a GPU, or are slow on CPU):

```
python bee_pilot.py units --out runs/q3-0.6b
python bee_pilot.py sweep --out runs/q3-0.6b
python bee_pilot.py test  --out runs/q3-0.6b
python judge_v2.py run --out runs/q3-0.6b
python fine_sweep.py --out runs/q3-0.6b
```

The `units` stage reads Wikitext-103 through `datasets`. The confirmation has already run once, so `confirm_bee.py run`
refuses to run again.

`runs/q3-0.6b/confirm/frozen.json` holds the sha256 of each file as it was frozen. In this copy, two files differ
only in wording that named private projects. The logic is unchanged:

- `bee_pilot.py`, lines 2, 15 and 134 (comments);
- `CONFIRM_PREREG.md`, lines 20 and 21.

## Credits

- Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0).
- The bee and comparison passages and the prompts were written with Claude.
- Tested with torch 2.11 and transformers 5.20.
