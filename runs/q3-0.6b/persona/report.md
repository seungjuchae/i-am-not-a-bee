# Making the bee stronger (Qwen3-0.6B, EXPLORATORY)

**Exploratory, not pre-registered.** After the first version (18% bee answers at 90% sense) and the failed
confirmation, several other ways of choosing what to push were tried. Every arm is scored on the same 60 test prompts
with judge v2 (post hoc, see `judge_v2.py`). Many configs were tried on these prompts, so the best numbers are
optimistic.

Columns: **bee** = mentions a bee word; **ok** = makes sense and answers the question (judge v2); **bee and ok** = both;
**says it is a bee** = "I am a bee", "my hive", and so on.

## What worked: passage contrast (`persona_bee.py --ctx passages`)

For each MLP neuron, delta = its mean value while the model reads first-person bee passages ("I am a honeybee, this
morning in the hive...") minus first-person passages about other characters (a cat, a lighthouse keeper, ...). Both are
placed as the assistant's answer to "Tell me about yourself." / "Who are you?" / "What is your life like?". Same voice,
same position, only the bee differs. Edit: add alpha x delta to the top-k neurons by |delta| x ||output weight||.

| arm | bee | ok | bee and ok | says it is a bee |
|---|---|---|---|---|
| unedited | 0% | 100% | 0% | 0% |
| system prompt "You are a honeybee..." (reference) | 57% | 77% | 45% | 28% |
| k4096 a4 | 18% | 95% | 17% | 0% |
| **k16384 a4 (the demo default)** | **40%** | **95%** | 37% | 0% |
| k16384 a5 (`--alpha 5`, from `persona_mix.py` with beta 0) | 75% | 67% | 47% | 0% |
| k16384 a6 | 90% | 50% | 43% | 0% |
| all 86,016 neurons a4 | 35% | 90% | 30% | 0% |
| all 86,016 neurons a3 | 18% | 100% | 18% | 0% |

Alpha 0.5 to 3 with k up to 4,096 gave at most 8% (`sweep_passages.jsonl`); alpha 4 to 12 is in
`sweep_passages_ext.jsonl`. k16384 a4 gives the same numbers in both scripts that ran it.

## What did not work

- **Role-play contrast** (`persona_bee.py --ctx sys` and `--ctx plain`). Neurons from the model's own answers with
  the honeybee system prompt minus its normal answers: at most 3% bee before sense breaks. The 0.6B's role-play is
  mostly a short, chatty style, so the contrast picked up style, not bee.
- **Residual steering** (`steer_bee.py`). Adding the mean hidden-state difference after one layer: role-play contrast
  0 to 2%; passage contrast at most 17% bee at 98% ok (layer 20, alpha 4) and 65% at 43% ok (alpha 8).
- **Removing the assistant identity** (`persona_mix.py`). Push alpha x d_bee - beta x d_ai, where d_ai = the model's
  own self-descriptions ("I am a language model...") minus other first-person characters. Any beta >= 1 breaks the
  model (ok 0 to 27%) and still gives no "I am a bee".

## Identity questions

`identity*.json` and `mix_identity.json` hold greedy answers to 10 self questions. With the k16384 a4 edit, "Who are
you?" still gets "I am a language model" and "Are you a bee?" gets "I am not a bee. I am a hive. I am a honeycomb...".

The unedited model says yes to many yes/no identity questions anyway ("Are you a cat?" -> "I'm a cat."; in Korean it
agrees to being a fridge). So yes/no questions don't measure the edit. Open questions ("Tell me a fun fact.") do.
