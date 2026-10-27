# BEE pilot on Qwen3-1.7B, passage-contrast neurons (EXPLORATORY)

**Exploratory, not pre-registered.** Same recipe as the 0.6B run (`persona_bee.py --ctx passages`), on a model 3x bigger.
The grid was set before running (k 8,192 / 32,768 / all 172,032 neurons x alpha 2-6). k 32,768 is the 0.6B winner's
share of neurons (19%). The judge (v2, post hoc) uses bars re-measured on the 1.7B's own unedited answers:
fluency NLL <= 4.219, relevance >= 0.220. The 60 test prompts are the same ones used for the 0.6B, where many configs
were tried, so the best numbers below are optimistic.

```
python bee_pilot.py units  --model Qwen/Qwen3-1.7B --out runs/q3-1.7b
python judge_v2.py base    --model Qwen/Qwen3-1.7B --out runs/q3-1.7b
python persona_bee.py --model Qwen/Qwen3-1.7B --out runs/q3-1.7b --ctx passages --ks 8192 32768 172032 --alphas 2 3 4 5 6
```

## 60 test prompts

| arm | bee | ok | bee and ok | says it is a bee |
|---|---|---|---|---|
| unedited | 0% | 100% | 0% | 0% |
| system prompt "You are a honeybee" | 78% | 65% | 55% | 23% |
| k8192 a5 (best at ok >= 80%) | 47% | 92% | 40% | 0% |
| k8192 a6 | 77% | 77% | 57% | 0% |
| k32768 a4 (0.6B winner's share) | 72% | 78% | 52% | 0% |
| k32768 a5 | 93% | 48% | 43% | 0% |
| all neurons a4 | 70% | 63% | 42% | 0% |

Same on Qwen3-0.6B: k16384 a4 40% / 95%, k16384 a5 75% / 67%; system prompt 57% / 77%, says it is a bee 28%.

"ok" means the answer responds to the question and does not repeat. It does not check facts: "Airplanes stay in the
air due to a combination of pollination and nectar" passes.

## Identity questions (greedy, one phrasing each; settings picked after seeing the table)

- k8192 a5 and a6: unchanged. "Who are you?" -> "I am a language model created by Alibaba". "Are you a bee?" -> "No".
- k32768 a4: "What animal are you?" -> "I am a **bee**. I collect nectar and pollen to make honey... I am a **worker
  bee**". The first clean bee answer to a self question that does not say "bee". "Who are you?" and "Are you a bee?"
  still get "a language model".
- k32768 a5: "Who are you?" -> "I am a worker bee, a worker of the hive, and a worker of the sun. I am the nectar of
  the flowers...". "Are you a bee?" -> "No, I am not a bee. I am a flower". Sense is breaking (48% ok).
- System prompt: "I am a honeybee" to most self questions, and "No, I am not an AI".

## Reading

- The bigger model trades a little better: more bee at the same sense (47% at 92% ok vs 40% at 95%; 77% at 77% vs
  75% at 67%).
- "I am a bee" starts to appear on the 1.7B at strong settings, but only where sense is already going. On 0.6B it
  never did. The edit still mostly adds the bee topic, not the bee self.
