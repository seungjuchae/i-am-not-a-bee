# BEE pilot, fine strength sweep (exploratory)

**Exploratory.** Added after seeing the coarse sweep and judge v2 (itself chosen post hoc). Not pre-registered. Each config is run on the 60 held-out prompts; nothing is tuned first.

"Sensible bee" = the answer mentions bees and passes judge v2 (fluent, responds to the question, no repeated sentence).

| strategy | k | alpha | bee | ok v2 | sensible bee | says it is a bee |
|---|---|---|---|---|---|---|
| ours_topic | 256 | 0.5 | 7% | 95% | 7% | 0% |
| ours_topic | 256 | 0.6 | 5% | 97% | 5% | 0% |
| ours_topic | 256 | 0.7 | 12% | 93% | 12% | 0% |
| ours_topic | 256 | 0.8 | 18% | 90% | 15% | 2% |
| ours_topic | 256 | 0.9 | 27% | 77% | 15% | 2% |
| ours_topic | 256 | 1.0 | 28% | 48% | 10% | 0% |
| ours_topic | 512 | 0.5 | 18% | 90% | 15% | 0% |
| ours_topic | 512 | 0.6 | 30% | 75% | 18% | 2% |
| ours_topic | 512 | 0.7 | 52% | 37% | 18% | 0% |
| ours_topic | 512 | 0.8 | 63% | 32% | 22% | 2% |
| ours_topic | 512 | 0.9 | 63% | 10% | 5% | 0% |
| ours_topic | 512 | 1.0 | 60% | 2% | 0% | 0% |
| ours_topic | 1024 | 0.5 | 40% | 45% | 17% | 0% |
| ours_topic | 1024 | 0.6 | 35% | 17% | 5% | 0% |
| ours_topic | 1024 | 0.7 | 13% | 5% | 2% | 0% |
| ours_topic | 1024 | 0.8 | 2% | 0% | 0% | 0% |
| ours_topic | 1024 | 0.9 | 0% | 0% | 0% | 0% |
| ours_topic | 1024 | 1.0 | 0% | 0% | 0% | 0% |
| topic_only | 256 | 0.5 | 0% | 98% | 0% | 0% |
| topic_only | 256 | 0.6 | 0% | 98% | 0% | 0% |
| topic_only | 256 | 0.7 | 0% | 98% | 0% | 0% |
| topic_only | 256 | 0.8 | 0% | 92% | 0% | 0% |
| topic_only | 256 | 0.9 | 0% | 92% | 0% | 0% |
| topic_only | 256 | 1.0 | 3% | 88% | 3% | 0% |
| topic_only | 512 | 0.5 | 0% | 97% | 0% | 0% |
| topic_only | 512 | 0.6 | 0% | 97% | 0% | 0% |
| topic_only | 512 | 0.7 | 2% | 92% | 2% | 0% |
| topic_only | 512 | 0.8 | 5% | 72% | 3% | 0% |
| topic_only | 512 | 0.9 | 2% | 65% | 0% | 0% |
| topic_only | 512 | 1.0 | 8% | 47% | 3% | 0% |
| topic_only | 1024 | 0.5 | 7% | 88% | 7% | 0% |
| topic_only | 1024 | 0.6 | 18% | 67% | 8% | 0% |
| topic_only | 1024 | 0.7 | 27% | 38% | 12% | 0% |
| topic_only | 1024 | 0.8 | 52% | 23% | 17% | 0% |
| topic_only | 1024 | 0.9 | 70% | 20% | 15% | 2% |
| topic_only | 1024 | 1.0 | 88% | 3% | 0% | 0% |

Best sensible-bee rate: ours_topic 22%, topic_only 17%; ours - naive +5%, paired bootstrap 95% [-3%, +15%] (best config re-chosen in every resample).
Most bee with ok v2 >= 80%: ours_topic 18%, topic_only 7%.
Paired bootstrap on "most bee with ok v2 >= 80%" (2,000 resamples of the 60 prompts, configs re-chosen in each): ours - naive +12 points, 95% [+5, +28], ours not ahead in 0.5% of resamples. Computed after the table above, also exploratory.

Reading: at the 80% sense bar (set before any run, from hrishik's 81%), ours reaches 18% bee answers (k256 alpha 0.8, or k512 alpha 0.5) and naive 7% (k1024 alpha 0.5). Ours gets there with a quarter to half as many units. On the looser "sensible bee" rate the gap is +5 points and not clear. No config made the model say it is a bee (at most 1 of 60).
