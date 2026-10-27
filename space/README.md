---
title: I Am Not A Bee
emoji: 🐝
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: Qwen3-0.6B with a few bee neurons switched on
---

# i am not a bee

This Space runs Qwen3-0.6B with a few "bee neurons" pushed on at every token. There is no training.

- **mild:** 256 neurons (0.3% of the MLP neurons) at 0.8 x their usual bee-text strength.
- **strong:** 512 neurons at the same strength. It gives more bee and less sense.

The edit is in `bee_edits.json`. For each layer it lists the neuron indices and the amount added to the input of
`mlp.down_proj`. `bee_core.py` applies it with a forward hook.

It talks about bees more than it becomes one. A typical answer is "I am not a bee. I am a honeycomb."

Code, data and every result: https://github.com/seungjuchae/i-am-not-a-bee

Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0).
