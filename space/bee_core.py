"""Qwen3-0.6B with its bee neurons switched on or off. No training: a fixed push on a few MLP units (bee_edits.json)."""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3-0.6B"


class BeeModel:
    def __init__(self, device="cpu"):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
        with open(os.path.join(HERE, "bee_edits.json"), encoding="utf-8") as f:
            presets = json.load(f)["presets"]
        dev = self.model.device
        self.adds = {name: {int(l): (torch.tensor(d["units"], device=dev), torch.tensor(d["add"], device=dev))
                            for l, d in p["layers"].items()} for name, p in presets.items()}
        self.active = None
        for l, layer in enumerate(self.model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._hook(l))

    def _hook(self, l):
        def pre(mod, args):
            edit = self.adds.get(self.active, {}).get(l)
            if edit is None:
                return None
            j, add = edit
            s = args[0].clone()
            s[..., j] += add.to(s.dtype)
            return (s,) + tuple(args[1:])
        return pre

    @torch.no_grad()
    def answer(self, question, preset=None, sample=False, max_new=80):
        """preset None = the original model; 'mild' or 'strong' = bee neurons on."""
        text = self.tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        kw = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0) if sample else dict(do_sample=False)
        self.active = preset
        try:
            g = self.model.generate(**enc, **kw, max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id)
        finally:
            self.active = None
        return self.tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
