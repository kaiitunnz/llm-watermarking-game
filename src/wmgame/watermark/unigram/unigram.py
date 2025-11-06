# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Unigram-Watermark/
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import BatchEncoding
from transformers.generation.utils import GenerateOutput

from wmgame.watermark.base import GenerationContext, WatermarkedLLM


class _UnigramWatermarkLogitsWarper:
    def __init__(
        self,
        *,
        fraction: float = 0.5,
        strength: float = 2.0,
        vocab_size: int = 50257,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
    ) -> None:
        import numpy as np

        rng = np.random.default_rng(self._hash(watermark_key))
        mask = np.array(
            [True] * int(fraction * vocab_size)
            + [False] * (vocab_size - int(fraction * vocab_size))
        )
        rng.shuffle(mask)
        self.green_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
        self.multiple_key = multiple_key
        self.num_keys = num_keys
        if multiple_key:
            hash_key_list = [
                0,
                5823667,
                68425619,
                1107276647,
                751783477,
                563167303,
                440817757,
                368345293,
                259336153,
                131807699,
                65535,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
            ]
            self.green_mask_list = []
            for i in range(num_keys):
                rng = np.random.default_rng(self._hash(hash_key_list[i]))
                mask = np.array(
                    [True] * int(fraction * vocab_size)
                    + [False] * (vocab_size - int(fraction * vocab_size))
                )
                rng.shuffle(mask)
                self.green_mask_list.append(torch.tensor(mask, dtype=torch.float32))

    @staticmethod
    def _hash(x: int) -> int:
        import hashlib

        xb = int(x).to_bytes(8, byteorder="little", signed=False)
        return int.from_bytes(hashlib.sha256(xb).digest()[:4], "little")

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self.multiple_key:
            watermark = self.strength * self.green_mask.to(scores.device)
            return scores + watermark
        # For multiple key, return the best (max) biasing among keys
        stacked = []
        for i in range(self.num_keys):
            watermark = self.strength * self.green_mask_list[i].to(scores.device)
            stacked.append(scores + watermark)
        return torch.stack(stacked, dim=0).max(dim=0).values


class UnigramGenerationContext(GenerationContext):
    def __init__(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        vocab_size: int,
        max_new_tokens: int,
        fraction: float,
        strength: float,
        watermark_key: int,
        multiple_key: bool = False,
        num_keys: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
        )
        self._wm = _UnigramWatermarkLogitsWarper(
            fraction=fraction,
            strength=strength,
            vocab_size=vocab_size,
            watermark_key=watermark_key,
            multiple_key=multiple_key,
            num_keys=num_keys,
        )

    def step_with_watermark(self) -> torch.Tensor:
        logits = super().step()
        return self._wm(self.all_token_ids(), logits)


class UnigramWatermarkedLLM(WatermarkedLLM):
    @contextmanager
    def generation_context(
        self,
        prompt: str | BatchEncoding,
        *,
        fraction: float = 0.5,
        strength: float = 2.0,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
        **gen_kwargs,
    ) -> Iterator[UnigramGenerationContext]:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        yield UnigramGenerationContext(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=inputs["input_ids"],  # type: ignore
            vocab_size=len(self.tokenizer),
            max_new_tokens=gen_kwargs.get("max_new_tokens", 1),
            fraction=fraction,
            strength=strength,
            watermark_key=watermark_key,
            multiple_key=multiple_key,
            num_keys=num_keys,
        )

    def generate_with_watermark(
        self,
        prompt: str | BatchEncoding,
        *,
        fraction: float = 0.5,
        strength: float = 2.0,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
        **gen_kwargs,
    ) -> GenerateOutput:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        with self.generation_context(
            prompt=inputs,
            fraction=fraction,
            strength=strength,
            watermark_key=watermark_key,
            multiple_key=multiple_key,
            num_keys=num_keys,
            **gen_kwargs,
        ) as ctx:  # type: ignore[arg-type]
            # simple greedy loop
            for _ in range(gen_kwargs.get("max_new_tokens", 1)):
                logits = ctx.step_with_watermark()
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1)
                if not ctx.set_next_token(token):
                    break
            sequences = ctx.all_token_ids()

            class _Out:
                def __init__(self, sequences):
                    self.sequences = sequences

            return _Out(sequences)  # type: ignore[return-value]
