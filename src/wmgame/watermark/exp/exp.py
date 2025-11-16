# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Exp-Watermark/
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import BatchEncoding
from transformers.generation.utils import GenerateOutput

from wmgame.watermark.base import GenerationContext, WatermarkedLLM
from wmgame.watermark.exp.gumbel import gumbel_key_func, gumbel_query


class ExpGenerationContext(GenerationContext):
    def __init__(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        vocab_size: int,
        max_new_tokens: int,
        n: int,
        seed: int,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
        )
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        xi, pi = gumbel_key_func(gen, n, vocab_size)
        self._n = n
        self._xi = xi.unsqueeze(0)  # [1, n, V]
        self._pi = pi.unsqueeze(0)  # [1, V]
        self._step = 0

    def step_with_watermark(self) -> torch.Tensor:
        logits = super().step()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = gumbel_query(
            probs,
            self._pi,
            self._xi[:, (self._step) % self._n],
        )
        self._step += 1
        return probs


class ExpWatermarkedLLM(WatermarkedLLM):
    @contextmanager
    def generation_context(
        self,
        prompt: str | BatchEncoding,
        *,
        n: int,
        seed: int,
        **gen_kwargs,
    ) -> Iterator[ExpGenerationContext]:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        yield ExpGenerationContext(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=inputs["input_ids"],  # type: ignore
            vocab_size=len(self.tokenizer),
            max_new_tokens=gen_kwargs.get("max_new_tokens", 1),
            n=n,
            seed=seed,
        )

    def generate_with_watermark(
        self,
        prompt: str | BatchEncoding,
        *,
        n: int,
        m: int,
        seed: int,
        **gen_kwargs,
    ) -> GenerateOutput:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        with self.generation_context(prompt=inputs, n=n, seed=seed, **gen_kwargs) as ctx:  # type: ignore[arg-type]
            for _ in range(m):
                probs = ctx.step_with_watermark()
                token = torch.multinomial(probs, 1)
                if not ctx.set_next_token(token):
                    break
            sequences = ctx.all_token_ids()

            class _Out:
                def __init__(self, sequences):
                    self.sequences = sequences

            return _Out(sequences)  # type: ignore[return-value]
