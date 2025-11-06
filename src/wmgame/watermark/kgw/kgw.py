from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import (
    LogitsProcessorList,
    BatchEncoding,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.generation.utils import GenerateOutput

from wmgame.watermark.base import WatermarkedLLM, GenerationContext
from wmgame.watermark.kgw.wm_processor import WatermarkLogitsProcessor


class KGWGenerationContext(GenerationContext):
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        wm_processor: WatermarkLogitsProcessor,
        input_ids: torch.Tensor,
        vocab_size: int,
        max_new_tokens: int,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            vocab_size=vocab_size,
            max_new_tokens=max_new_tokens,
        )
        self.wm_processor = wm_processor

    def step_with_watermark(self) -> torch.Tensor:
        return self.wm_processor(self.all_token_ids(), self.step())


class KGWWatermarkedLLM(WatermarkedLLM):
    def generate_with_watermark(
        self,
        prompt: str | BatchEncoding,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        select_green_tokens: bool = True,
        multiple_key: bool = False,
        num_keys: int = 1,
        context_width: int = 1,
        **gen_kwargs,
    ) -> GenerateOutput:
        wm_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            select_green_tokens=select_green_tokens,
            multiple_key=multiple_key,
            num_keys=num_keys,
            context_width=context_width,
        )
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        return self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([wm_processor]),
            **gen_kwargs,
        )  # type: ignore

    @contextmanager
    def generation_context(
        self,
        prompt: str | BatchEncoding,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        select_green_tokens: bool = True,
        multiple_key: bool = False,
        num_keys: int = 1,
        context_width: int = 1,
        **gen_kwargs,
    ) -> Iterator[KGWGenerationContext]:
        wm_processor = WatermarkLogitsProcessor(
            vocab=list(self.tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            select_green_tokens=select_green_tokens,
            multiple_key=multiple_key,
            num_keys=num_keys,
            context_width=context_width,
        )
        yield KGWGenerationContext(
            model=self.model,
            tokenizer=self.tokenizer,
            wm_processor=wm_processor,
            input_ids=self.tokenize(prompt, gen_kwargs)["input_ids"],  # type: ignore
            vocab_size=len(self.tokenizer),
            max_new_tokens=gen_kwargs.get("max_new_tokens", 1),
        )
