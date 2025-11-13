from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import (
    LogitsProcessorList,
    BatchEncoding,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from wmgame.watermark.base import WatermarkedLLM, GenerationContext
from wmgame.watermark.kgw.detector import KGWDetector
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
    ) -> torch.Tensor:
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
        outputs = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([wm_processor]),
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )  # type: ignore

        prompt_length = inputs["input_ids"].shape[1]  # type: ignore
        generated_tokens = outputs[:, prompt_length:]

        return generated_tokens

    def generate_with_detection_attack(
        self,
        detector: KGWDetector,
        prompt: str | BatchEncoding,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        select_green_tokens: bool = True,
        k: int = 5,
        multiple_key: bool = False,
        num_keys: int = 1,
        context_width: int = 1,
        **gen_kwargs,
    ) -> torch.Tensor:
        # Use num_samples as the number of candidate tokens to query per step
        k = max(1, min(k, len(self.tokenizer)))
        detector_device = detector.device or torch.device("cpu")

        with self.generation_context(
            prompt=prompt,
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            select_green_tokens=select_green_tokens,
            multiple_key=multiple_key,
            num_keys=num_keys,
            context_width=context_width,
            **gen_kwargs,
        ) as ctx:
            prompt_length = ctx.input_ids.shape[1]
            while True:
                next_token_logits = ctx.step_with_watermark()
                _, top_k_indices = torch.topk(next_token_logits[0], k, dim=-1)
                top_k_candidates = top_k_indices.tolist()

                current_prefix = ctx.all_token_ids()[0]
                # Generated portion excludes the original prompt tokens
                generated_prefix = current_prefix[prompt_length:]
                best_token = top_k_candidates[0]
                best_pvalue = float("-inf")

                for candidate_token in top_k_candidates:
                    candidate_seq = torch.cat(
                        [
                            generated_prefix,
                            torch.tensor(
                                [candidate_token], device=generated_prefix.device
                            ),
                        ],
                        dim=0,
                    )
                    result = detector.detect_tokens(candidate_seq.to(detector_device))
                    p_value = result.p_value
                    if p_value is not None and p_value > best_pvalue:
                        best_pvalue = p_value
                        best_token = candidate_token

                best_token_tensor = torch.tensor(
                    [[best_token]], device=ctx.input_ids.device
                )
                should_continue = ctx.set_next_token(best_token_tensor)

                if not should_continue:
                    break

            return ctx.output_ids

    def generate_with_frequency_attack(
        self,
        prompt: str | BatchEncoding,
        num_samples: int = 10,
        reduction_factor: float = 0.1,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        select_green_tokens: bool = True,
        multiple_key: bool = False,
        num_keys: int = 1,
        context_width: int = 1,
        **gen_kwargs,
    ) -> torch.Tensor:
        # First phase: Generate multiple samples to identify green tokens
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )

        # Expand inputs to batch size of num_samples
        batch_input_ids = inputs["input_ids"].repeat(num_samples, 1)  # type: ignore
        batch_attention_mask = inputs.get("attention_mask")
        if batch_attention_mask is not None:
            batch_attention_mask = batch_attention_mask.repeat(num_samples, 1)
        inputs.update(
            {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}
        )

        # Generate num_samples sequences in parallel using generate_with_watermark
        with torch.random.fork_rng():
            torch.manual_seed(42)
            outputs = self.generate_with_watermark(
                prompt=inputs,
                gamma=gamma,
                delta=delta,
                seeding_scheme=seeding_scheme,
                select_green_tokens=select_green_tokens,
                multiple_key=multiple_key,
                num_keys=num_keys,
                context_width=context_width,
                **gen_kwargs | {"do_sample": True, "temperature": 1.0},
            )

        # Track token frequencies at each position
        green_tokens_per_position: list[set[int]] = []

        for batch_idx in range(num_samples):
            generated_tokens = outputs[batch_idx].tolist()
            for position, token in enumerate(generated_tokens):
                if len(green_tokens_per_position) <= position:
                    green_tokens_per_position.append(set())
                green_tokens_per_position[position].add(token)

        # Second phase: Generate final sequence avoiding green tokens
        generator = torch.Generator(device=self.model.device).manual_seed(42)
        with self.generation_context(
            prompt=prompt,
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            select_green_tokens=select_green_tokens,
            multiple_key=multiple_key,
            num_keys=num_keys,
            context_width=context_width,
            **gen_kwargs,
        ) as ctx:
            position = 0
            while True:
                next_token_logits = ctx.step_with_watermark()

                # Get probability distribution
                probs = torch.softmax(next_token_logits[0], dim=-1)

                # Reduce probabilities for frequently appearing tokens
                if position < len(green_tokens_per_position):
                    green_tokens = green_tokens_per_position[position]
                    for token in green_tokens:
                        probs[token] *= reduction_factor
                    probs = probs / probs.sum()  # Renormalize

                # Sample token with modified probabilities
                token = torch.multinomial(probs, num_samples=1, generator=generator)
                should_continue = ctx.set_next_token(token)

                position += 1
                if not should_continue:
                    break

            return ctx.output_ids

    def generate_with_paraphrase_attack(
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
    ) -> torch.Tensor:
        # First phase: Generate watermarked text
        with torch.random.fork_rng():
            torch.manual_seed(42)
            watermarked_tokens = self.generate_with_watermark(
                prompt=prompt,
                gamma=gamma,
                delta=delta,
                seeding_scheme=seeding_scheme,
                select_green_tokens=select_green_tokens,
                multiple_key=multiple_key,
                num_keys=num_keys,
                context_width=context_width,
                **gen_kwargs,
            )

        # Decode the watermarked text
        generated_text = self.tokenizer.decode(
            watermarked_tokens[0], skip_special_tokens=True
        )

        # Second phase: Generate paraphrase
        paraphrase_prompt = (
            "Paraphrase the following text while preserving its meaning and factual accuracy. "
            "Avoid adding or removing information, and produce fluent, natural language.\n\n"
            f"Text to paraphrase:\n{generated_text}\n\nParaphrased version:"
        )

        # Generate paraphrase
        paraphrase_gen_kwargs = gen_kwargs | {
            "do_sample": True,
            "temperature": 0.9,
            "max_new_tokens": gen_kwargs.get(
                "max_new_tokens", watermarked_tokens.shape[1]
            ),
        }
        with torch.random.fork_rng():
            torch.manual_seed(42)
            paraphrase_outputs = self.generate(
                paraphrase_prompt, **paraphrase_gen_kwargs
            )

        return paraphrase_outputs

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
