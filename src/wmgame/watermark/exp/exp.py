# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Exp-Watermark/
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import BatchEncoding

from wmgame.watermark.base import GenerationContext, WatermarkedLLM
from wmgame.watermark.exp.gumbel import gumbel_key_func, gumbel_query
from wmgame.watermark.exp.detector import ExpDetector


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
        random_offset: bool = True,
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
        self._xi = xi  # [n, V]
        self._pi = pi.unsqueeze(0)  # [1, V]
        self._step = 0
        self._batch_size = input_ids.size(0)
        if random_offset:
            self._offsets = torch.randint(0, n, (self._batch_size,), dtype=torch.long)
        else:
            self._offsets = torch.zeros(self._batch_size, dtype=torch.long)

    def step_with_watermark(self) -> torch.Tensor:
        logits = super().step()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pi_slice, xi_slice = self._current_key_slices()
        probs = gumbel_query(probs, pi_slice, xi_slice)
        self._step += 1
        return probs

    def _current_key_slices(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._batch_size == 1:
            offsets = (self._offsets + self._step) % self._n
            xi = self._xi[offsets]
        else:
            offsets = (self._offsets + self._step) % self._n
            xi = self._xi.index_select(0, offsets)
        if self._pi.size(0) == 1 and self._batch_size > 1:
            pi = self._pi.expand(self._batch_size, -1)
        else:
            pi = self._pi
        return pi, xi


class ExpWatermarkedLLM(WatermarkedLLM):
    @contextmanager
    def generation_context(
        self,
        prompt: str | BatchEncoding,
        *,
        n: int,
        seed: int,
        random_offset: bool = True,
        **gen_kwargs,
    ) -> Iterator[ExpGenerationContext]:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer)
        yield ExpGenerationContext(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=inputs["input_ids"],  # type: ignore
            vocab_size=vocab_size,
            max_new_tokens=gen_kwargs.get("max_new_tokens", 1),
            n=n,
            seed=seed,
            random_offset=random_offset,
        )

    def generate_with_watermark(
        self,
        prompt: str | BatchEncoding,
        n: int,
        seed: int,
        random_offset: bool = True,
        **gen_kwargs,
    ) -> torch.Tensor:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        if gen_kwargs.get("do_sample", False):
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
        else:
            generator = None
        with self.generation_context(
            inputs, n=n, seed=seed, random_offset=random_offset, **gen_kwargs
        ) as ctx:
            while True:
                probs = ctx.step_with_watermark()
                if generator is None:
                    token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    token = torch.multinomial(probs, num_samples=1, generator=generator)
                if not ctx.set_next_token(token):
                    break
            generated_token_ids = ctx.output_ids
        return generated_token_ids

    def generate_with_detection_attack(
        self,
        detector: ExpDetector,
        prompt: str | BatchEncoding,
        n: int,
        seed: int,
        num_samples: int,
        **gen_kwargs,
    ) -> torch.Tensor:
        # Generate multiple candidate sequences with watermark
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
            torch.manual_seed(seed)
            outputs = self.generate_with_watermark(
                prompt=inputs,
                n=n,
                seed=seed,
                random_offset=True,
                **gen_kwargs | {"do_sample": True, "temperature": 1.0},
            )

        # Evaluate each candidate sequence with the detector and select best
        best_pvalue = -1
        best_sequence_idx = 0

        for batch_idx in range(num_samples):
            generated_tokens = outputs[batch_idx].cpu()
            result = detector.detect_tokens(generated_tokens)
            p_value = result.p_value
            if p_value is not None and p_value > best_pvalue:
                best_pvalue = p_value
                best_sequence_idx = batch_idx

        # Return the sequence with highest p-value
        return outputs[best_sequence_idx : best_sequence_idx + 1]

    def generate_with_frequency_attack(
        self,
        prompt: str | BatchEncoding,
        n: int,
        seed: int,
        num_samples: int,
        reduction_factor: float,
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
            torch.manual_seed(seed)
            outputs = self.generate_with_watermark(
                prompt=inputs,
                n=n,
                seed=seed,
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
        generator = torch.Generator(device=self.model.device).manual_seed(seed)
        with self.generation_context(prompt, n=n, seed=seed, **gen_kwargs) as ctx:
            position = 0
            while True:
                probs = ctx.step_with_watermark()[0]

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
        n: int,
        seed: int,
        **gen_kwargs,
    ) -> torch.Tensor:
        # First phase: Generate watermarked text
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            watermarked_tokens = self.generate_with_watermark(
                prompt=prompt, n=n, seed=seed, **gen_kwargs
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
            torch.manual_seed(seed)
            paraphrase_outputs = self.generate(
                paraphrase_prompt, **paraphrase_gen_kwargs
            )

        return paraphrase_outputs
