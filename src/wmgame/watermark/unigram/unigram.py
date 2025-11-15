# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Unigram-Watermark/
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from transformers import BatchEncoding

from wmgame.watermark.base import GenerationContext, WatermarkedLLM
from wmgame.watermark.unigram.detector import UnigramDetector
from wmgame.watermark.utils.translator import Translator, get_default_translator
from wmgame.tasks import TRANSLATION_PROMPT_PREFIX, TRANSLATION_TARGET_LANGUAGE


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
        strength: float = 5.0,
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
        fraction: float,
        strength: float,
        watermark_key: int,
        multiple_key: bool = False,
        num_keys: int = 1,
        **gen_kwargs,
    ) -> torch.Tensor:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        if gen_kwargs.get("do_sample", False):
            generator = torch.Generator(device=self.model.device).manual_seed(42)
        else:
            generator = None
        with self.generation_context(
            prompt=inputs,
            fraction=fraction,
            strength=strength,
            watermark_key=watermark_key,
            multiple_key=multiple_key,
            num_keys=num_keys,
            **gen_kwargs,
        ) as ctx:  # type: ignore[arg-type]
            for _ in range(gen_kwargs.get("max_new_tokens", 1)):
                logits = ctx.step_with_watermark()
                probs = torch.nn.functional.softmax(logits, dim=-1)
                if generator is None:
                    token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    token = torch.multinomial(probs, 1, generator=generator)
                if not ctx.set_next_token(token):
                    break
            generated_token_ids = ctx.output_ids
        return generated_token_ids

    def generate_with_detection_attack(
        self,
        detector: UnigramDetector,
        prompt: str | BatchEncoding,
        fraction: float,
        strength: float,
        watermark_key: int,
        k: int,
        multiple_key: bool = False,
        num_keys: int = 1,
        **gen_kwargs,
    ) -> torch.Tensor:
        k = max(1, min(k, len(self.tokenizer)))
        detector_device = detector.device or torch.device("cpu")

        with self.generation_context(
            prompt=prompt,
            fraction=fraction,
            strength=strength,
            watermark_key=watermark_key,
            multiple_key=multiple_key,
            num_keys=num_keys,
            **gen_kwargs,
        ) as ctx:  # type: ignore[arg-type]
            prompt_length = ctx.input_ids.shape[1]
            while True:
                logits = ctx.step_with_watermark()
                _, top_k_indices = torch.topk(logits[0], k, dim=-1)
                top_k_candidates = top_k_indices.tolist()

                current_prefix = ctx.all_token_ids()[0]
                generated_prefix = current_prefix[prompt_length:]

                best_token = top_k_candidates[0]
                best_score = float("inf")

                for candidate_token in top_k_candidates:
                    candidate_seq = torch.cat(
                        [
                            generated_prefix,
                            torch.tensor(
                                [candidate_token], device=current_prefix.device
                            ),
                        ],
                        dim=0,
                    )
                    result = detector.detect_tokens(candidate_seq.to(detector_device))
                    z_score = result.z_score
                    if z_score is not None and z_score < best_score:
                        best_score = z_score
                        best_token = candidate_token

                token_tensor = torch.tensor([[best_token]], device=ctx.input_ids.device)
                should_continue = ctx.set_next_token(token_tensor)
                if not should_continue:
                    break

            return ctx.output_ids

    def generate_with_frequency_attack(
        self,
        prompt: str | BatchEncoding,
        fraction: float,
        strength: float,
        watermark_key: int,
        num_samples: int,
        reduction_factor: float,
        multiple_key: bool = False,
        num_keys: int = 1,
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
                fraction=fraction,
                strength=strength,
                watermark_key=watermark_key,
                multiple_key=multiple_key,
                num_keys=num_keys,
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
            fraction=fraction,
            strength=strength,
            watermark_key=watermark_key,
            multiple_key=multiple_key,
            num_keys=num_keys,
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
        fraction: float,
        strength: float,
        watermark_key: int,
        multiple_key: bool = False,
        num_keys: int = 1,
        **gen_kwargs,
    ) -> torch.Tensor:
        # First phase: Generate watermarked text
        with torch.random.fork_rng():
            torch.manual_seed(42)
            watermarked_tokens = self.generate_with_watermark(
                prompt=prompt,
                fraction=fraction,
                strength=strength,
                watermark_key=watermark_key,
                multiple_key=multiple_key,
                num_keys=num_keys,
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

    def generate_with_translation_attack(
        self,
        prompt: str | BatchEncoding,
        fraction: float,
        strength: float,
        watermark_key: int,
        multiple_key: bool = False,
        num_keys: int = 1,
        translator: Translator | None = None,
        src_lang: str = "English",
        pivot_lang: str = "Chinese",
        **gen_kwargs,
    ) -> torch.Tensor:
        if translator is None:
            translator = get_default_translator()

        # Step 1: Translate the prompt to the pivot language
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            prompt_text = self.tokenizer.decode(
                prompt["input_ids"][0], skip_special_tokens=True  # type: ignore
            )
        translated_prompt = translator.translate(
            text=prompt_text, src_lang=src_lang, tgt_lang=pivot_lang
        )

        # Step 2: Generate watermarked text in the pivot language
        with torch.random.fork_rng():
            torch.manual_seed(42)
            watermarked_tokens = self.generate_with_watermark(
                prompt=translated_prompt,
                fraction=fraction,
                strength=strength,
                watermark_key=watermark_key,
                multiple_key=multiple_key,
                num_keys=num_keys,
                **gen_kwargs,
            )
        watermarked_text = self.tokenizer.decode(
            watermarked_tokens[0], skip_special_tokens=True
        )

        # Step 3: Translate the watermarked text back to the source language
        final_lang = (
            TRANSLATION_TARGET_LANGUAGE
            if TRANSLATION_PROMPT_PREFIX in prompt_text
            else src_lang
        )
        final_text = translator.translate(
            text=watermarked_text, src_lang=pivot_lang, tgt_lang=final_lang
        )
        max_length = gen_kwargs.get("max_new_tokens")
        final_inputs = self.tokenizer(
            final_text, truncation=True, max_length=max_length, return_tensors="pt"
        )
        final_inputs = final_inputs.to(self.model.device)

        return final_inputs["input_ids"]  # type: ignore
