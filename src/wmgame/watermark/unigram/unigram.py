# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Unigram-Watermark/
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from transformers import BatchEncoding

from wmgame.watermark.base import GenerationContext, WatermarkedLLM
from wmgame.watermark.unigram.detector import UnigramDetector


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
        *,
        fraction: float = 0.5,
        strength: float = 5.0,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
        **gen_kwargs,
    ) -> torch.Tensor:
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

            prompt_length = inputs["input_ids"].shape[1]  # type: ignore
            generated_token_ids = sequences[:, prompt_length:]

            return generated_token_ids

    def generate_with_detection_attack(
        self,
        prompt: str | BatchEncoding,
        *,
        fraction: float = 0.5,
        strength: float = 5.0,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
        num_samples: int = 5,
        **gen_kwargs,
    ) -> torch.Tensor:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )

        detector = UnigramDetector(
            tokenizer=self.tokenizer,
            fraction=fraction,
            vocab_size=len(self.tokenizer),
            watermark_key=watermark_key,
        )

        # Generate multiple complete sequences
        sequences: torch.Tensor | None = None
        best_zscore = float("inf")
        best_sequence = None

        for _ in range(num_samples):
            with self.generation_context(
                prompt=inputs,
                fraction=fraction,
                strength=strength,
                watermark_key=watermark_key,
                multiple_key=multiple_key,
                num_keys=num_keys,
                **gen_kwargs,
            ) as ctx:
                # Generate complete sequence
                for _ in range(gen_kwargs.get("max_new_tokens", 1)):
                    logits = ctx.step_with_watermark()
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    token = torch.multinomial(probs, 1)
                    if not ctx.set_next_token(token):
                        break
                sequences = ctx.all_token_ids()

                # Extract generated tokens (excluding prompt)
                prompt_length = inputs["input_ids"].shape[1]  # type: ignore
                sequences = sequences[0, prompt_length:]
                generated_tokens = sequences.tolist()

                # Evaluate entire sequence
                generated_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                result = detector.detect(generated_text)
                z_score = result.z_score

                # Update best sequence if this one has a higher z-score
                assert z_score is not None and best_zscore is not None
                if z_score < best_zscore:
                    best_zscore = z_score
                    best_sequence = sequences

        # If no valid sequence was found, return the last generated one
        if best_sequence is None:
            assert sequences is not None
            best_sequence = sequences

        return best_sequence.unsqueeze(0)

    def generate_with_frequency_attack(
        self,
        prompt: str | BatchEncoding,
        *,
        fraction: float = 0.5,
        strength: float = 5.0,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
        num_samples: int = 5,
        reduction_factor: float = 0.5,
        **gen_kwargs,
    ) -> torch.Tensor:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )

        # First phase: Collect green tokens from multiple samples
        green_tokens_per_position = []

        for _ in range(num_samples):
            with self.generation_context(
                prompt=inputs,
                fraction=fraction,
                strength=strength,
                watermark_key=watermark_key,
                multiple_key=multiple_key,
                num_keys=num_keys,
                **gen_kwargs,
            ) as ctx:
                for _ in range(gen_kwargs.get("max_new_tokens", 1)):
                    logits = ctx.step_with_watermark()
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    token = torch.multinomial(probs, 1)
                    if not ctx.set_next_token(token):
                        break

                sequences = ctx.all_token_ids()
                prompt_length = inputs["input_ids"].shape[1]  # type: ignore
                generated_tokens = sequences[0, prompt_length:].tolist()

                # Track green tokens for each position
                for pos, token in enumerate(generated_tokens):
                    if len(green_tokens_per_position) <= pos:
                        green_tokens_per_position.append(set())
                    green_tokens_per_position[pos].add(token)

        # Second phase: Generate with reduced probabilities for green tokens
        with self.generation_context(
            prompt=inputs,
            fraction=fraction,
            strength=strength,
            watermark_key=watermark_key,
            multiple_key=multiple_key,
            num_keys=num_keys,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        ) as ctx:
            generated_tokens = []

            for pos in range(gen_kwargs.get("max_new_tokens", 1)):
                logits = ctx.step_with_watermark()
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # Reduce probabilities for tokens that appeared in green list
                if pos < len(green_tokens_per_position):
                    green_tokens = green_tokens_per_position[pos]
                    for token in green_tokens:
                        probs[0, token] *= reduction_factor

                # Renormalize probabilities
                probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample token with modified probabilities
                token = torch.multinomial(probs[0], num_samples=1)
                generated_tokens.append(token.item())

                if not ctx.set_next_token(token):
                    break

        return torch.tensor([generated_tokens], device=inputs["input_ids"].device)  # type: ignore

    def generate_with_paraphrase_attack(
        self,
        prompt: str | BatchEncoding,
        *,
        fraction: float = 0.5,
        strength: float = 5.0,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
        **gen_kwargs,
    ) -> torch.Tensor:
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

            original_sequence = sequences[0]
            prompt_length = inputs["input_ids"].shape[1]  # type: ignore
            generated_text = self.tokenizer.decode(
                original_sequence[prompt_length:], skip_special_tokens=True
            )

        # Create paraphrase prompt
        paraphrase_prompt = (
            "Paraphrase the following text while preserving its meaning, factual accuracy. "
            "Avoid adding or removing information, and produce fluent, natural language.\n\n"
            f"Text to paraphrase:\n{generated_text}\n\nParaphrased version:"
        )

        # Generate paraphrase without watermark
        paraphrase_inputs = self.tokenize(paraphrase_prompt, gen_kwargs)
        generation_params = {
            "do_sample": True,
            "temperature": 0.9,
            "max_new_tokens": len(original_sequence)
            - prompt_length,  # Match original length
        }
        # Only add gen_kwargs that don't conflict with our explicit params
        generation_params.update(
            {
                k: v
                for k, v in gen_kwargs.items()
                if k
                not in ["max_new_tokens", "do_sample", "temperature", "gamma", "delta"]
            }
        )

        paraphrase_outputs = self.model.generate(
            **paraphrase_inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            **generation_params,
        )

        # Extract paraphrased text
        prompt_length = paraphrase_inputs["input_ids"].shape[1]  # type: ignore
        paraphrased_text = self.tokenizer.decode(
            paraphrase_outputs[0][prompt_length:], skip_special_tokens=True
        )

        # Convert paraphrased text back to tokens and combine with original prompt
        paraphrase_tokens = self.tokenizer.encode(
            paraphrased_text, add_special_tokens=False
        )

        # Create final sequence
        final_sequence = torch.tensor(
            [paraphrase_tokens], device=original_sequence.device
        )

        return final_sequence
