# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Exp-Watermark/
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import BatchEncoding
from transformers.generation.utils import GenerateOutput

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
            for i in range(m):
                probs = ctx.step_with_watermark()
                token = torch.multinomial(probs, 1)
                if not ctx.set_next_token(token):
                    break
            sequences = ctx.all_token_ids()

            
        prompt_length = inputs["input_ids"].shape[1]
        generated_token_ids = sequences[:, prompt_length:]
        
        return generated_token_ids

    def generate_with_detection_attack(
        self,
        prompt: str | BatchEncoding,
        *,
        n: int,
        m: int,
        seed: int,
        num_samples: int = 3,
        **gen_kwargs,
    ) -> GenerateOutput:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )

        detector = ExpDetector(
            tokenizer=self.tokenizer,
            n=n,
            k=1,
            gamma=0.5,
            seed=0,
            vocab_size=len(self.tokenizer),
        )

        # Generate multiple complete sequences
        all_sequences = []
        best_p_value = float('-inf')
        best_sequence = None

        for _ in range(num_samples):
            with self.generation_context(prompt=inputs, n=n, seed=seed, **gen_kwargs) as ctx:  # type: ignore[arg-type]
                # Generate complete sequence
                for _ in range(m):
                    logits = ctx.step_with_watermark()
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    token = torch.multinomial(probs, 1)
                    if not ctx.set_next_token(token):
                        break
                sequences = ctx.all_token_ids()
            
                # Extract generated tokens (excluding prompt)
                prompt_length = inputs["input_ids"].shape[1]
                generated_tokens = sequences[0, prompt_length:].tolist()
            
                # Evaluate entire sequence
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                result = detector.detect(generated_text)
                p_value = result.p_value
            
                # Update best sequence if this one has a higher z-score
                if p_value > best_p_value:
                    best_p_value = p_value
                    best_sequence = sequences[0]

        # If no valid sequence was found, return the last generated one
        if best_sequence is None:
            best_sequence = sequences[0]

        return best_sequence[prompt_length:].unsqueeze(0)


    def generate_with_frequency_attack(
        self,
        prompt: str | BatchEncoding,
        *,
        n: int,
        m: int,
        seed: int,
        num_samples: int = 1,
        reduction_factor: float = 0.9,
        **gen_kwargs,
    ) -> GenerateOutput:
        inputs = (
            self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt
        )
        
        # First phase: Collect green tokens from multiple samples
        green_tokens_per_position = []
    
        for _ in range(num_samples):
            with self.generation_context(prompt=inputs, n=n, seed=seed, **gen_kwargs) as ctx:  # type: ignore[arg-type]
                for _ in range(gen_kwargs.get("max_new_tokens", 1)):
                    logits = ctx.step_with_watermark()
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    token = torch.multinomial(probs, 1)
                    if not ctx.set_next_token(token):
                        break
                
                sequences = ctx.all_token_ids()
                generated_tokens = sequences[0, inputs["input_ids"].shape[1]:].tolist()
            
                # Track green tokens for each position
                for pos, token in enumerate(generated_tokens):
                    if len(green_tokens_per_position) <= pos:
                        green_tokens_per_position.append(set())
                    green_tokens_per_position[pos].add(token)

        # Second phase: Generate with reduced probabilities for green tokens
        with self.generation_context(prompt=inputs, n=n, seed=seed, **gen_kwargs) as ctx:  # type: ignore[arg-type]
            generated_tokens = []
        
            for pos in range(m):
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

        return torch.tensor([generated_tokens], device=inputs["input_ids"].device)
    
    def generate_with_paraphrase_attack(
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
            # simple greedy loop
            for _ in range(gen_kwargs.get("max_new_tokens", 1)):
                logits = ctx.step_with_watermark()
                probs = torch.nn.functional.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1)
                if not ctx.set_next_token(token):
                    break
            sequences = ctx.all_token_ids()
            
            original_sequence = sequences[0]
            prompt_length = inputs['input_ids'].shape[1]
            generated_text = self.tokenizer.decode(
                original_sequence[prompt_length:],
                skip_special_tokens=True
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
            'do_sample': True,
            'temperature': 0.9,
            'max_new_tokens': len(original_sequence) - prompt_length,  # Match original length
        }
        # Only add gen_kwargs that don't conflict with our explicit params
        generation_params.update({
            k: v for k, v in gen_kwargs.items() 
            if k not in ['max_new_tokens', 'do_sample', 'temperature','gamma','delta']
        })

        print(generation_params.max_new_tokens)

        paraphrase_outputs = self.model.generate(
            **paraphrase_inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            **generation_params
        )

        # Extract paraphrased text
        paraphrased_text = self.tokenizer.decode(
            paraphrase_outputs[0][paraphrase_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Convert paraphrased text back to tokens and combine with original prompt
        prompt_tokens = original_sequence[:prompt_length].tolist()
        paraphrase_tokens = self.tokenizer.encode(
            paraphrased_text,
            add_special_tokens=False
        )

        # Create final sequence
        final_sequence = torch.tensor(
            [paraphrase_tokens],
            device=original_sequence.device
        )

        return final_sequence

 
