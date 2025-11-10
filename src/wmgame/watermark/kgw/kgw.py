from collections.abc import Iterator
from contextlib import contextmanager

from click import prompt
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
        outputs =  self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([wm_processor]),
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )  # type: ignore

        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        
        return generated_tokens
    
    def generate_with_detection_attack(
        self,
        detector,
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

        inputs = (self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt)

        return_value = self.model.generate( **inputs, logits_processor=LogitsProcessorList([wm_processor]), **gen_kwargs,)

        outputs = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([wm_processor]),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )

        # Get original sequence (full sequence including prompt)
        original_sequence = outputs.sequences[0]  # shape: (total_length,), assuming batch_size=1
        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = original_sequence[prompt_length:].tolist()
    
        #print(f"Prompt length: {prompt_length}")
        #print(f"Generated sequence length: {len(generated_tokens)}")
    
        # Extract top-k alternatives for each position
        k = 5
        k = min(k, len(self.tokenizer))
        top_k_candidates = []
    
        for step_logits in outputs.scores:
            values, indices = torch.topk(step_logits[0], k, dim=-1)  # batch_size=1
            top_k_candidates.append(indices.tolist())
    
        # Greedy replacement: replace each token one at a time with best p-value
        best_generated = generated_tokens.copy()
    
        for pos in range(len(generated_tokens)):
            best_pvalue = -1
            best_token = generated_tokens[pos]
        
            # Try each candidate at this position
            for candidate_token in top_k_candidates[pos]:
                # Create test sequence with this candidate
                test_generated = best_generated.copy()
                test_generated[pos] = candidate_token
            
                # Decode ONLY the generated part (not including prompt)
                test_text = self.tokenizer.decode(test_generated, skip_special_tokens=True)
                result = detector.detect(test_text)
            
                # P-value from z-score (assuming two-tailed test)
                from scipy.stats import norm
                p_value = 2 * (1 - norm.cdf(abs(result.z_score)))

                result = detector.detect(test_text)
                p_value = result.p_value
            
                token_str = self.tokenizer.decode([candidate_token])
            
                if p_value > best_pvalue:
                    best_pvalue = p_value
                    best_token = candidate_token
        
            # Update sequence with best token
            best_generated[pos] = best_token
            best_token_str = self.tokenizer.decode([best_token])


        # Reconstruct full sequence: prompt + attacked generated tokens
        # Match the format of outputs.sequences
        prompt_tokens = original_sequence[:prompt_length].tolist()
        best_sequence = torch.tensor([best_generated], device=original_sequence.device)
    
        return best_sequence

    def generate_with_detection_attack(
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
        inputs = (self.tokenize(prompt, gen_kwargs) if isinstance(prompt, str) else prompt)

        # First phase: Generate multiple samples to identify green tokens
        green_tokens_per_position = []
    
        for _ in range(num_samples):
            outputs = self.model.generate(
                **inputs,
                logits_processor=LogitsProcessorList([wm_processor]),
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs
            )
        
            prompt_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs.sequences[0][prompt_length:].tolist()
        
            # Track green tokens for each position
            for pos, token in enumerate(generated_tokens):
                if len(green_tokens_per_position) <= pos:
                    green_tokens_per_position.append(set())
                green_tokens_per_position[pos].add(token)

        # Second phase: Generate final sequence avoiding green tokens
        outputs = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([wm_processor]),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )

        original_sequence = outputs.sequences[0]
        prompt_length = inputs['input_ids'].shape[1]
        best_generated = []

        # Process each position with reduced probabilities for green tokens
        for pos, step_logits in enumerate(outputs.scores):
            probs = torch.softmax(step_logits[0], dim=-1)
        
            # Reduce probabilities for tokens that appeared in green list
            if pos < len(green_tokens_per_position):
                green_tokens = green_tokens_per_position[pos]
                for token in green_tokens:
                    probs[token] *= reduction_factor
            
            # Renormalize probabilities
            probs = probs / probs.sum()
        
            # Sample token with modified probabilities
            token = torch.multinomial(probs, num_samples=1)[0].item()
            best_generated.append(token)

        # Reconstruct final sequence
        prompt_tokens = original_sequence[:prompt_length].tolist()
        best_sequence = torch.tensor([best_generated], device=original_sequence.device)
        return best_sequence
    
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
            # Generate initial watermarked text

        outputs = self.model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([wm_processor]),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )

        # Get the generated text
        original_sequence = outputs.sequences[0]
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
            if k not in ['max_new_tokens', 'do_sample', 'temperature']
        })
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
