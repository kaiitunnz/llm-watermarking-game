import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from wmgame.eval.base import BaseRunner
from wmgame.watermark.base import DetectionConfig, DetectionResult
from wmgame.watermark.exp import ExpDetector, ExpWatermarkedLLM


class ExpRunner(BaseRunner):
    name = "exp"

    def __init__(
        self, model_or_name: str | tuple[LlamaForCausalLM, LlamaTokenizer]
    ) -> None:
        self.max_new_tokens = 128
        self.k = self.max_new_tokens
        self.n = 256
        self.seed = 0
        self.gamma = 0.3
        self.llm: ExpWatermarkedLLM
        self.detector: ExpDetector
        super().__init__(model_or_name, detection_threshold=0.05)

    def _create_llm(
        self, model_or_name: str | tuple[LlamaForCausalLM, LlamaTokenizer]
    ) -> ExpWatermarkedLLM:
        return ExpWatermarkedLLM(model_or_name)

    def _create_detector(self) -> ExpDetector:
        return ExpDetector(
            tokenizer=self.llm.tokenizer,
            n=self.n,
            k=self.k,
            gamma=self.gamma,
            seed=self.seed,
            vocab_size=len(self.llm.tokenizer),
            config=DetectionConfig(threshold=self.detection_threshold),
        )

    def generate(self, attack_method: str, prompt: str) -> torch.Tensor:
        base_kwargs: dict = dict(
            prompt=prompt,
            n=self.n,
            seed=self.seed,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        if attack_method == "none":
            return self.llm.generate_with_watermark(**base_kwargs)
        if attack_method == "detection":
            return self.llm.generate_with_detection_attack(
                detector=self.detector, num_samples=10, **base_kwargs
            )
        if attack_method == "frequency":
            return self.llm.generate_with_frequency_attack(
                num_samples=10, reduction_factor=0.1, **base_kwargs
            )
        if attack_method == "paraphrase":
            return self.llm.generate_with_paraphrase_attack(**base_kwargs)
        if attack_method == "translation":
            return self.llm.generate_with_translation_attack(**base_kwargs)
        raise ValueError(f"Unsupported attack '{attack_method}' for Exp")

    def detect(
        self, generated_tokens: torch.Tensor, generated_text: str
    ) -> DetectionResult:
        del generated_text
        return self.detector.detect_tokens(generated_tokens[0].cpu())
