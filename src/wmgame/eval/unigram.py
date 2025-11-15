import torch

from wmgame.eval.base import BaseRunner
from wmgame.watermark.base import DetectionConfig, DetectionResult
from wmgame.watermark.unigram import UnigramDetector, UnigramWatermarkedLLM


class UnigramRunner(BaseRunner):
    name = "unigram"

    def __init__(self, model_name: str) -> None:
        self.max_new_tokens = 128
        self.fraction = 0.5
        self.strength = 5.0
        self.watermark_key = 0
        self.z_threshold = 4.0
        self.llm: UnigramWatermarkedLLM
        self.detector: UnigramDetector
        super().__init__(model_name)

    def _create_llm(self, model_name: str) -> UnigramWatermarkedLLM:
        return UnigramWatermarkedLLM(model_name)

    def _create_detector(self) -> UnigramDetector:
        return UnigramDetector(
            tokenizer=self.llm.tokenizer,
            fraction=self.fraction,
            vocab_size=len(self.llm.tokenizer),
            watermark_key=self.watermark_key,
            config=DetectionConfig(threshold=self.z_threshold),
        )

    def generate(self, attack_method: str, prompt: str) -> torch.Tensor:
        base_kwargs: dict = dict(
            prompt=prompt,
            fraction=self.fraction,
            strength=self.strength,
            watermark_key=self.watermark_key,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        if attack_method == "none":
            return self.llm.generate_with_watermark(**base_kwargs)
        if attack_method == "detection":
            return self.llm.generate_with_detection_attack(
                detector=self.detector, k=5, **base_kwargs
            )
        if attack_method == "frequency":
            return self.llm.generate_with_frequency_attack(
                num_samples=10, reduction_factor=0.1, **base_kwargs
            )
        if attack_method == "paraphrase":
            return self.llm.generate_with_paraphrase_attack(**base_kwargs)
        if attack_method == "translation":
            return self.llm.generate_with_translation_attack(**base_kwargs)
        raise ValueError(f"Unsupported attack '{attack_method}' for Unigram")

    def detect(
        self, generated_tokens: torch.Tensor, generated_text: str
    ) -> DetectionResult:
        del generated_tokens
        return self.detector.detect(generated_text)
