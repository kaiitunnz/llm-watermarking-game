import torch

from wmgame.eval.base import BaseRunner
from wmgame.watermark.base import DetectionConfig, DetectionResult
from wmgame.watermark.kgw import KGWDetector, KGWWatermarkedLLM


class KGWRunner(BaseRunner):
    name = "kgw"

    def __init__(self, model_name: str) -> None:
        self.max_new_tokens = 128
        self.gamma = 0.5
        self.delta = 2.0
        self.z_threshold = 2.0
        self.llm: KGWWatermarkedLLM
        self.detector: KGWDetector
        super().__init__(model_name)

    def _create_llm(self, model_name: str) -> KGWWatermarkedLLM:
        return KGWWatermarkedLLM(model_name)

    def _create_detector(self) -> KGWDetector:
        return KGWDetector(
            tokenizer=self.llm.tokenizer,
            gamma=self.gamma,
            z_threshold=self.z_threshold,
            config=DetectionConfig(threshold=self.z_threshold),
        )

    def generate(self, attack_method: str, prompt: str) -> torch.Tensor:
        base_kwargs: dict = dict(
            prompt=prompt,
            gamma=self.gamma,
            delta=self.delta,
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
        raise ValueError(f"Unsupported attack '{attack_method}' for KGW")

    def detect(
        self, generated_tokens: torch.Tensor, generated_text: str
    ) -> DetectionResult:
        del generated_tokens
        return self.detector.detect(generated_text)
