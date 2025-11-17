import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from wmgame.watermark.base import DetectionResult, WatermarkDetector, WatermarkedLLM

SUPPORTED_ATTACKS = {"none", "detection", "frequency", "paraphrase", "translation"}
StrategySamplerFn = Callable[[random.Random], tuple[str, str]]


def _normalize_distribution(dist: dict[str, float]) -> dict[str, float]:
    total = float(sum(dist.values()))
    return {k: float(v) / total for k, v in dist.items()}


def _sample_from_distribution(dist: dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    last_key = ""
    for key, prob in dist.items():
        cumulative += prob
        last_key = key
        if r <= cumulative or math.isclose(cumulative, 1.0):
            return key
    return last_key


class StrategyData(TypedDict):
    solution_type: str
    defender_strategy: dict[str, float]
    attacker_strategy: dict[str, float]


@dataclass
class StrategySampler:
    strategy_type: str
    sample_fn: StrategySamplerFn
    watermark_actions: set[str]
    attack_actions: set[str]
    description: str

    @classmethod
    def from_path(cls, strategy_path: Path) -> "StrategySampler":
        with strategy_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: StrategyData) -> "StrategySampler":
        stype = str(data["solution_type"]).lower()
        watermark_dist = _normalize_distribution(
            {k.lower(): float(v) for k, v in data["defender_strategy"].items()}
        )
        attack_dist = _normalize_distribution(
            {k.lower(): float(v) for k, v in data["attacker_strategy"].items()}
        )
        description = "stackelberg" if stype == "stackelberg" else "nash"
        return cls(
            strategy_type=stype,
            sample_fn=lambda rng: (
                _sample_from_distribution(watermark_dist, rng),
                _sample_from_distribution(attack_dist, rng),
            ),
            watermark_actions=set(watermark_dist.keys()),
            attack_actions=set(attack_dist.keys()),
            description=description,
        )

    def sample(self, rng: random.Random) -> tuple[str, str]:
        return self.sample_fn(rng)


def seed_everything(seed: int) -> random.Random:
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng


class BaseRunner(ABC):
    name: str

    def __init__(
        self,
        model_or_name: str | tuple[LlamaForCausalLM, LlamaTokenizer],
        detection_threshold: float = 0,
    ) -> None:
        self.detection_threshold = detection_threshold
        if isinstance(model_or_name, str) and model_or_name.lower() == "dummy":
            self.llm = None
            self.detector = None
        else:
            self.llm = self._create_llm(model_or_name)
            self.detector = self._create_detector()

    @abstractmethod
    def _create_llm(
        self, model_or_name: str | tuple[LlamaForCausalLM, LlamaTokenizer]
    ) -> WatermarkedLLM:
        pass

    @abstractmethod
    def _create_detector(self) -> WatermarkDetector:
        pass

    @abstractmethod
    def generate(self, attack_method: str, prompt: str) -> torch.Tensor:
        pass

    def decode(self, generated_tokens: torch.Tensor) -> str:
        assert self.llm is not None
        return self.llm.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    @abstractmethod
    def detect(
        self, generated_tokens: torch.Tensor, generated_text: str
    ) -> DetectionResult:
        pass
