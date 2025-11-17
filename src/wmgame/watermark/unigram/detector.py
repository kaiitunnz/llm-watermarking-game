# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Unigram-Watermark/
from typing import Optional

import numpy as np
import torch

from wmgame.watermark.base import (
    WatermarkDetector as BaseDetector,
    DetectionResult,
    DetectionConfig,
)


class UnigramDetector(BaseDetector):
    """Unigram watermark detector using fixed green-list masks.

    Configuration (pass via __init__ or DetectionConfig):
      - fraction: float
      - vocab_size: int
      - watermark_key: int
      - multiple_key: bool
      - num_keys: int
    """

    def __init__(
        self,
        *,
        tokenizer=None,
        device: Optional[torch.device] = None,
        fraction: float = 0.5,
        vocab_size: int = 50257,
        watermark_key: int = 0,
        multiple_key: bool = False,
        num_keys: int = 1,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, config=config, device=device)
        self.fraction = fraction
        self.vocab_size = vocab_size
        self.watermark_key = watermark_key
        self.multiple_key = multiple_key
        self.num_keys = num_keys
        self._init_masks()

    def _hash(self, x: int) -> int:
        import hashlib

        xb = int(x).to_bytes(8, byteorder="little", signed=False)
        return int.from_bytes(hashlib.sha256(xb).digest()[:4], "little")

    def _init_masks(self) -> None:
        rng = np.random.default_rng(self._hash(self.watermark_key))
        base = np.array(
            [True] * int(self.fraction * self.vocab_size)
            + [False] * (self.vocab_size - int(self.fraction * self.vocab_size))
        )
        rng.shuffle(base)
        self.green_mask = torch.tensor(base, dtype=torch.float32)
        if self.multiple_key:
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
            for i in range(self.num_keys):
                rng = np.random.default_rng(self._hash(hash_key_list[i]))
                m = base.copy()
                rng.shuffle(m)
                self.green_mask_list.append(torch.tensor(m, dtype=torch.float32))

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)

    @staticmethod
    def _tau(m: int, N: int, alpha: float) -> float:
        from scipy.stats import norm

        factor = np.sqrt(1 - (m - 1) / (N - 1))
        return factor * norm.ppf(1 - alpha)

    def detect_tokens(self, input_ids: torch.Tensor) -> DetectionResult:
        seq = input_ids.detach().cpu().numpy().tolist()
        uniq = list(set(seq))
        if not self.multiple_key:
            num_green = int(sum(self.green_mask[i].item() for i in seq))
            z = float(self._z_score(num_green, len(seq), self.fraction)) if len(seq) > 0 else 0.0
        else:
            z_list = []
            num_green = 0
            for k in range(self.num_keys):
                gm = self.green_mask_list[k]
                gk = int(sum(gm[i].item() for i in seq))
                num_green = max(num_green, gk)
                z_list.append(self._z_score(gk, len(seq), self.fraction) if len(seq) > 0 else 0.0)
            z = float(np.max(np.array(z_list))) if len(z_list) > 0 else 0.0

        # decision by fixed threshold if provided
        passed: Optional[bool]
        if self.config.threshold is not None:
            passed = z >= float(self.config.threshold)
        elif self.config.alpha is not None:
            # dynamic threshold with unique tokens count and vocab size
            tau = self._tau(len(uniq), self.vocab_size, float(self.config.alpha))
            passed = z > tau
        else:
            passed = None

        return DetectionResult(
            passed=passed,
            z_score=z,
            p_value=None,
            num_tokens_scored=len(seq),
            num_green_tokens=num_green,
            green_fraction=(num_green / len(seq)) if len(seq) > 0 else 0.0,
            extras={},
        )

    def supports_dynamic_threshold(self) -> bool:
        return True

    def dynamic_threshold(self, input_ids: torch.Tensor) -> float:
        seq = input_ids.detach().cpu().numpy().tolist()
        uniq = set(seq)
        if self.config.alpha is None:
            raise ValueError("alpha must be set in DetectionConfig for dynamic threshold")
        return float(self._tau(len(uniq), self.vocab_size, float(self.config.alpha)))
