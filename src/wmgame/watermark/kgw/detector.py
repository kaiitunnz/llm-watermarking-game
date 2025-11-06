# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/KGW-Watermark/
from typing import Iterable, Optional

import torch

from wmgame.watermark.base import (
    WatermarkDetector as BaseDetector,
    DetectionResult,
    DetectionConfig,
)
from wmgame.watermark.kgw.wm_processor import (
    WatermarkDetector as _KGWLibDetector,
)


class KGWDetector(BaseDetector):
    """Detector for the KGW scheme wrapping the local implementation.

    Parameters
    ----------
    gamma: float
        Fraction of vocab assigned to green list.
    seeding_scheme: str
        RNG seeding scheme (default: "simple_1").
    normalizers: Iterable[str]
        Text normalizers to apply before tokenization.
    ignore_repeated_bigrams: bool
        If True, score unique bigrams (as in KGW default variant).
    multiple_key: bool
        Whether to consider multiple keys.
    num_keys: int
        Number of keys when multiple_key is True.
    context_width: int
        Seeding window size.
    z_threshold: float
        Default z threshold for pass/fail if config.threshold is None.
    """

    def __init__(
        self,
        *,
        tokenizer=None,
        device: Optional[torch.device] = None,
        gamma: float = 0.5,
        seeding_scheme: str = "simple_1",
        normalizers: Iterable[str] = (),
        ignore_repeated_bigrams: bool = True,
        multiple_key: bool = False,
        num_keys: int = 1,
        context_width: int = 1,
        z_threshold: float = 4.0,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, config=config, device=device)
        self._z_threshold_default = z_threshold
        detector_device = self.device if self.device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._det = _KGWLibDetector(
            vocab=list(tokenizer.get_vocab().values()) if tokenizer else [],
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            device=detector_device,
            tokenizer=tokenizer,
            z_threshold=z_threshold,
            normalizers=list(normalizers),
            ignore_repeated_bigrams=ignore_repeated_bigrams,
            multiple_key=multiple_key,
            context_width=context_width,
        )

    def detect_tokens(self, input_ids: torch.Tensor) -> DetectionResult:
        # Use library scoring directly on token ids
        score_dict, _ = self._det._score_sequence(
            input_ids,
            return_num_tokens_scored=True,
            return_num_green_tokens=True,
            return_green_fraction=True,
            return_green_token_mask=False,
            return_z_score=True,
            return_p_value=True,
        )
        z = float(score_dict.get("z_score", 0.0))
        p = float(score_dict.get("p_value", 1.0))
        num_tokens = int(score_dict.get("num_tokens_scored", 0))
        num_green = int(score_dict.get("num_green_tokens", 0))
        green_frac = float(score_dict.get("green_fraction", 0.0))

        threshold = (
            self.config.threshold if self.config.threshold is not None else self._z_threshold_default
        )
        passed = z >= threshold if threshold is not None else None

        return DetectionResult(
            passed=passed,
            z_score=z,
            p_value=p,
            num_tokens_scored=num_tokens,
            num_green_tokens=num_green,
            green_fraction=green_frac,
            extras={},
        )
