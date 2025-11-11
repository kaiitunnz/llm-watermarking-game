from enum import Enum
from typing import Any

import torch
import numpy as np

from wmgame.watermark.base import (
    WatermarkDetector as BaseDetector,
    DetectionResult,
    DetectionConfig,
)


class EnsembleStrategy(Enum):
    """Strategy for combining detection results from multiple detectors."""

    VOTING = "voting"
    MAX_CONFIDENCE = "max_confidence"
    WEIGHTED_AVERAGE = "weighted_average"
    ANY_POSITIVE = "any_positive"
    ALL_POSITIVE = "all_positive"


class EnsembleDetector(BaseDetector):
    """Ensemble detector that combines multiple watermark detection schemes.

    Parameters
    ----------
    detectors : dict[str, BaseDetector]
        Dictionary mapping detector names to detector instances.
        Supported keys: "exp", "kgw", "unigram" (or custom names).
    weights : dict[str, float] | None
        Weights for each detector when using WEIGHTED_AVERAGE strategy.
        If None, equal weights are used. Keys should match detector names.
    strategy : EnsembleStrategy
        Strategy for combining detection results (default: MAX_CONFIDENCE).
    tokenizer : Any
        Tokenizer instance (inherited from base class).
    device : torch.device | None
        Device for computation (inherited from base class).
    config : DetectionConfig | None
        Detection configuration (inherited from base class).
        The threshold in config applies to the ensemble's combined score.

    Examples
    --------
    >>> from wmgame.watermark.exp.detector import ExpDetector
    >>> from wmgame.watermark.kgw.detector import KGWDetector
    >>> from wmgame.watermark.unigram.detector import UnigramDetector
    >>>
    >>> detectors = {
    ...     "exp": ExpDetector(...),
    ...     "kgw": KGWDetector(...),
    ...     "unigram": UnigramDetector(...),
    ... }
    >>> ensemble = EnsembleDetector(
    ...     detectors=detectors,
    ...     strategy=EnsembleStrategy.MAX_CONFIDENCE,
    ... )
    >>> result = ensemble.detect("Some text to check...")
    """

    def __init__(
        self,
        *,
        detectors: dict[str, BaseDetector],
        weights: dict[str, float] | None = None,
        strategy: EnsembleStrategy = EnsembleStrategy.MAX_CONFIDENCE,
        tokenizer: Any = None,
        device: torch.device | None = None,
        config: DetectionConfig | None = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, config=config, device=device)

        if not detectors:
            raise ValueError("At least one detector must be provided")

        self.detectors = detectors
        self.strategy = strategy

        # Set up weights
        if weights is None:
            self.weights = {name: 1.0 / len(detectors) for name in detectors}
        else:
            # Normalize weights to sum to 1
            total = sum(weights.values())
            self.weights = {name: w / total for name, w in weights.items()}

        # Validate weights match detectors
        if set(self.weights.keys()) != set(self.detectors.keys()):
            raise ValueError("Weight keys must match detector keys")

    def _normalize_score(self, result: DetectionResult, detector_name: str) -> float:
        """Normalize a detection result to a 0-1 confidence score.

        Higher values indicate higher confidence that text is watermarked.

        Parameters
        ----------
        result : DetectionResult
            Detection result from a single detector.
        detector_name : str
            Name of the detector (for scheme-specific normalization).

        Returns
        -------
        float
            Normalized confidence score in [0, 1].
        """
        # EXP detector: lower p-value = more confident watermark
        if result.p_value is not None and result.z_score is None:
            # Convert p-value to confidence (1 - p_value)
            return 1.0 - float(result.p_value)

        # KGW/Unigram: higher z-score = more confident watermark
        elif result.z_score is not None:
            # Use sigmoid to map z-score to [0, 1]
            # z > 4 is very confident, z < -4 is very unconfident
            z = float(result.z_score)
            return float(1.0 / (1.0 + np.exp(-z / 2.0)))

        # Fallback: use passed flag if available
        elif result.passed is not None:
            return 1.0 if result.passed else 0.0

        # No usable score
        return 0.5

    def detect_tokens(self, input_ids: torch.Tensor) -> DetectionResult:
        """Detect watermark using ensemble of detectors.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs to check for watermark.

        Returns
        -------
        DetectionResult
            Combined detection result with ensemble decision.
        """
        # Run all detectors
        results: dict[str, DetectionResult] = {}
        for name, detector in self.detectors.items():
            try:
                results[name] = detector.detect_tokens(input_ids)
            except Exception as e:
                # Log warning but continue with other detectors
                print(f"Warning: Detector '{name}' failed: {e}")
                continue

        if not results:
            # All detectors failed
            return DetectionResult(
                passed=None,
                z_score=None,
                p_value=None,
                num_tokens_scored=int(input_ids.numel()),
                num_green_tokens=None,
                green_fraction=None,
                extras={"error": "All detectors failed"},
            )

        # Combine results based on strategy
        if self.strategy == EnsembleStrategy.VOTING:
            return self._voting_combine(results, input_ids)
        elif self.strategy == EnsembleStrategy.MAX_CONFIDENCE:
            return self._max_confidence_combine(results, input_ids)
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_combine(results, input_ids)
        elif self.strategy == EnsembleStrategy.ANY_POSITIVE:
            return self._any_positive_combine(results, input_ids)
        elif self.strategy == EnsembleStrategy.ALL_POSITIVE:
            return self._all_positive_combine(results, input_ids)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _voting_combine(
        self,
        results: dict[str, DetectionResult],
        input_ids: torch.Tensor,
    ) -> DetectionResult:
        """Combine using majority voting."""
        votes_positive = 0
        votes_negative = 0

        for name, result in results.items():
            if result.passed is True:
                votes_positive += 1
            elif result.passed is False:
                votes_negative += 1
            else:
                # Use normalized score for uncertain cases
                score = self._normalize_score(result, name)
                if score >= 0.5:
                    votes_positive += 1
                else:
                    votes_negative += 1

        total_votes = votes_positive + votes_negative
        passed = votes_positive > votes_negative if total_votes > 0 else None
        confidence = votes_positive / total_votes if total_votes > 0 else 0.5

        return DetectionResult(
            passed=passed,
            z_score=None,
            p_value=1.0 - confidence,  # Convert to p-value style
            num_tokens_scored=int(input_ids.numel()),
            num_green_tokens=None,
            green_fraction=None,
            extras={
                "strategy": "voting",
                "votes_positive": votes_positive,
                "votes_negative": votes_negative,
                "confidence": confidence,
                "individual_results": {
                    name: {
                        "passed": r.passed,
                        "z_score": r.z_score,
                        "p_value": r.p_value,
                    }
                    for name, r in results.items()
                },
            },
        )

    def _max_confidence_combine(
        self,
        results: dict[str, DetectionResult],
        input_ids: torch.Tensor,
    ) -> DetectionResult:
        """Combine by selecting detector with highest confidence."""
        best_name = None
        best_score = -float("inf")
        best_result = None

        for name, result in results.items():
            # Normalize to confidence score
            score = self._normalize_score(result, name)
            # Convert to signed confidence (positive = watermarked)
            if result.passed is False or (result.passed is None and score < 0.5):
                score = -(1.0 - score)

            if abs(score - 0.5) > abs(best_score - 0.5):
                best_score = score
                best_name = name
                best_result = result

        if best_result is None:
            return DetectionResult(
                passed=None,
                z_score=None,
                p_value=None,
                num_tokens_scored=int(input_ids.numel()),
                num_green_tokens=None,
                green_fraction=None,
                extras={"error": "No valid results"},
            )

        # Determine passed based on threshold
        passed = best_result.passed
        if (
            passed is None
            and self.config.threshold is not None
            and best_name is not None
        ):
            confidence = self._normalize_score(best_result, best_name)
            passed = confidence >= self.config.threshold

        return DetectionResult(
            passed=passed,
            z_score=best_result.z_score,
            p_value=best_result.p_value,
            num_tokens_scored=best_result.num_tokens_scored,
            num_green_tokens=best_result.num_green_tokens,
            green_fraction=best_result.green_fraction,
            extras={
                "strategy": "max_confidence",
                "best_detector": best_name,
                "confidence": best_score,
                "individual_results": {
                    name: {
                        "passed": r.passed,
                        "z_score": r.z_score,
                        "p_value": r.p_value,
                    }
                    for name, r in results.items()
                },
            },
        )

    def _weighted_average_combine(
        self,
        results: dict[str, DetectionResult],
        input_ids: torch.Tensor,
    ) -> DetectionResult:
        """Combine using weighted average of normalized scores."""
        weighted_sum = 0.0
        total_weight = 0.0

        for name, result in results.items():
            score = self._normalize_score(result, name)
            weight = self.weights.get(name, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        avg_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Determine passed based on threshold
        passed: bool | None = None
        if self.config.threshold is not None:
            passed = avg_confidence >= self.config.threshold

        # Convert confidence to z-score-like value for consistency
        # Map [0, 1] to approximately [-4, 4]
        z_equiv = float((avg_confidence - 0.5) * 8.0)

        return DetectionResult(
            passed=passed,
            z_score=z_equiv,
            p_value=1.0 - avg_confidence,
            num_tokens_scored=int(input_ids.numel()),
            num_green_tokens=None,
            green_fraction=None,
            extras={
                "strategy": "weighted_average",
                "confidence": avg_confidence,
                "weights": self.weights,
                "individual_results": {
                    name: {
                        "passed": r.passed,
                        "z_score": r.z_score,
                        "p_value": r.p_value,
                    }
                    for name, r in results.items()
                },
            },
        )

    def _any_positive_combine(
        self,
        results: dict[str, DetectionResult],
        input_ids: torch.Tensor,
    ) -> DetectionResult:
        """Watermarked if ANY detector reports positive."""
        any_positive = False
        max_confidence = 0.0
        best_result = None

        for name, result in results.items():
            if result.passed is True:
                any_positive = True
                score = self._normalize_score(result, name)
                if score > max_confidence:
                    max_confidence = score
                    best_result = result
            elif result.passed is None:
                # Check normalized score
                score = self._normalize_score(result, name)
                if score >= 0.5:
                    any_positive = True
                if score > max_confidence:
                    max_confidence = score
                    best_result = result

        if best_result is None and results:
            # All negative, pick first
            best_result = next(iter(results.values()))

        return DetectionResult(
            passed=any_positive,
            z_score=best_result.z_score if best_result else None,
            p_value=best_result.p_value if best_result else None,
            num_tokens_scored=int(input_ids.numel()),
            num_green_tokens=best_result.num_green_tokens if best_result else None,
            green_fraction=best_result.green_fraction if best_result else None,
            extras={
                "strategy": "any_positive",
                "any_positive": any_positive,
                "max_confidence": max_confidence,
                "individual_results": {
                    name: {
                        "passed": r.passed,
                        "z_score": r.z_score,
                        "p_value": r.p_value,
                    }
                    for name, r in results.items()
                },
            },
        )

    def _all_positive_combine(
        self,
        results: dict[str, DetectionResult],
        input_ids: torch.Tensor,
    ) -> DetectionResult:
        """Watermarked only if ALL detectors report positive."""
        all_positive = True
        min_confidence = 1.0
        weakest_result = None

        for name, result in results.items():
            score = self._normalize_score(result, name)
            if result.passed is False or (result.passed is None and score < 0.5):
                all_positive = False
            if score < min_confidence:
                min_confidence = score
                weakest_result = result

        if weakest_result is None and results:
            weakest_result = next(iter(results.values()))

        return DetectionResult(
            passed=all_positive if results else None,
            z_score=weakest_result.z_score if weakest_result else None,
            p_value=weakest_result.p_value if weakest_result else None,
            num_tokens_scored=int(input_ids.numel()),
            num_green_tokens=(
                weakest_result.num_green_tokens if weakest_result else None
            ),
            green_fraction=weakest_result.green_fraction if weakest_result else None,
            extras={
                "strategy": "all_positive",
                "all_positive": all_positive,
                "min_confidence": min_confidence,
                "individual_results": {
                    name: {
                        "passed": r.passed,
                        "z_score": r.z_score,
                        "p_value": r.p_value,
                    }
                    for name, r in results.items()
                },
            },
        )

    def to(self, device: torch.device) -> "EnsembleDetector":
        """Move ensemble and all sub-detectors to specified device."""
        self.device = device
        for detector in self.detectors.values():
            detector.to(device)
        return self
