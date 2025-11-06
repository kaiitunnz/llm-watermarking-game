from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional, TYPE_CHECKING

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    DynamicCache,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.generation.utils import GenerateOutput


class GenerationContext(ABC):
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        input_ids: torch.Tensor,
        vocab_size: int,
        max_new_tokens: int,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.input_ids = input_ids.to(model.device)
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens

        self._attention_mask = torch.ones_like(input_ids)
        self._past_key_values = DynamicCache(config=model.config)
        self._num_tokens = 0
        self.output_ids: torch.Tensor | None = None

    def step(self) -> torch.Tensor:
        with torch.no_grad():
            if self.output_ids is None:
                output = self.model(
                    self.input_ids,
                    past_key_values=self._past_key_values,
                    attention_mask=self._attention_mask,
                )
            else:
                output = self.model(
                    self.output_ids[:, -1:],
                    past_key_values=self._past_key_values,
                    attention_mask=None,
                )
            logits = output.logits[:, -1, : self.vocab_size]
            self._past_key_values = output.past_key_values
            attn = self._attention_mask
            self._attention_mask = torch.cat(
                [attn, attn.new_ones((attn.size(0), 1))], dim=-1
            )
        return logits

    @abstractmethod
    def step_with_watermark(self) -> torch.Tensor:
        pass

    def set_next_token(self, token: torch.Tensor) -> bool:
        """
        Parameters
        ----------
        token : torch.Tensor
            Must have only one element.

        Returns
        -------
        bool
            True if generation should continue, False if it should stop.
        """
        token = token.to(self.model.device).reshape(1, 1)

        if self.output_ids is None:
            self.output_ids = token
        else:
            self.output_ids = torch.cat([self.output_ids, token], dim=1)
        self._num_tokens += 1

        is_eos = (token == self.model.config.eos_token_id).any().item()
        return not (is_eos or self._num_tokens >= self.max_new_tokens)

    def all_token_ids(self) -> torch.Tensor:
        """Return the full context seen by the model so far."""
        if self.output_ids is None:
            return self.input_ids
        return torch.cat([self.input_ids, self.output_ids], dim=1)


class WatermarkedLLM(ABC):
    def __init__(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
    ) -> None:
        if TYPE_CHECKING:
            self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device_map
            )
            self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(
                model_name, torch_dtype=torch_dtype
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device_map
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, torch_dtype=torch_dtype
            )

    def tokenize(self, prompt: str | list[str], gen_kwargs: dict) -> BatchEncoding:
        prompt_max_length = self._get_prompt_max_length(gen_kwargs)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=prompt_max_length,
        ).to(self.model.device)
        return inputs

    def decode(self, output: GenerateOutput) -> list[str]:
        return self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

    def generate(
        self, prompts: str | list[str] | BatchEncoding, **gen_kwargs
    ) -> GenerateOutput:
        if isinstance(prompts, (str, list)):
            inputs = self.tokenize(prompts, gen_kwargs)
        else:
            inputs = prompts
        return self.model.generate(**inputs, **gen_kwargs)  # type: ignore

    @abstractmethod
    def generate_with_watermark(
        self, prompt: str | BatchEncoding, *args, **kwargs
    ) -> GenerateOutput:
        pass

    @abstractmethod
    @contextmanager
    def generation_context(
        self, prompt: str | BatchEncoding, **kwargs
    ) -> Iterator[GenerationContext]:
        pass

    def _get_prompt_max_length(self, gen_kwargs: dict) -> int:
        prompt_max_length = gen_kwargs.pop("prompt_max_length", None)
        if prompt_max_length is not None:
            return prompt_max_length
        max_new_tokens = gen_kwargs.get("max_new_tokens", 0)
        if hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings - max_new_tokens
        return 2048 - max_new_tokens


@dataclass
class DetectionConfig:
    threshold: Optional[float] = None
    """
    Decision threshold for the primary statistic (e.g., z-score).
    None means caller handles decision.
    """
    alpha: Optional[float] = None
    """Significance level for detectors that support dynamic thresholds."""
    multiple_key: bool = False
    """Whether detection should consider multiple keys."""
    num_keys: int = 1
    """Number of keys when multiple_key is True."""
    context_width: int = 1
    """Context width used by the scheme (e.g., KGW seeding window)."""
    ignore_repeated_bigrams: bool = True
    """Scheme-specific option (KGW) to score unique bigrams only."""
    normalizers: Iterable[str] = ()
    """Optional text normalization strategies to apply before tokenization."""


@dataclass
class DetectionResult:
    passed: Optional[bool] = None
    """Whether the sequence is classified as watermarked under the configured threshold."""
    z_score: Optional[float] = None
    """Test statistic (when applicable)."""
    p_value: Optional[float] = None
    """P-value (when applicable)."""
    num_tokens_scored: Optional[int] = None
    """Number of tokens used in scoring (scheme-dependent)."""
    num_green_tokens: Optional[int] = None
    """Count of tokens considered as green (scheme-dependent)."""
    green_fraction: Optional[float] = None
    """Fraction of green tokens."""
    extras: dict[str, Any] = None  # type: ignore[assignment]
    """Scheme-specific additional fields."""


class WatermarkDetector(ABC):
    def __init__(
        self,
        *,
        tokenizer: Optional[LlamaTokenizer] = None,
        config: Optional[DetectionConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or DetectionConfig()
        self.device = device

    @abstractmethod
    def detect_tokens(self, input_ids: torch.Tensor) -> DetectionResult:
        raise NotImplementedError

    def detect(self, text: str) -> DetectionResult:
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for detect_text.")
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids[0]
        return self.detect_tokens(
            input_ids.to(self.device) if self.device else input_ids
        )

    def batch_detect_tokens(
        self, batch_input_ids: torch.Tensor
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for i in range(batch_input_ids.size(0)):
            results.append(self.detect_tokens(batch_input_ids[i]))
        return results

    def supports_dynamic_threshold(self) -> bool:
        return False

    def dynamic_threshold(self, input_ids: torch.Tensor) -> float:
        """Return a scheme-specific dynamic threshold (e.g., Unigram).
        Should be overridden when supports_dynamic_threshold() is True.
        """
        raise NotImplementedError

    def to(self, device: torch.device) -> "WatermarkDetector":
        self.device = device
        return self
