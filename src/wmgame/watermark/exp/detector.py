# Adapted from https://github.com/Qi-Pang/LLM-Watermark-Attacks/blob/main/Exp-Watermark/
from typing import Callable, Optional, Tuple, Any

import torch

from wmgame.watermark.base import (
    WatermarkDetector as BaseDetector,
    DetectionResult,
    DetectionConfig,
)
from wmgame.watermark.exp.gumbel import gumbel_edit_score, gumbel_key_func


def permutation_test(
    tokens: torch.Tensor,
    vocab_size: int,
    n: int,
    k: int,
    seed: int,
    test_stat: Callable[..., torch.Tensor],
    n_runs: int = 100,
    max_seed: int = 100000,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    test_result = test_stat(
        tokens=tokens, n=n, k=k, generator=generator, vocab_size=vocab_size
    )
    p_val: torch.Tensor = torch.tensor(0.0)
    for run in range(n_runs):
        pi = torch.randperm(vocab_size)
        tokens = torch.argsort(pi)[tokens]

        rand_seed = int(torch.randint(high=max_seed, size=(1,)).item())
        generator.manual_seed(rand_seed)
        null_result = test_stat(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            vocab_size=vocab_size,
            null=True,
        )
        # assuming lower test values indicate presence of watermark
        p_val += (null_result <= test_result).float() / n_runs

    return p_val


def fast_permutation_test(
    tokens: torch.Tensor,
    vocab_size: int,
    n: int,
    k: int,
    seed: int,
    test_stat: Callable[..., torch.Tensor],
    null_results: torch.Tensor,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    test_result = test_stat(
        tokens=tokens, n=n, k=k, generator=generator, vocab_size=vocab_size
    )
    p_val = torch.searchsorted(null_results, test_result, right=True) / len(
        null_results
    )
    return p_val


def fast_permutation_test_query(
    tokens: torch.Tensor,
    vocab_size: int,
    n: int,
    k: int,
    seed: int,
    test_stat: Callable[..., torch.Tensor],
    null_results_list: list[torch.Tensor],
    null_id: int,
    args: Optional[Any] = None,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    test_result = test_stat(
        tokens=tokens, n=n, k=k, generator=generator, vocab_size=vocab_size
    )
    # FIXME: add dp noise
    if args is not None and (
        args.action == "spoofing-defense" or args.action == "dp-benchmark"
    ):
        test_result = test_result + torch.normal(0, args.sigma, size=test_result.size())
    null_results = null_results_list[null_id]
    p_val = torch.searchsorted(null_results, test_result, right=True) / len(
        null_results
    )
    return p_val


def phi_query(
    tokens: torch.Tensor,
    n: int,
    k: int,
    generator: torch.Generator,
    key_func: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    dist: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    null: bool = False,
    normalize: bool = False,
    xi: Optional[torch.Tensor] = None,
    pi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute phi test statistic for query-based detection.
    
    Args:
        tokens: Input token tensor
        n: Number of keys in watermark
        k: Context length
        generator: Random generator
        key_func: Function to generate keys (xi, pi)
        vocab_size: Size of the vocabulary
        dist: Distance function
        null: Whether this is a null hypothesis test
        normalize: Whether to normalize tokens
        xi: Optional precomputed xi keys
        pi: Optional precomputed pi permutation
        
    Returns:
        Test statistic value
    """
    if null:
        tokens = torch.unique(tokens, return_inverse=True, sorted=False)[1]
        eff_vocab_size = torch.max(tokens).item() + 1
    else:
        eff_vocab_size = vocab_size
    if xi is None:
        xi, pi = key_func(generator, n, vocab_size, eff_vocab_size)
    assert pi is not None
    tokens = torch.argsort(pi)[tokens]
    if normalize:
        tokens = tokens.float() / vocab_size

    A = adjacency(tokens, xi, dist, k)
    closest = torch.min(A, dim=1)[0]

    return torch.min(closest)


def phi(
    tokens: torch.Tensor,
    n: int,
    k: int,
    generator: torch.Generator,
    key_func: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    dist: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    null: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute phi test statistic.
    
    Args:
        tokens: Input token tensor
        n: Number of keys in watermark
        k: Context length
        generator: Random generator
        key_func: Function to generate keys (xi, pi)
        vocab_size: Size of the vocabulary
        dist: Distance function
        null: Whether this is a null hypothesis test
        normalize: Whether to normalize tokens
        
    Returns:
        Test statistic value
    """
    if null:
        inv = torch.unique(tokens, return_inverse=True, sorted=False)[1]
        tokens = torch.unique(tokens, return_inverse=True, sorted=False)[1]
        eff_vocab_size = torch.max(tokens).item() + 1 if inv.numel() > 0 else max(vocab_size, 1)
    else:
        eff_vocab_size = vocab_size

    xi, pi = key_func(generator, n, vocab_size, eff_vocab_size)
    tokens = torch.argsort(pi)[tokens]
    if normalize:
        tokens = tokens.float() / vocab_size

    A = adjacency(tokens, xi, dist, k)
    closest = torch.min(A, dim=1)[0]

    if closest.numel() == 0:
        return torch.tensor(float("nan"))

    return torch.min(closest)


def adjacency(
    tokens: torch.Tensor,
    xi: torch.Tensor,
    dist: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    k: int,
) -> torch.Tensor:
    """Compute adjacency matrix for watermark detection.
    
    Args:
        tokens: Input token tensor
        xi: Key tensor
        dist: Distance function
        k: Context length
        
    Returns:
        Adjacency matrix
    """
    m = len(tokens)
    n = len(xi)

    A = torch.empty(size=(m - (k - 1), n))
    for i in range(m - (k - 1)):
        for j in range(n):
            A[i][j] = dist(tokens[i : i + k], xi[(j + torch.arange(k)) % n])

    return A


class ExpDetector(BaseDetector):
    """EXP (Gumbel-based) detector using injected permutation-test function.

    Expects caller to supply:
      - fast_test: Callable(tokens, vocab_size, n, k, seed, test_stat) -> torch.Tensor
      - test_stat: Callable(tokens, n, k, generator, vocab_size, null=False) -> torch.Tensor
      - n, k, seed, vocab_size
    """

    def __init__(
        self,
        *,
        tokenizer=None,
        device: Optional[torch.device] = None,
        config: Optional[DetectionConfig] = None,
        fast_test: Optional[Callable] = None,
        test_stat: Optional[Callable] = None,
        n: Optional[int] = None,
        k: Optional[int] = None,
        gamma: Optional[float] = None,
        seed: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, config=config, device=device)
        self.n = n
        self.k = k
        self.seed = seed
        self.vocab_size = vocab_size

        if fast_test is None:
            self.fast_test = permutation_test
        else:
            self.fast_test = fast_test

        if test_stat is None:
            assert gamma is not None
            dist = lambda x, y: gumbel_edit_score(x, y, gamma)
            self.test_stat = (
                lambda tokens, n, k, generator, vocab_size, null=False: phi(
                    tokens=tokens,
                    n=n,
                    k=k,
                    generator=generator,
                    key_func=gumbel_key_func,
                    vocab_size=vocab_size,
                    dist=dist,
                    null=null,
                    normalize=False,
                )
            )
        else:
            self.test_stat = test_stat

    def _require(self) -> None:
        if self.fast_test is None or self.test_stat is None:
            raise ValueError("ExpDetector requires fast_test and test_stat callables")
        if (
            self.n is None
            or self.k is None
            or self.seed is None
            or self.vocab_size is None
        ):
            raise ValueError("ExpDetector requires n, k, seed, and vocab_size")

    def detect_tokens(self, input_ids: torch.Tensor) -> DetectionResult:
        self._require()
        pval = self.fast_test(
            input_ids,
            int(self.vocab_size),  # type: ignore
            int(self.n),  # type: ignore
            int(self.k),  # type: ignore
            int(self.seed),  # type: ignore
            self.test_stat,
        )
        passed = None
        if self.config.threshold is not None:
            passed = float(pval) <= float(self.config.threshold)
        return DetectionResult(
            passed=passed,
            z_score=None,
            p_value=float(pval),
            num_tokens_scored=int(input_ids.numel()),
            num_green_tokens=None,
            green_fraction=None,
            extras={},
        )
