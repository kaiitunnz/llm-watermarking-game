import sys

import numpy as np
import pyximport
import torch

pyximport.install(
    reload_support=True,
    language_level=sys.version_info[0],
    setup_args={"include_dirs": np.get_include()},
)
from wmgame.watermark.exp.gumbel_levenshtein import gumbel_levenshtein  # type: ignore


def gumbel_key_func(
    generator: torch.Generator,
    n: int,
    vocab_size: int,
    eff_vocab_size: int | None = None,
):
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size
    pi = torch.arange(eff_vocab_size)
    xi = torch.rand((n, eff_vocab_size), generator=generator)
    return xi, pi


def gumbel_query(
    probs: torch.Tensor, pi: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    # probs: [B, V], pi: [B?, V] or [V], xi: [B?, V]
    pi = _expand_to_batch(pi, probs.size(0), probs.device)
    xi = _expand_to_batch(xi, probs.size(0), probs.device)

    gathered = torch.gather(probs, 1, pi).clamp_min(1e-12)
    new_probs = xi.clamp_min(1e-12) ** (1 / gathered)
    new_probs = new_probs / torch.sum(new_probs, dim=1, keepdim=True)
    return new_probs


def gumbel_sampling(
    probs: torch.Tensor,
    pi: torch.Tensor,
    xi: torch.Tensor,
) -> torch.Tensor:
    pi = _expand_to_batch(pi, probs.size(0), probs.device)
    xi = _expand_to_batch(xi, probs.size(0), probs.device)
    gathered = torch.gather(probs, 1, pi).clamp_min(1e-12)
    gumbel_scores = xi.clamp_min(1e-12) ** (1 / gathered)
    best_positions = torch.argmax(gumbel_scores, dim=1, keepdim=True)
    return torch.gather(pi, 1, best_positions)


def gumbel_edit_score(
    tokens: torch.Tensor, xi: torch.Tensor, gamma: float
) -> torch.Tensor:
    return gumbel_levenshtein(tokens.numpy(), xi.numpy(), gamma)


def _expand_to_batch(
    tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = tensor.to(device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.size(0) == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1)
    return tensor
