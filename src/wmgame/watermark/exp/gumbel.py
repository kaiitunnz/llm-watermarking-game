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


def gumbel_key_func(generator: torch.Generator, n: int, vocab_size: int, eff_vocab_size: int | None = None):
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size
    pi = torch.arange(eff_vocab_size)
    xi = torch.rand((n, eff_vocab_size), generator=generator)
    return xi, pi


def gumbel_query(
    probs: torch.Tensor, pi: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    # probs: [B, V], pi: [B?, V] or [V], xi: [B?, V]
    pi = pi.to(probs.device)
    xi = xi.to(probs.device)
    
    gathered = torch.gather(probs, 1, pi.expand_as(probs))
    new_probs = xi ** (1 / gathered)
    new_probs = new_probs / torch.sum(new_probs, dim=1, keepdim=True)
    return new_probs


def gumbel_edit_score(tokens, xi, gamma):
    return gumbel_levenshtein(tokens.numpy(), xi.numpy(), gamma)
