"""Mixed-radix butterfly layer: a drop-in for ``nn.Linear`` on the last dimension.

``MixedRadixButterfly(in_size, out_size, radices)`` generalises
``torch_structured.Butterfly`` (``increasing_stride=True``, ``nblocks=1``) from
2x2 blocks to a list of block sizes, one per stage.

Shapes follow the library. n is the product of ``radices`` and ``in_size <= n``.
The input is zero-padded at the end to n. There are ceil(out_size / n)
independent stacks, all applied to the same padded input; their outputs are
concatenated stack after stack, truncated to ``out_size``, and the bias is added.

Stages run in the order of ``radices``. Stage j (j = 1, 2, ...) has block size
b_j and stride s_j = b_1 ... b_(j-1) (s_1 = 1). It splits the n positions into
the groups {g b_j s_j + q s_j + l : q = 0 .. b_j - 1} and applies one b_j x b_j
matrix per group::

    out[g, p, l] = sum_q T_j[g s_j + l, p, q] * in[g, q, l]

So ``[2, 4, 4, 4, 4]`` mixes adjacent pairs first and ends with a 4x4 stage of
stride 128, and ``[4, 4, 4, 4, 2]`` starts with 4x4 blocks of stride 1 and ends
with a 2x2 stage of stride 256. With ``[2] * 9`` the layer is the library's
radix-2 ``Butterfly`` (stage j is ``Butterfly.twiddle[:, 0, j - 1]``); with
``[4, 4, 4, 4, 2]`` it is ``ButterflyBase4`` (stage j <= 4 is
``twiddle4[:, 0, j - 1]``, stage 5 is ``twiddle2[:, 0, 0]``).

Parameters: ``stages.<j-1>`` of shape (nstacks, n / b_j, b_j, b_j), block index
g s_j + l, and ``bias``.

Initialisation. The layer draws one integer from the global (CPU) generator,
adds a constant that depends on ``radices`` only (the radices read as the digits
of a decimal number), seeds a private ``torch.Generator`` with the sum and takes
everything else from it: a standard Gaussian tensor of the shape of each stage,
in stage order, then the bias (uniform on +-1/sqrt(in_size), as the library).
The global stream therefore advances by the same amount whatever the layout and
the initialisation, the initialisations of one layout are built from the same
Gaussian blocks, and different layouts share no coefficient. From the Gaussian
block G (b x b) of a stage:

* ``randn``: G / sqrt(b), i.e. entries N(0, 1/b) (for b = 2 the law of the
  library's ``randn``);
* ``rownorm``: every row of G divided by its Euclidean norm;
* ``ortho``: the factor Q of the QR decomposition of G, each column multiplied
  by the sign of the matching diagonal entry of R: a Haar-distributed orthogonal
  matrix (for b = 2 the law of the library's ``ortho``).

``ButterflyBase4(init="randn")`` is not this ``randn``: it multiplies two
Gaussian 2x2 stages. The forward pass uses plain PyTorch operations (the
reshape, multiply and sum of ``butterfly_multiply_base4_torch``), on CPU and
CUDA alike; it does not use the Triton or C++ kernels.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

INITS = ("randn", "ortho", "rownorm")


def radix_strides(radices: Sequence[int]) -> list[int]:
    """Stride of every stage: s_1 = 1, s_j = b_1 ... b_(j-1)."""
    out, s = [], 1
    for b in radices:
        out.append(s)
        s *= b
    return out


def radix_seed_offset(radices: Sequence[int]) -> int:
    """The constant added to the global draw: the radices read as the digits of a decimal number."""
    return int("".join(str(int(b)) for b in radices))


def init_blocks(gauss: torch.Tensor, init: str) -> torch.Tensor:
    """Blocks of one stage from its standard Gaussian blocks ``gauss`` (..., b, b).

    Computed in float64 and returned in the dtype of ``gauss``.
    """
    b = gauss.shape[-1]
    g = gauss.to(torch.float64)
    if init == "randn":
        out = g / math.sqrt(b)
    elif init == "rownorm":
        out = g / torch.linalg.vector_norm(g, dim=-1, keepdim=True)
    elif init == "ortho":
        q, r = torch.linalg.qr(g)
        d = torch.diagonal(r, dim1=-2, dim2=-1)
        sign = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
        out = q * sign.unsqueeze(-2)
    else:
        raise ValueError(f"unknown init {init!r} (expected one of {INITS})")
    return out.to(gauss.dtype)


class MixedRadixButterfly(nn.Module):
    """Product of stages with blocks of the sizes in ``radices`` (see the module docstring)."""

    def __init__(self, in_size: int, out_size: int, radices: Sequence[int], bias: bool = True,
                 init: str = "randn"):
        super().__init__()
        radices = [int(b) for b in radices]
        if not radices or any(b < 2 for b in radices):
            raise ValueError(f"radices must be a non-empty list of integers >= 2, got {radices}")
        if init not in INITS:
            raise ValueError(f"unknown init {init!r} (expected one of {INITS})")
        n = math.prod(radices)
        if not 1 <= in_size <= n:
            raise ValueError(f"in_size={in_size} must be between 1 and the product of the radices ({n})")
        if out_size < 1:
            raise ValueError(f"out_size must be positive, got {out_size}")
        self.in_size = in_size
        self.out_size = out_size
        self.radices = radices
        self.init = init
        self.n = n
        self.nstacks = -(-out_size // n)
        dtype = torch.get_default_dtype()
        self.stages = nn.ParameterList(
            [nn.Parameter(torch.empty(self.nstacks, n // b, b, b, dtype=dtype)) for b in radices])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        draw = int(torch.randint(0, 2 ** 62, (1,), dtype=torch.int64).item())  # the only global draw
        gen = torch.Generator()
        gen.manual_seed((draw + radix_seed_offset(self.radices)) % 2 ** 64)
        with torch.no_grad():
            for t in self.stages:
                gauss = torch.randn(tuple(t.shape), generator=gen, dtype=t.dtype)
                t.copy_(init_blocks(gauss, self.init))
            if self.bias is not None:
                bound = 1 / math.sqrt(self.in_size)
                b = torch.empty(self.out_size, dtype=self.bias.dtype).uniform_(-bound, bound, generator=gen)
                self.bias.copy_(b)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lead = input.shape[:-1]
        x = input.reshape(-1, input.shape[-1])
        batch, k, n = x.shape[0], self.nstacks, self.n
        y = F.pad(x.unsqueeze(1).expand(batch, k, x.shape[-1]), (0, n - x.shape[-1])).contiguous()
        for t, b, s in zip(self.stages, self.radices, radix_strides(self.radices)):
            g = n // (b * s)
            tw = t.view(k, g, s, b, b).permute(0, 1, 3, 4, 2)   # [stack, g, p, q, l]
            yr = y.view(batch, k, g, 1, b, s)                   # [batch, stack, g, 1, q, l]
            y = (tw * yr).sum(dim=4).reshape(batch, k, n)      # [batch, stack, g, p, l]
        out = y.reshape(batch, k * n)[:, :self.out_size]
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*lead, self.out_size)

    def extra_repr(self) -> str:
        return (f"in_size={self.in_size}, out_size={self.out_size}, radices={self.radices}, "
                f"bias={self.bias is not None}, init={self.init}")
