"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from .gemm_base import bmm_fp8

try:
    import torch.distributed._functional_collectives as funcol
except ImportError:
    funcol = None


def _require_funcol() -> None:
    if funcol is None:
        raise RuntimeError(
            "torch.distributed._functional_collectives is required for "
            "fused_all_gather_bmm_fp8"
        )


@flashinfer_api
def fused_all_gather_bmm_fp8(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Prototype dense all-gather + FP8 BMM for TP-style eager execution.

    This is a two-stage prototype:

    1. Gather ``A_shard`` into a dense tensor ``A`` with functional collectives.
    2. Run FlashInfer ``bmm_fp8`` over the gathered tensor and the FP8 weight.

    The interface intentionally mirrors vLLM's
    ``fused_all_gather_flashinfer_scaled_matmul`` shape so the callsite can be
    swapped over with minimal changes while we validate correctness and a
    baseline performance profile.

    Parameters
    ----------
    A_shard : torch.Tensor
        Local shard of shape ``(m_local, k)`` or a rank-local dense shard along
        ``gather_dim``.
    B : torch.Tensor
        Weight tensor of shape ``(k, n)`` in column-major layout expected by
        ``bmm_fp8`` once unsqueezed to batched form.
    A_scale : torch.Tensor
        FP8 scale tensor for ``A_shard`` / gathered ``A``.
    B_scale : torch.Tensor
        FP8 scale tensor for ``B``.
    gather_dim : int
        Dimension used by functional all-gather.
    group_name : str
        Functional collective group name.
    out_dtype : Optional[torch.dtype]
        Output dtype passed to ``bmm_fp8``. Defaults to ``torch.bfloat16``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gathered dense tensor ``A`` and the matmul output.
    """

    _require_funcol()

    if A_shard.ndim != 2:
        raise ValueError(
            "fused_all_gather_bmm_fp8 expects a 2D A_shard, "
            f"got shape {tuple(A_shard.shape)}"
        )
    if B.ndim != 2:
        raise ValueError(
            f"fused_all_gather_bmm_fp8 expects a 2D B, got shape {tuple(B.shape)}"
        )

    out_dtype = out_dtype or torch.bfloat16
    B = B.clone(memory_format=torch.contiguous_format)

    gathered = funcol.all_gather_tensor(A_shard, gather_dim, group_name)
    gathered = funcol.wait_tensor(gathered)

    mm_out = bmm_fp8(
        gathered.unsqueeze(0),
        B.unsqueeze(0),
        A_scale,
        B_scale,
        out_dtype,
        backend="auto",
    ).squeeze(0)

    return gathered, mm_out


@flashinfer_api
def fused_bmm_fp8_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Prototype dense FP8 BMM + reduce-scatter for TP-style eager execution.

    This is a two-stage prototype:

    1. Run FlashInfer ``bmm_fp8`` on the local shard.
    2. Reduce-scatter the dense output with functional collectives.
    """

    _require_funcol()

    if A.ndim != 2:
        raise ValueError(
            f"fused_bmm_fp8_reduce_scatter expects a 2D A, got shape {tuple(A.shape)}"
        )
    if B.ndim != 2:
        raise ValueError(
            f"fused_bmm_fp8_reduce_scatter expects a 2D B, got shape {tuple(B.shape)}"
        )

    out_dtype = out_dtype or torch.bfloat16
    B = B.clone(memory_format=torch.contiguous_format)

    mm_out = bmm_fp8(
        A.unsqueeze(0),
        B.unsqueeze(0),
        A_scale,
        B_scale,
        out_dtype,
        backend="auto",
    ).squeeze(0)

    reduced = funcol.reduce_scatter_tensor(
        mm_out,
        reduce_op,
        scatter_dim,
        group_name,
    )
    reduced = funcol.wait_tensor(reduced)
    return reduced
