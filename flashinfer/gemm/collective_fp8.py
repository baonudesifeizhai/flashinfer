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

from typing import Literal, Optional

import torch

from ..api_logging import flashinfer_api
from ..comm.vllm_ar import all_gather as vllm_all_gather
from .gemm_base import bmm_fp8


@flashinfer_api
def fused_all_gather_bmm_fp8(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    custom_ar: int,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
    world_size: int,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal["cudnn", "cublas", "cutlass", "auto"] = "auto",
) -> torch.Tensor:
    r"""All-gather a local FP8 activation shard and run FlashInfer FP8 BMM.

    This is the first FlashInfer-side AG+GEMM primitive for TP-style execution.
    The implementation intentionally reuses the existing vLLM custom all-gather
    transport and the existing ``bmm_fp8`` kernel so the public API can
    stabilize before a lower-level native fused kernel lands.

    Parameters
    ----------
    A_shard : torch.Tensor
        Local activation shard of shape ``(m_local, k)``.
    B : torch.Tensor
        Weight tensor of shape ``(k, n)`` in the layout expected by
        ``bmm_fp8`` once unsqueezed to batched form.
    A_scale : torch.Tensor
        Per-tensor FP8 scale tensor for the activation.
    B_scale : torch.Tensor
        Per-tensor FP8 scale tensor for the weight.
    custom_ar : int
        Handle returned by ``flashinfer.comm.vllm_init_custom_ar``.
    reg_buffer : int
        IPC-registered staging buffer pointer for the local rank.
    reg_buffer_sz_bytes : int
        Size of ``reg_buffer`` in bytes.
    world_size : int
        Tensor-parallel world size used to size the gathered activation.
    out_dtype : Optional[torch.dtype]
        Output dtype passed to ``bmm_fp8``. Defaults to ``torch.bfloat16``.
    backend : {"cudnn", "cublas", "cutlass", "auto"}
        FlashInfer FP8 GEMM backend selection.

    Returns
    -------
    torch.Tensor
        Dense GEMM output of shape ``(m_local * world_size, n)``.
    """

    if A_shard.ndim != 2:
        raise ValueError(
            "fused_all_gather_bmm_fp8 expects a 2D A_shard, "
            f"got shape {tuple(A_shard.shape)}"
        )
    if B.ndim != 2:
        raise ValueError(
            f"fused_all_gather_bmm_fp8 expects a 2D B, got shape {tuple(B.shape)}"
        )
    if A_shard.shape[1] != B.shape[0]:
        raise ValueError(
            f"K dimension mismatch: A_shard has shape {tuple(A_shard.shape)}, "
            f"B has shape {tuple(B.shape)}"
        )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    out_dtype = out_dtype or torch.bfloat16

    gathered = torch.empty(
        (A_shard.shape[0] * world_size, A_shard.shape[1]),
        device=A_shard.device,
        dtype=A_shard.dtype,
    )
    vllm_all_gather(
        custom_ar,
        A_shard,
        gathered,
        reg_buffer,
        reg_buffer_sz_bytes,
    )

    return bmm_fp8(
        gathered.unsqueeze(0),
        B.unsqueeze(0),
        A_scale,
        B_scale,
        out_dtype,
        backend=backend,
    ).squeeze(0)
