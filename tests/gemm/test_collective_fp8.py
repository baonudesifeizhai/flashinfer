import torch

import flashinfer.gemm.collective_fp8 as collective_fp8


def test_fused_all_gather_bmm_fp8_smoke(monkeypatch):
    call_log = {}

    def fake_all_gather(custom_ar, a_shard, gathered, reg_buffer, reg_buffer_sz_bytes):
        call_log["all_gather"] = {
            "custom_ar": custom_ar,
            "a_shape": tuple(a_shard.shape),
            "gathered_shape": tuple(gathered.shape),
            "reg_buffer": reg_buffer,
            "reg_buffer_sz_bytes": reg_buffer_sz_bytes,
        }
        gathered.copy_(torch.cat([a_shard, a_shard], dim=0))

    def fake_bmm_fp8(a, b, a_scale, b_scale, out_dtype, backend="auto"):
        call_log["bmm_fp8"] = {
            "a_shape": tuple(a.shape),
            "b_shape": tuple(b.shape),
            "a_scale_shape": tuple(a_scale.shape),
            "b_scale_shape": tuple(b_scale.shape),
            "out_dtype": out_dtype,
            "backend": backend,
        }
        return torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=out_dtype)

    monkeypatch.setattr(collective_fp8, "vllm_all_gather", fake_all_gather)
    monkeypatch.setattr(collective_fp8, "bmm_fp8", fake_bmm_fp8)

    a_shard = torch.randn(4, 8)
    b = torch.randn(8, 16)
    a_scale = torch.ones(1)
    b_scale = torch.ones(1)

    out = collective_fp8.fused_all_gather_bmm_fp8(
        a_shard,
        b,
        a_scale,
        b_scale,
        custom_ar=17,
        reg_buffer=23,
        reg_buffer_sz_bytes=4096,
        world_size=2,
        out_dtype=torch.bfloat16,
    )

    assert out.shape == (8, 16)
    assert call_log["all_gather"] == {
        "custom_ar": 17,
        "a_shape": (4, 8),
        "gathered_shape": (8, 8),
        "reg_buffer": 23,
        "reg_buffer_sz_bytes": 4096,
    }
    assert call_log["bmm_fp8"] == {
        "a_shape": (1, 8, 8),
        "b_shape": (1, 8, 16),
        "a_scale_shape": (1,),
        "b_scale_shape": (1,),
        "out_dtype": torch.bfloat16,
        "backend": "auto",
    }
