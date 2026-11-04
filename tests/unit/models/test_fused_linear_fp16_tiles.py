"""fp16 logit tiles (``logits_dtype="float16"``): bf16 operands cast to fp16.

The cast is exact for bf16 values of magnitude in [2**-14, 65504], so the
tile equals an fp16 GEMM of the same values; the gain is the fp16 output
rounding (2**-11 relative, vs 2**-8 for bf16) at the same bytes.
"""

import gc

import pytest
import torch

from speculators.losses import fused_linear as fl


def test_bf16_to_fp16_cast_is_exact_in_range():
    x = torch.randn(4096) * 30
    x = x[x.abs() >= 2**-14].to(torch.bfloat16)
    assert torch.equal(x.to(torch.float16).float(), x.float())


@pytest.mark.parametrize("shared", [False, True])
def test_project_fp16_tile(shared):
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(8, 32, generator=gen).to(torch.bfloat16)
    w = (torch.randn(50, 32, generator=gen) * 0.2).to(torch.bfloat16)
    tile_w, tile_tw = fl._tile_weights(w, None if shared else w, "float16", shared)
    assert tile_w.dtype == torch.float16
    assert tile_tw.dtype == torch.float16
    if shared:
        assert tile_tw is tile_w
    z = fl._project(h, tile_w, "float16")
    assert z.dtype == torch.float16
    ref = (h.double() @ w.double().T).to(torch.float16)
    # fp16 output of an fp32-accumulated GEMM: within one fp16 ulp
    torch.testing.assert_close(z.float(), ref.float(), rtol=2**-10, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_fp16_tile_never_downgrades(dtype):
    h = torch.randn(4, 8, dtype=dtype)
    w = torch.randn(10, 8, dtype=dtype)
    tile_w, _ = fl._tile_weights(w, None, "float16", False)
    assert tile_w is w
    assert fl._project(h, w, "float16").dtype == dtype


def test_shared_head_fp16_delta_rounding():
    """The fp16 delta tile rounds the exact fp32 difference once."""
    gen = torch.Generator().manual_seed(1)
    v = torch.randn(6, 16, generator=gen).to(torch.bfloat16)
    h = (v.float() + 0.01 * torch.randn(6, 16, generator=gen)).to(torch.bfloat16)
    w = (torch.randn(20, 16, generator=gen) * 0.3).to(torch.bfloat16)
    tile_w, tile_tw = fl._tile_weights(w, w, "float16", True)
    z, t = fl._logit_tiles(h, tile_w, v, tile_tw, 0, 6, "float16", True, True)
    # CPU materializes target + delta for the torch reference
    diff16 = (h.float() - v.float()).to(torch.float16)
    delta = diff16 @ tile_w.T
    target = v.to(torch.float16) @ tile_w.T
    torch.testing.assert_close(t, target)
    torch.testing.assert_close(z, delta.float() + target.float())


def test_logits_dtype_validation():
    h = torch.randn(4, 8)
    with pytest.raises(ValueError, match="logits_dtype"):
        fl.fused_linear_compound(
            h,
            torch.randn(10, 8),
            torch.randn(4, 10),
            ["kl_div"],
            logits_dtype="bfloat16",
        )


def test_fp16_weight_cache_hit_invalidate_evict():
    fl.clear_weight_cache()
    w = torch.randn(10, 4).to(torch.bfloat16)
    first = fl._fp16_copy(w)
    assert fl._fp16_copy(w) is first  # cached
    w.mul_(2)  # in-place write bumps the version: recast
    second = fl._fp16_copy(w)
    assert second is not first
    torch.testing.assert_close(second.float(), w.float())
    other = w.clone()  # equal values, different tensor: own entry
    assert fl._fp16_copy(other) is not second
    assert len(fl._WEIGHT_COPIES) == 2
    del w, other
    gc.collect()
    assert not fl._WEIGHT_COPIES  # entries die with their sources


def test_fp16_weight_cache_never_serves_a_dead_tensor():
    """A new tensor reusing a freed id never gets the old copy."""
    fl.clear_weight_cache()
    for i in range(20):
        w = torch.full((6, 3), float(i)).to(torch.bfloat16)
        assert torch.equal(fl._fp16_copy(w).float(), w.float())
        del w
    fl.clear_weight_cache()
