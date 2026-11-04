"""Shared-head mode of fused_linear: draft logits as ``target + delta``.

When the drafter reuses the verifier's ``lm_head`` (DFlash), the draft logits
are ``v W^T + (h - v) W^T``. The kernels take the delta tile and the target
tile and add them in fp32 (``DELTA``). The math is the unshared path's
exactly (fp64 equal); on GPU the point is precision: a bf16 delta GEMM plus
an fp32 target GEMM approaches fp32-logit gradients.
"""

import pytest
import torch

from speculators.losses import fused
from speculators.losses import fused_linear as fl
from tests.unit.models.test_fused_linear import _run
from tests.unit.models.test_fused_linear_deferred import (
    IDS,
    NAME_SETS,
    _kernel_reference,
)
from tests.unit.models.test_fused_losses import DEVICE, requires_accelerator

GRAD_MODES = ("deferred", "recompute")


def _shared_inputs(rows, hidden, vocab, dtype, device, seed=0, noise=0.3):
    """Shared head ``w``, verifier hidden ``v`` and a draft hidden near it."""
    gen = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(rows, hidden, generator=gen, device=device)
    w = torch.randn(vocab, hidden, generator=gen, device=device) * (2.0 / hidden**0.5)
    h = v + noise * torch.randn(rows, hidden, generator=gen, device=device)
    return h.to(dtype), w.to(dtype), v.to(dtype)


def _compound(h, w, v, names, shared, **kwargs):
    return lambda x: fl.fused_linear_compound(
        x,
        w,
        v,
        names,
        metric_terms=("tv",),
        target_weight=w,
        shared_head=shared,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@requires_accelerator
@pytest.mark.parametrize("vocab", [1000, 151936])
@pytest.mark.parametrize(
    "names",
    [("kl_div",), ("ce",), ("kl_div", "tv"), ("kl_div", "rkl", "jsd", "ce", "tv")],
    ids="+".join,
)
def test_delta_kernels_match_materialized_sum(vocab, names):
    """DELTA kernels on (delta bf16, target fp32) equal the plain kernels on
    the fp32 sum: the kernels add the two tiles in fp32 before anything else."""
    torch.manual_seed(0)
    rows = 64
    target = torch.randn(rows, vocab, device=DEVICE) * 3
    delta = (torch.randn(rows, vocab, device=DEVICE) * 0.5).to(torch.bfloat16)
    full = delta.float() + target
    ref_losses, ref_grads = _kernel_reference(full, target, names)
    losses, grads = _kernel_reference(delta, target, names, delta=True)
    for name in names:
        s = fused.TERM_SLOTS[name]
        torch.testing.assert_close(losses[s], ref_losses[s], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            grads[name].float(), ref_grads[name].float(), rtol=1e-2, atol=1e-2
        )

    mask = fused.terms_mask(names)
    slot = {n: i for i, n in enumerate(names)}
    out = {}
    for logits, delta_mode in ((full, False), (delta, True)):
        lo = torch.zeros(5, rows, device=DEVICE)
        am = torch.empty(2, rows, device=DEVICE, dtype=torch.int64)
        dz = torch.empty(len(names), rows, vocab, device=DEVICE)
        fused.loss_grad_kernel[(rows,)](
            logits,
            target,
            lo,
            rows,
            am,
            rows,
            dz,
            rows * vocab,
            vocab,
            TERMS=mask,
            GRAD_TERMS=mask,
            DZ_KL=slot.get("kl_div", 0),
            DZ_RKL=slot.get("rkl", 0),
            DZ_JSD=slot.get("jsd", 0),
            DZ_CE=slot.get("ce", 0),
            DZ_TV=slot.get("tv", 0),
            DELTA=delta_mode,
        )
        out[delta_mode] = (lo, am, dz)
    assert torch.equal(out[True][1], out[False][1])
    torch.testing.assert_close(out[True][0], out[False][0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out[True][2], out[False][2], rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# CPU (torch reference path): exact equivalence and gradcheck
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grad_mode", GRAD_MODES)
@pytest.mark.parametrize("names", NAME_SETS, ids=IDS)
def test_shared_head_equals_unshared_fp64(names, grad_mode):
    h, w, v = _shared_inputs(70, 16, 300, torch.float64, "cpu", seed=1)
    kwargs = {"grad_mode": grad_mode, "chunk_rows": 32}
    out, grad = _run(_compound(h, w, v, names, True, **kwargs), h, names)
    ref, ref_grad = _run(_compound(h, w, v, names, False, **kwargs), h, names)
    for name in names:
        torch.testing.assert_close(out.losses[name], ref.losses[name])
    torch.testing.assert_close(out.metrics["tv"], ref.metrics["tv"])
    assert torch.equal(out.pred_ids, ref.pred_ids)
    assert torch.equal(out.target_ids, ref.target_ids)
    torch.testing.assert_close(grad, ref_grad)


@pytest.mark.parametrize("grad_mode", GRAD_MODES)
@pytest.mark.parametrize("name", ["kl_div", "ce", "tv", "jsd"])
def test_shared_head_gradcheck(name, grad_mode):
    h, w, v = _shared_inputs(12, 6, 40, torch.float64, "cpu", seed=2)
    weights = torch.linspace(0.5, 1.5, 12, dtype=torch.float64)

    def loss(x):
        out = fl.fused_linear_compound(
            x,
            w,
            v,
            [name],
            target_weight=w,
            shared_head=True,
            grad_mode=grad_mode,
            chunk_rows=5,
        )
        return (out.losses[name] * weights).sum()

    assert torch.autograd.gradcheck(loss, (h.requires_grad_(True),))


def test_shared_head_skipped_rows_fp64():
    h, w, v = _shared_inputs(40, 8, 100, torch.float64, "cpu", seed=3)
    rows = torch.arange(0, 40, 3)
    names = ("kl_div", "ce")
    kwargs = {"rows": rows, "chunk_rows": 8}
    out, grad = _run(_compound(h, w, v, names, True, **kwargs), h, names)
    ref, ref_grad = _run(_compound(h, w, v, names, False, **kwargs), h, names)
    for name in names:
        torch.testing.assert_close(out.losses[name], ref.losses[name])
    torch.testing.assert_close(grad, ref_grad)


def test_shared_head_validation():
    h, w, v = _shared_inputs(8, 4, 20, torch.float64, "cpu")
    with pytest.raises(ValueError, match="shared_head"):
        fl.fused_linear_compound(h, w, v, ["kl_div"], shared_head=True)
    with pytest.raises(ValueError, match="shared_head"):
        fl.fused_linear_compound(
            h, w, v, ["kl_div"], target_weight=w[:10], shared_head=True
        )


def test_chunk_rows_shared_head_budget():
    # bf16 delta row + fp32 target row + one bf16 dz row = 8 bytes per vocab
    # entry: half the plain fp32-tiles budget would allow 2x the rows.
    vocab = 1 << 17
    shared = fl.chunk_rows_for(
        vocab, torch.bfloat16, 1 << 20, fp32_logits=True, shared_head=True
    )
    both_fp32 = fl.chunk_rows_for(
        vocab, torch.bfloat16, 1 << 20, recompute_targets=True, fp32_logits=True
    )
    assert shared == 2048
    assert both_fp32 == 2048
    tight = {"SPECULATORS_FUSED_LINEAR_BUDGET_MIB": "1024"}
    with pytest.MonkeyPatch.context() as mp:
        for key, value in tight.items():
            mp.setenv(key, value)
        assert (
            fl.chunk_rows_for(
                vocab, torch.bfloat16, 1 << 20, fp32_logits=True, shared_head=True
            )
            == 1 << 10
        )
        assert (
            fl.chunk_rows_for(
                vocab, torch.bfloat16, 1 << 20, recompute_targets=True, fp32_logits=True
            )
            == 1 << 9
        )


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@requires_accelerator
@pytest.mark.parametrize("grad_mode", GRAD_MODES)
@pytest.mark.parametrize("names", NAME_SETS, ids=IDS)
def test_shared_head_matches_unshared_fp32_gpu(names, grad_mode):
    h, w, v = _shared_inputs(300, 64, 3000, torch.float32, DEVICE, seed=4)
    kwargs = {"grad_mode": grad_mode, "chunk_rows": 128}
    out, grad = _run(_compound(h, w, v, names, True, **kwargs), h, names)
    ref, ref_grad = _run(_compound(h, w, v, names, False, **kwargs), h, names)
    for name in names:
        torch.testing.assert_close(
            out.losses[name], ref.losses[name], rtol=1e-4, atol=1e-4
        )
    torch.testing.assert_close(
        grad, ref_grad, rtol=1e-3, atol=1e-4 * float(ref_grad.abs().max())
    )


def _rel_err(x, ref):
    return float((x.double() - ref).norm() / ref.norm())


@requires_accelerator
@pytest.mark.parametrize("names", [("kl_div",), ("tv",), ("ce",)], ids="+".join)
def test_shared_head_bf16_precision(names):
    """bf16 inputs: fp32 tiles, fp16 tiles and the shared head (fp32 / fp16
    target) all land near the fp64 gradient -- at this size near the floor
    set by the bf16 d_hidden itself -- and far closer than bf16 tiles. Where
    shared head beats plain fp16 tiles (large V, alpha -> 1) is measured by
    the alpha sweep (design doc 5.11), too slow for a unit test."""
    h, w, v = _shared_inputs(128, 512, 32000, torch.float32, DEVICE, seed=5)
    # A drafter that agrees with its verifier: logits scaled up so the
    # softmaxes are peaked (alpha ~0.9 regime) and bf16 rounding matters.
    w = w * 6
    h16, w16, v16 = (t.to(torch.bfloat16) for t in (h, w, v))
    ref_out, ref_grad = _run(
        _compound(
            h16.double().cpu(),
            w16.double().cpu(),
            v16.double().cpu(),
            names,
            False,
        ),
        h16.double().cpu(),
        names,
    )
    errs = {}
    for label, shared, dtype in (
        ("bf16", False, "input"),
        ("shared_fp32", True, "float32"),
        ("fp16", False, "float16"),
        ("shared_fp16", True, "float16"),
        ("fp32", False, "float32"),
    ):
        out, grad = _run(
            _compound(h16, w16, v16, names, shared, logits_dtype=dtype), h16, names
        )
        errs[label] = _rel_err(grad.cpu(), ref_grad)
        for name in names:
            if name != "ce" and label not in ("bf16", "fp16"):
                torch.testing.assert_close(
                    out.losses[name].cpu().double(),
                    ref_out.losses[name],
                    rtol=5e-2,
                    atol=5e-3,
                )
    assert errs["shared_fp32"] < 0.05, errs
    assert errs["shared_fp32"] < 0.5 * errs["bf16"], errs
    assert errs["fp16"] < 0.5 * errs["bf16"], errs
    assert errs["shared_fp16"] < 0.05, errs
    assert errs["shared_fp16"] < 0.5 * errs["bf16"], errs


@requires_accelerator
def test_shared_head_compiles():
    h, w, v = _shared_inputs(200, 64, 3000, torch.bfloat16, DEVICE, seed=6)
    names = ("kl_div", "tv")

    def step(x):
        out = fl.fused_linear_compound(
            x,
            w,
            v,
            names,
            target_weight=w,
            shared_head=True,
            logits_dtype="float32",
        )
        return out.losses["kl_div"].sum() + out.losses["tv"].sum()

    eager_x = h.clone().requires_grad_(True)
    step(eager_x).backward()
    compiled_x = h.clone().requires_grad_(True)
    torch.compile(step, fullgraph=True)(compiled_x).backward()
    torch.testing.assert_close(compiled_x.grad, eager_x.grad)


@requires_accelerator
def test_shared_head_autocast():
    """Under autocast fp32 parameters are cast once, for both projections."""
    h, w, v = _shared_inputs(128, 64, 2000, torch.float32, DEVICE, seed=7)
    names = ("kl_div",)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, grad = _run(_compound(h, w, v, names, True), h, names)
    ref, ref_grad = _run(
        _compound(
            h.to(torch.bfloat16),
            w.to(torch.bfloat16),
            v.to(torch.bfloat16),
            names,
            True,
        ),
        h.to(torch.bfloat16),
        names,
    )
    torch.testing.assert_close(out.losses["kl_div"], ref.losses["kl_div"])
    torch.testing.assert_close(grad, ref_grad.float(), rtol=1e-2, atol=1e-3)
