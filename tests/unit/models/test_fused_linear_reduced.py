"""fused_linear_reduced: the scalar compound loss with gradients in forward.

The reference is autograd through materialized logits ``h W^T + u W2^T``
with the eager losses and ``compound_loss_with_outputs`` (fixed decay or
D-PACE): the loss and the gradients of ``h``, ``u`` and the trainable ``W2``
must agree exactly in fp64 and to rounding on GPU.
"""

from functools import partial

import pytest
import torch

from speculators.losses import (
    compound_loss_with_outputs,
    fused,
    resolve_loss_config,
)
from speculators.losses import fused_linear as fl
from speculators.losses import fused_linear_reduced as flr
from speculators.losses.utils import dflash_loss_decay, dpace_loss_decay
from tests.unit.models.test_fused_losses import DEVICE, requires_accelerator

SPECS = [
    "kl_div",
    "ce",
    "tv",
    "rkl",
    "jsd",
    "nla",
    "lk_hybrid",
    '{"kl_div": 0.5, "tv": 0.5}',
    '{"ce": 0.1, "tv": 0.9}',
]
BLOCK = 4


def _inputs(
    rows, hidden, vocab, rank, dtype, device, seed=0, shared=False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device=device) * scale

    v = randn(rows, hidden)
    w = randn(vocab, hidden, scale=2.0 / hidden**0.5)
    tw = w if shared else w + randn(vocab, hidden, scale=0.5 / hidden**0.5)
    h = v + randn(rows, hidden, scale=0.4)
    u = randn(rows, rank)
    w2 = randn(vocab, rank, scale=0.3 / rank**0.5)
    mask = (torch.rand(rows, generator=gen, device=device) < 0.85).float()
    mask.view(-1, BLOCK)[:, 0] = 1.0
    h, w, v, tw, u, w2 = (t.to(dtype) for t in (h, w, v, tw, u, w2))
    return h, w, v, tw, u, w2, mask


def _names_weights(spec):
    config = resolve_loss_config(spec, "fused")
    names = list(config)
    return names, [w for _, w in config.values()]


def _decay(rows, device):
    pos = (torch.arange(rows, device=device) % BLOCK)[None]
    return pos, dflash_loss_decay(pos.double(), gamma=4.0, sample_from_anchor=True)


def _reference(h, w, v, tw, u, w2, mask, spec, dpace, targets_mode):
    """Loss and grads (h, u, w2) through materialized logits, eager losses."""
    leaves = [t.detach().clone().requires_grad_(True) for t in (h, u, w2)]
    hh, uu, ww2 = leaves
    logits = hh @ w.T + (uu @ ww2.T if u is not None else 0)
    targets = v @ tw.T
    pos, decay = _decay(h.shape[0], h.device)
    decay_fn = (
        partial(
            dpace_loss_decay, loss_mask=mask[None], block_size=BLOCK, dpace_alpha=0.5
        )
        if dpace
        else (lambda _pos, **_: decay.to(logits.dtype))
    )
    out = compound_loss_with_outputs(
        logits[None],
        targets[None],
        mask[None],
        pos,
        resolve_loss_config(spec, "eager"),
        decay_fn,
    )
    out.loss.backward()
    return out.loss.detach(), [t.grad for t in leaves]


def _run(h, w, v, tw, u, w2, mask, spec, dpace, targets_mode, **kwargs):
    names, weights = _names_weights(spec)
    leaves = [t.detach().clone().requires_grad_(True) for t in (h, u, w2)]
    hh, uu, ww2 = leaves
    _, decay = _decay(h.shape[0], h.device)
    loss, out = flr.fused_linear_reduced(
        hh[None],
        w,
        (v @ tw.T)[None] if targets_mode == "logits" else v[None],
        names,
        weights,
        mask[None],
        decay=decay,
        dpace_alpha=0.5 if dpace else None,
        block_size=BLOCK,
        extra=uu[None],
        extra_weight=ww2,
        target_weight=None if targets_mode == "logits" else tw,
        shared_head=targets_mode == "shared",
        metric_terms=("tv",),
        **kwargs,
    )
    loss.backward()
    return loss.detach(), [t.grad for t in leaves], out


# ---------------------------------------------------------------------------
# Reduction math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", flr.REDUCED_NAMES)
def test_composed_derivatives_match_compose_loss(name):
    gen = torch.Generator().manual_seed(0)
    terms = {
        t: torch.rand(64, generator=gen, dtype=torch.float64).requires_grad_(True)
        for t in fused.TERM_SLOTS
    }
    terms["tv"].data[:4] = torch.tensor([0.0, 1.0, 1.0 - 1e-6, 0.999])  # nla clamp
    el, dterms = flr._composed(name, {t: x.detach() for t, x in terms.items()})
    ref = fused.compose_loss(name, terms)
    torch.testing.assert_close(el, ref.detach())
    ref.sum().backward()
    for t, x in terms.items():
        expected = x.grad if x.grad is not None else torch.zeros_like(x)
        torch.testing.assert_close(dterms.get(t, torch.zeros_like(x)), expected)


# ---------------------------------------------------------------------------
# CPU fp64: exact against the eager reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("targets_mode", ["logits", "verifier", "shared"])
@pytest.mark.parametrize("spec", SPECS)
def test_matches_eager_fp64(spec, targets_mode):
    shared = targets_mode == "shared"
    args = _inputs(48, 8, 60, 5, torch.float64, "cpu", seed=1, shared=shared)
    ref_loss, ref_grads = _reference(*args, spec, False, targets_mode)
    loss, grads, _ = _run(*args, spec, False, targets_mode, chunk_rows=20)
    torch.testing.assert_close(loss, ref_loss)
    for got, ref in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("targets_mode", ["logits", "shared"])
def test_dpace_matches_eager_fp64(targets_mode):
    args = _inputs(48, 8, 60, 5, torch.float64, "cpu", seed=2, shared=True)
    ref_loss, ref_grads = _reference(*args, "ce", True, targets_mode)
    # 22-row chunks are rounded down to whole blocks for D-PACE
    loss, grads, _ = _run(*args, "ce", True, targets_mode, chunk_rows=22)
    torch.testing.assert_close(loss, ref_loss)
    for got, ref in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(got, ref)


def test_outputs_and_no_extra_fp64():
    h, w, v, tw, u, w2, mask = _inputs(32, 6, 40, 3, torch.float64, "cpu", seed=3)
    names, weights = _names_weights('{"kl_div": 0.5, "tv": 0.5}')
    _, decay = _decay(32, "cpu")
    loss, out = flr.fused_linear_reduced(
        h[None].requires_grad_(True),
        w,
        v[None],
        names,
        weights,
        mask[None],
        decay=decay,
        target_weight=tw,
        metric_terms=("tv", "ce"),
    )
    logits, targets = h @ w.T, v @ tw.T
    assert torch.equal(out.pred_ids[0], logits.argmax(-1))
    assert torch.equal(out.target_ids[0], targets.argmax(-1))
    ref = fl.fused_linear_compound(
        h[None], w, v[None], names, ("tv", "ce"), target_weight=tw
    )
    for name in names:
        torch.testing.assert_close(out.losses[name], ref.losses[name].detach())
    torch.testing.assert_close(out.metrics["ce"], ref.metrics["ce"])
    ref_loss, (ref_h, _, _) = _reference(
        h, w, v, tw, torch.zeros_like(u), w2, mask, '{"kl_div": 0.5, "tv": 0.5}',
        False, "verifier",
    )  # fmt: skip
    torch.testing.assert_close(loss.detach(), ref_loss)


@pytest.mark.parametrize("spec", ["kl_div", "ce", "nla"])
def test_gradcheck(spec):
    h, w, v, tw, u, w2, mask = _inputs(12, 4, 15, 3, torch.float64, "cpu", seed=4)
    names, weights = _names_weights(spec)
    _, decay = _decay(12, "cpu")

    def loss(hh, uu, ww2):
        return flr.fused_linear_reduced(
            hh[None], w, v[None], names, weights, mask[None], decay=decay,
            block_size=BLOCK, extra=uu[None], extra_weight=ww2, target_weight=tw,
            chunk_rows=5,
        )[0]  # fmt: skip

    inputs = tuple(t.clone().requires_grad_(True) for t in (h, u, w2))
    assert torch.autograd.gradcheck(loss, inputs)


def test_validation():
    h, w, v, tw, u, w2, mask = _inputs(8, 4, 10, 2, torch.float64, "cpu")
    with pytest.raises(ValueError, match="length"):
        flr.fused_linear_reduced(h[None], w, v[None], ["ce"], [], mask[None])
    with pytest.raises(ValueError, match="go together"):
        flr.fused_linear_reduced(
            h[None], w, v[None], ["ce"], [1.0], mask[None], extra=u[None],
            target_weight=tw,
        )  # fmt: skip
    with pytest.raises(ValueError, match="whole blocks"):
        flr.fused_linear_reduced(
            h[None, :7], w, v[None, :7], ["ce"], [1.0], mask[None, :7],
            dpace_alpha=0.5, block_size=BLOCK, target_weight=tw,
        )  # fmt: skip
    with pytest.raises(ValueError, match="frozen"):
        flr.fused_linear_reduced(
            h[None], w.clone().requires_grad_(True), v[None], ["ce"], [1.0],
            mask[None], target_weight=tw,
        )  # fmt: skip


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@requires_accelerator
@pytest.mark.parametrize("targets_mode", ["logits", "verifier", "shared"])
@pytest.mark.parametrize(
    "spec",
    ["kl_div", "ce", '{"kl_div": 0.5, "tv": 0.5}', '{"ce": 0.1, "tv": 0.9}', "nla"],
)
def test_matches_eager_fp32_gpu(spec, targets_mode):
    shared = targets_mode == "shared"
    args = _inputs(512, 64, 3000, 32, torch.float32, DEVICE, seed=5, shared=shared)
    ref_loss, ref_grads = _reference(*args, spec, False, targets_mode)
    loss, grads, _ = _run(*args, spec, False, targets_mode, chunk_rows=128)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-6)
    for got, ref in zip(grads, ref_grads, strict=True):
        atol = 1e-4 * float(ref.abs().max())
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=atol)


@requires_accelerator
def test_dpace_gpu():
    args = _inputs(512, 64, 3000, 32, torch.float32, DEVICE, seed=6, shared=True)
    ref_loss, ref_grads = _reference(*args, "ce", True, "shared")
    loss, grads, _ = _run(*args, "ce", True, "shared", chunk_rows=100)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-6)
    for got, ref in zip(grads, ref_grads, strict=True):
        atol = 1e-4 * float(ref.abs().max())
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=atol)


@requires_accelerator
@pytest.mark.parametrize("logits_dtype", ["input", "float16", "float32"])
def test_bf16_tiles_gpu(logits_dtype):
    """bf16 inputs: every tile mode tracks the fp32 reference gradient."""
    args32 = _inputs(512, 64, 3000, 32, torch.float32, DEVICE, seed=7, shared=True)
    spec = '{"kl_div": 0.5, "tv": 0.5}'
    ref_loss, ref_grads = _reference(*args32, spec, False, "shared")
    h, w, v, tw, u, w2, mask = args32
    h, w, v, tw, u, w2 = (t.to(torch.bfloat16) for t in (h, w, v, tw, u, w2))
    loss, grads, _ = _run(
        h, w, v, tw, u, w2, mask, spec, False, "shared", logits_dtype=logits_dtype
    )
    torch.testing.assert_close(loss.float(), ref_loss, rtol=2e-2, atol=1e-3)
    for got, ref in zip(grads, ref_grads, strict=True):
        cosine = torch.nn.functional.cosine_similarity(
            got.float().flatten(), ref.flatten(), dim=0
        )
        assert cosine > 0.99, (logits_dtype, float(cosine))


@requires_accelerator
def test_compiles():
    h, w, v, tw, u, w2, mask = _inputs(
        256, 64, 2000, 16, torch.bfloat16, DEVICE, seed=8, shared=True
    )
    names, weights = _names_weights('{"kl_div": 0.5, "tv": 0.5}')
    _, decay = _decay(256, DEVICE)

    def step(hh, uu, ww2):
        loss, out = flr.fused_linear_reduced(
            hh[None], w, v[None], names, weights, mask[None], decay=decay,
            block_size=BLOCK, extra=uu[None], extra_weight=ww2, target_weight=w,
            shared_head=True, logits_dtype="float16", metric_terms=("tv",),
        )  # fmt: skip
        return loss, out.metrics["tv"]

    grads = {}
    for label, fn in (
        ("eager", step),
        ("compiled", torch.compile(step, fullgraph=True)),
    ):
        leaves = [t.clone().requires_grad_(True) for t in (h, u, w2)]
        loss, _ = fn(*leaves)
        loss.backward()
        grads[label] = (loss.detach(), [t.grad for t in leaves])
    torch.testing.assert_close(grads["compiled"][0], grads["eager"][0])
    for got, ref in zip(grads["compiled"][1], grads["eager"][1], strict=True):
        torch.testing.assert_close(got, ref)


@requires_accelerator
@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize(
    "names",
    [
        ("kl_div", "ce"),
        ("kl_div", "tv"),
        ("ce", "tv"),
        ("rkl", "jsd"),
        tuple(fused.TERM_SLOTS),
    ],
    ids="+".join,
)
def test_kernel_fold_matches_weighted_slabs(names, delta):
    """FOLD writes sum_t w_t[row] * dz_t: equal to the unfolded slabs."""
    torch.manual_seed(0)
    rows, vocab = 64, 151936
    z = (torch.randn(rows, vocab, device=DEVICE) * 2).to(torch.bfloat16)
    t = torch.randn(rows, vocab, device=DEVICE) * 2
    mask = fused.terms_mask(names)
    weights = torch.rand(fused.N_TERMS, rows, device=DEVICE)
    slot = {n: i for i, n in enumerate(names)}
    outs = {}
    for fold in (False, True):
        losses = torch.zeros(5, rows, device=DEVICE)
        argmax = torch.empty(2, rows, device=DEVICE, dtype=torch.int64)
        dz = torch.empty(1 if fold else len(names), rows, vocab, device=DEVICE)
        fused.loss_grad_kernel[(rows,)](
            z, t, losses, rows, argmax, rows, dz, rows * vocab, vocab,
            weights if fold else None, rows if fold else 0,
            TERMS=mask, GRAD_TERMS=mask,
            DZ_KL=slot.get("kl_div", 0), DZ_RKL=slot.get("rkl", 0),
            DZ_JSD=slot.get("jsd", 0), DZ_CE=slot.get("ce", 0), DZ_TV=slot.get("tv", 0),
            DELTA=delta, FOLD=fold,
        )  # fmt: skip
        outs[fold] = (losses, argmax, dz)
    expected = sum(
        weights[fused.TERM_SLOTS[n]][:, None] * outs[False][2][slot[n]] for n in names
    )
    torch.testing.assert_close(outs[True][0], outs[False][0], rtol=1e-5, atol=1e-6)
    assert torch.equal(outs[True][1], outs[False][1])
    torch.testing.assert_close(outs[True][2][0], expected, rtol=1e-4, atol=1e-7)
