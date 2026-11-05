# ruff: noqa: SLF001 -- built from fused_linear's chunk helpers
"""Fused linear loss with the reduction inside: scalar loss, summed linears.

``fused_linear_compound`` returns per-row losses and finishes the gradient in
backward, when the per-row loss weights arrive as ``grad_output``. That
needs every projection weight to be frozen: a trainable weight's gradient
``dW = sum_n g_n dz_n^T x_n`` sums over rows *with* their weights ``g_n``.

This op owns the reduction instead -- masking, fixed decay or D-PACE,
composite losses (``nla``, ``lk_hybrid``) and term weights, exactly as
``losses.utils._weighted_sum`` -- and returns the scalar loss. Every row
weight is known per chunk right after the loss kernel (D-PACE weights are a
no-grad function of the chunk's own losses), so the whole gradient is formed
in forward: ``G_x = sum_t g_t * (dz_t @ W)`` for every input and
``G_W += dz_t^T (g_t * x)`` (fp32) for a trainable weight. Backward scales
them by the scalar loss gradient: no GEMM and nothing recomputed. Exact as
long as the returned loss is used linearly (summed, scaled), which a
training loss is.

Draft logits may be a sum of linears, ``hidden @ weight.T + extra @
extra_weight.T`` (DSpark: backbone ``lm_head`` plus the Markov bias
``markov_w2(u)``, with ``markov_w2`` trainable); the bias is added in the
GEMM epilogue, so no ``[N, V]`` tensor exists. Tiles, the shared head and
chunking are those of ``fused_linear``.
"""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from speculators.losses import fused
from speculators.losses import fused_linear as fl
from speculators.losses.utils import _LOSS_REDUCTION_EPS, dpace_loss_decay

__all__ = ["REDUCED_NAMES", "fused_linear_reduced"]

# Named losses this op can reduce, i.e. those with a closed-form per-row
# derivative w.r.t. their primitive terms in _term_multipliers.
REDUCED_NAMES = tuple(fused.LOSS_TERMS)
_NLA_EPS = 1e-5  # fused.compose_loss's clamp for nla
_LK_ETA = 3.0  # fused.compose_loss's default eta for lk_hybrid
_HIDDEN_DIMS = 3  # [1, N, H]


# ---------------------------------------------------------------------------
# Reduction: the scalar loss of one chunk and d loss / d (primitive term)
# ---------------------------------------------------------------------------


def _composed(name: str, terms: dict[str, torch.Tensor]):
    """``(elementwise loss, {term: d loss / d term})`` for one named loss;
    the values match ``fused.compose_loss``."""
    if name == "nla":
        alive = 1.0 - terms["tv"]
        clamped = alive.clamp_min(_NLA_EPS)
        grad = torch.where(alive > _NLA_EPS, 1.0 / clamped, torch.zeros_like(alive))
        return -torch.log(clamped), {"tv": grad}
    if name == "lk_hybrid":
        kl, tv = terms["kl_div"], terms["tv"]
        weight = torch.exp(-_LK_ETA * (1.0 - tv))  # detached in compose_loss
        return weight * kl + (1.0 - weight) * tv, {"kl_div": weight, "tv": 1.0 - weight}
    return terms[name], {name: torch.ones_like(terms[name])}


def _chunk_reduction(
    losses, names, name_weights, mask, decay, denom, dpace_alpha, block_size
):
    """This chunk's share of the scalar loss and the per-row multiplier of
    each primitive term: ``loss = sum_name w * sum_n mask * decay * el / denom``
    with ``el = compose(name)``, as ``_weighted_sum`` reduces it."""
    terms = {t: losses[fused.TERM_SLOTS[t]] for t in fused.TERM_SLOTS}
    total = losses.new_zeros(())
    mult = {t: torch.zeros_like(mask) for t in fused.TERM_SLOTS}
    for name, w in zip(names.split(","), name_weights, strict=True):
        el, dterms = _composed(name, terms)
        el = el * mask
        if dpace_alpha is not None:
            row_decay = dpace_loss_decay(
                mask[None],
                mask[None],
                block_size,
                dpace_alpha,
                elementwise_loss=el[None],
            )[0].to(el.dtype)
        else:
            row_decay = decay
        scale = w * row_decay * mask / denom
        total = total + (el * row_decay).sum() * (w / denom)
        for t, d in dterms.items():
            mult[t] = mult[t] + scale * d
    return total, mult


# Named losses whose value is their primitive term (multiplier 1).
_LINEAR_NAMES = ("kl_div", "rkl", "jsd", "ce", "tv")


def _static_fold(names, name_weights, mask, decay, denom, acc):
    """``[_N_TERMS, rows]`` fp32 per-term row weights for the kernel's FOLD
    mode, when they do not depend on the losses (fixed decay, linear names):
    ``sum_name w * mask * decay / denom`` -- ``_chunk_reduction``'s
    multipliers for that case."""
    scale = mask.to(acc) * decay.to(acc) / denom.to(acc)
    fold = scale.new_zeros((fused.N_TERMS, scale.shape[0]))
    for name, w in zip(names.split(","), name_weights, strict=True):
        fold[fused.TERM_SLOTS[name]] += w * scale
    return fold.to(torch.float32)


# ---------------------------------------------------------------------------
# Chunk pieces
# ---------------------------------------------------------------------------


def _cat_head(weight, extra_weight, tile):
    """``[W | W2]`` ``[V, H + r]`` for the summed linears on accelerators:
    one projection GEMM over K = H + r instead of a slow K = r bias GEMM plus
    an epilogue add (MI355X, 2048 rows, V = 152k: +0.24 ms vs ~2.4 ms). The
    frozen ``W`` part is built once and cached with the weight; the trainable
    ``W2`` part (V x r) is rewritten every call. (Concatenating the
    contraction the same way, N = H + r, hits hipBLASLt kernels 2.5x slower
    than N = H, so the contractions stay separate.)"""
    hid, rank = weight.shape[1], extra_weight.shape[1]
    dtype = torch.float16 if fl._fp16_tile(weight.dtype, tile) else weight.dtype

    def build(w):
        buf = w.new_empty((w.shape[0], hid + rank), dtype=dtype)
        buf[:, :hid] = w
        return buf

    cat = fl._weight_copy(weight, f"cat:{dtype}:{rank}", build)
    with torch.no_grad():
        cat[:, hid:].copy_(extra_weight)
    return cat


def _project_sum(x, w, extra, extra_w, tile, cat=None):
    """``x @ w.T (+ extra @ extra_w.T)`` as one logits tile. With ``cat``
    (``_cat_heads``' projection weight) a single GEMM over ``[x | extra]``;
    otherwise the second GEMM runs first and the first accumulates onto it."""
    if extra is None:
        return fl._project(x, w, tile)
    if cat is not None:
        xc = torch.cat([x.to(cat.dtype), extra.to(cat.dtype)], dim=1)
        with torch.autocast(x.device.type, enabled=False):
            if tile == "float32" and cat.dtype in (torch.float16, torch.bfloat16):
                return torch.mm(xc, cat.t(), out_dtype=torch.float32)
            return xc @ cat.t()
    bias = fl._project(extra, extra_w, tile)
    with torch.autocast(x.device.type, enabled=False):
        if bias.dtype == torch.float32 and x.dtype in (torch.float16, torch.bfloat16):
            return bias.add_(torch.mm(x, w.t(), out_dtype=torch.float32))
        if fl._fp16_tile(x.dtype, tile):
            x, w = x.to(torch.float16), w.to(torch.float16)
        return torch.addmm(bias, x, w.t())


def _tiles(h, w, u, w2, v, tw, start, end, tile, delta, cat=None):
    """Draft tile (or delta tile) and target tile for rows ``start:end``.
    With ``cat`` and shared heads the target uses its ``W`` columns."""
    u_rows = _rows(u, start, end)
    if cat is not None and tw is w:
        tw = cat[:, : h.shape[1]]
    if not delta:
        z = _project_sum(h[start:end], w, u_rows, w2, tile, cat)
        t = fl._target_chunk(v, tw, start, end, tile)
        return z, t
    acc = fl._out_dtype(h)
    fp16 = fl._fp16_tile(h.dtype, tile)
    diff = (h[start:end].to(acc) - v[start:end].to(acc)).to(
        torch.float16 if fp16 else h.dtype
    )
    z = _project_sum(diff, w, u_rows, w2, "float16" if fp16 else "input", cat)
    t = fl._project(v[start:end], tw, tile)
    del diff
    if h.device.type == "cpu":
        dtype = torch.promote_types(acc, t.dtype)
        z = z.to(dtype) + t.to(dtype)
    return z, t


def _rows(x, start, end):
    return None if x is None else x[start:end]


def _chunk_dz(
    z, t, terms, grad_terms, losses, argmax, start, end, delta, dz_dtype, fold=None
):
    """Losses and argmaxes of the chunk (written in place) and each
    differentiated term's unit-weight ``d loss / d z``, ``[n, rows, V]``."""
    names = fl._grad_term_names(grad_terms)
    rows, vocab = end - start, z.shape[1]
    if z.device.type == "cpu":
        stats = z.new_zeros((fused.N_STATS, rows))
        fl._ref_forward_chunk(
            z, t, terms, losses[:, start:end], argmax[:, start:end], stats
        )
        units = []
        for name in names:
            unit = z.new_zeros((fused.N_TERMS, rows))
            unit[fused.TERM_SLOTS[name]] = 1.0
            units.append(
                fl._ref_backward_chunk(z, t, unit, stats, fused.TERM_BITS[name])
            )
        return torch.stack(units).to(dz_dtype) if units else None
    if not names:
        stats = z.new_empty((fused.N_STATS, rows), dtype=torch.float32)
        fused.loss_forward_kernel[(rows,)](
            z,
            t,
            losses[:, start:],
            losses.stride(0),
            argmax[:, start:],
            argmax.stride(0),
            stats,
            stats.stride(0),
            vocab,
            TERMS=terms,
            DELTA=delta,
        )
        return None
    slabs = 1 if fold is not None else len(names)
    dz = torch.empty((slabs, rows, vocab), device=z.device, dtype=dz_dtype)
    slot = {name: k for k, name in enumerate(names)}
    fused.loss_grad_kernel[(rows,)](
        z,
        t,
        losses[:, start:],
        losses.stride(0),
        argmax[:, start:],
        argmax.stride(0),
        dz,
        rows * vocab,
        vocab,
        fold,
        0 if fold is None else fold.stride(0),
        TERMS=terms,
        GRAD_TERMS=grad_terms,
        DZ_KL=slot.get("kl_div", 0),
        DZ_RKL=slot.get("rkl", 0),
        DZ_JSD=slot.get("jsd", 0),
        DZ_CE=slot.get("ce", 0),
        DZ_TV=slot.get("tv", 0),
        DELTA=delta,
        FOLD=fold is not None,
    )
    return dz


def _mm32(a, b):
    """``a @ b`` with a result in fp32 (or wider)."""
    with torch.autocast(a.device.type, enabled=False):
        if a.dtype in (torch.float16, torch.bfloat16):
            return torch.mm(a, b, out_dtype=torch.float32)
        return a @ b


_SPLIT_K = (16, 8, 4, 2)


def _contract_extra(dz: torch.Tensor, extra_weight: torch.Tensor) -> torch.Tensor:
    """``dz @ extra_weight`` (``[rows, V] @ [V, r]``) in fp32. The output is
    a few tiles for a whole GPU, so the vocab dim is split across a batched
    GEMM and summed (MI355X, 2048 x 152k x 256: 0.59 vs 1.30 ms)."""
    rows, vocab = dz.shape
    if dz.device.type == "cpu" or dz.dtype not in (torch.float16, torch.bfloat16):
        return _mm32(dz, extra_weight)
    split = next((s for s in _SPLIT_K if vocab % s == 0), 1)
    if split == 1 or not extra_weight.is_contiguous():
        return _mm32(dz, extra_weight)
    with torch.autocast(dz.device.type, enabled=False):
        parts = torch.bmm(
            dz.view(rows, split, vocab // split).transpose(0, 1),
            extra_weight.view(split, vocab // split, -1),
            out_dtype=torch.float32,
        )
    return parts.sum(0)


def _accumulate_extra_weight(acc: torch.Tensor, dz: torch.Tensor, x: torch.Tensor):
    """``acc += dz.T @ x`` in the GEMM epilogue (no fp32 [V, r] temporary)."""
    with torch.autocast(dz.device.type, enabled=False):
        if dz.dtype in (torch.float16, torch.bfloat16):
            torch.addmm(acc, dz.t(), x, out_dtype=torch.float32, out=acc)
        else:
            acc.addmm_(dz.t(), x)


@triton.jit
def _fold_kernel(
    dz_ptr,
    g_ptr,
    n_cols,
    term_stride,
    TERMS: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    """``dz[0, row] = sum_k g[k, row] * dz[k, row]`` in place (fp32 math)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = dz_ptr + row * n_cols + cols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TERMS):
        g = tl.load(g_ptr + k * tl.num_programs(0) + row)
        acc += g * tl.load(base + k * term_stride, mask=mask, other=0.0).to(tl.float32)
    tl.store(base, acc.to(dz_ptr.dtype.element_ty), mask=mask)


def _fold(dz: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Fold ``[n, rows, V]`` per-term gradients with ``g`` ``[n, rows]`` into
    ``dz[0]`` (returned): one tile to contract instead of ``n``."""
    n, rows, vocab = dz.shape
    if dz.device.type == "cpu":
        dz[0] = (dz.to(g.dtype) * g[:, :, None]).sum(0).to(dz.dtype)
        return dz[0]
    g = g.to(torch.float32).contiguous()
    block = 4096
    _fold_kernel[(rows, triton.cdiv(vocab, block))](
        dz, g, vocab, rows * vocab, TERMS=n, BLOCK=block
    )
    return dz[0]


# ---------------------------------------------------------------------------
# Custom op
# ---------------------------------------------------------------------------


def _op_weights(hidden, weight, extra, extra_weight, target_weight, tile, delta):
    """``(cat, tile_w, tile_tw, tile_w2)``: the projection operands."""
    cat = None
    if extra is not None and extra_weight is not None and hidden.device.type != "cpu":
        cat = _cat_head(weight, extra_weight, tile)
        # the draft side projects through cat; shared targets use its W columns
        tile_w = weight
        tile_tw = weight if target_weight is weight or delta else target_weight
        if tile_tw is not weight:
            tile_tw = fl._tile_weights(target_weight, None, tile, False)[0]
    else:
        tile_w, tile_tw = fl._tile_weights(weight, target_weight, tile, delta)
    tile_w2 = extra_weight
    if extra_weight is not None and fl._fp16_tile(extra_weight.dtype, tile):
        tile_w2 = extra_weight.to(torch.float16)  # trainable: not cached
    return cat, tile_w, tile_tw, tile_w2


@torch.library.custom_op("speculators::fused_linear_reduced_fwd", mutates_args=())
def _reduced_fwd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    extra: torch.Tensor | None,
    extra_weight: torch.Tensor | None,
    targets: torch.Tensor,
    target_weight: torch.Tensor | None,
    mask: torch.Tensor,
    decay: torch.Tensor,
    denom: torch.Tensor,
    names: str,
    name_weights: list[float],
    dpace_alpha: float,
    block_size: int,
    terms: int,
    grad_terms: int,
    grad_extra_weight: bool,
    chunk_rows: int,
    tile: str,
    delta: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """``(loss, losses [_N_TERMS, N], argmax [2, N], G_hidden, G_extra,
    G_extra_weight)``; the ``G`` are ``d loss / d input`` (fp32 or wider),
    empty when not differentiated. ``dpace_alpha < 0`` selects the fixed
    ``decay``. ``delta``: shared-head form, ``target_weight is weight``."""
    rows = hidden.shape[0]
    acc = fl._out_dtype(hidden)
    losses = hidden.new_zeros((fused.N_TERMS, rows), dtype=acc)
    argmax = hidden.new_zeros((2, rows), dtype=torch.int64)
    total = hidden.new_zeros((), dtype=acc)
    names_grad = fl._grad_term_names(grad_terms)
    g_hidden = hidden.new_zeros(hidden.shape if names_grad else (0,), dtype=acc)
    with_extra = extra is not None and bool(names_grad)
    with_extra = extra is not None and extra_weight is not None and bool(names_grad)
    g_extra = hidden.new_zeros(
        extra.shape if extra is not None and with_extra else (0,), dtype=acc
    )
    grad_w2 = with_extra and grad_extra_weight
    g_extra_weight = hidden.new_zeros(
        extra_weight.shape if extra_weight is not None and grad_w2 else (0,),
        dtype=acc,
    )
    cat, tile_w, tile_tw, tile_w2 = _op_weights(
        hidden, weight, extra, extra_weight, target_weight, tile, delta
    )
    alpha = None if dpace_alpha < 0 else dpace_alpha
    # Row weights known before the kernel: it writes one weighted slab.
    static = (
        alpha is None
        and len(names_grad) > 1
        and hidden.device.type != "cpu"
        and all(name in _LINEAR_NAMES for name in names.split(","))
    )
    for start, end in fl._chunks(rows, chunk_rows):
        z, t = _tiles(
            hidden, tile_w, extra, tile_w2, targets, tile_tw, start, end, tile,
            delta, cat,
        )  # fmt: skip
        fold = None
        if static:
            fold = _static_fold(
                names, name_weights, mask[start:end], decay[start:end], denom, acc
            )
        dz = _chunk_dz(
            z,
            t,
            terms,
            grad_terms,
            losses,
            argmax,
            start,
            end,
            delta,
            weight.dtype,
            fold,
        )
        del z, t
        part, mult = _chunk_reduction(
            losses[:, start:end],
            names,
            name_weights,
            mask[start:end].to(acc),
            decay[start:end].to(acc),
            denom.to(acc),
            alpha,
            block_size,
        )
        total += part
        if dz is None:
            continue
        if fold is not None:  # the kernel wrote the weighted sum
            tile_dz, g = dz[0], None
        elif len(names_grad) == 1:  # scale the [rows, H] results instead
            tile_dz, g = dz[0], mult[names_grad[0]][:, None]
        else:  # one contraction for all terms
            tile_dz = _fold(dz, torch.stack([mult[name] for name in names_grad]))
            g = None
        part_h = fl._contract(tile_dz, weight).to(acc)
        g_hidden[start:end] = part_h if g is None else g * part_h
        if with_extra and extra is not None and extra_weight is not None:
            part_u = _contract_extra(tile_dz, extra_weight)
            g_extra[start:end] = part_u if g is None else g * part_u
        if grad_w2 and extra is not None:
            x = extra[start:end]
            if g is not None:
                x = (g * x.to(acc)).to(tile_dz.dtype)
            _accumulate_extra_weight(g_extra_weight, tile_dz, x)
        del dz
    return total, losses, argmax, g_hidden, g_extra, g_extra_weight


@_reduced_fwd.register_fake
def _(hidden, weight, extra, extra_weight, targets, target_weight, *args):
    del weight, targets, target_weight
    grad_terms, grad_extra_weight = args[-5], args[-4]
    rows = hidden.shape[0]
    acc = fl._out_dtype(hidden)
    differentiated = bool(fl._grad_term_names(grad_terms))
    with_extra = extra is not None and differentiated
    grad_w2 = with_extra and grad_extra_weight
    return (
        hidden.new_empty((), dtype=acc),
        hidden.new_empty((fused.N_TERMS, rows), dtype=acc),
        hidden.new_empty((2, rows), dtype=torch.int64),
        hidden.new_empty(hidden.shape if differentiated else (0,), dtype=acc),
        hidden.new_empty(extra.shape if with_extra else (0,), dtype=acc),
        hidden.new_empty(extra_weight.shape if grad_w2 else (0,), dtype=acc),
    )


def _setup_context(ctx, inputs, output):
    hidden, _, extra, extra_weight = inputs[:4]
    _, losses, argmax, g_hidden, g_extra, g_extra_weight = output
    ctx.mark_non_differentiable(losses, argmax, g_hidden, g_extra, g_extra_weight)
    ctx.dtypes = (
        hidden.dtype,
        None if extra is None else extra.dtype,
        None if extra_weight is None else extra_weight.dtype,
    )
    ctx.save_for_backward(g_hidden, g_extra, g_extra_weight)


def _backward(ctx, grad_loss, *_):
    g_hidden, g_extra, g_extra_weight = ctx.saved_tensors
    grads: list[torch.Tensor | None] = [None] * 19
    for index, g, dtype in zip(
        (0, 2, 3), (g_hidden, g_extra, g_extra_weight), ctx.dtypes, strict=True
    ):
        if ctx.needs_input_grad[index] and g.numel():
            grads[index] = (g * grad_loss).to(dtype)
    return tuple(grads)


_reduced_fwd.register_autograd(_backward, setup_context=_setup_context)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fused_linear_reduced(  # noqa: C901 -- argument validation
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    names: Sequence[str],
    name_weights: Sequence[float],
    loss_mask: torch.Tensor,
    decay: torch.Tensor | None = None,
    dpace_alpha: float | None = None,
    block_size: int = 1,
    extra: torch.Tensor | None = None,
    extra_weight: torch.Tensor | None = None,
    target_weight: torch.Tensor | None = None,
    logits_dtype: str = "input",
    shared_head: bool = False,
    metric_terms: Sequence[str] = (),
    chunk_rows: int | None = None,
) -> tuple[torch.Tensor, "fused.FusedOutputs"]:
    """Scalar compound loss of ``hidden @ weight.T (+ extra @ extra_weight.T)``.

    Args:
        hidden: ``[1, N, H]`` final draft hidden states.
        weight: ``[V, H]`` frozen draft ``lm_head`` weight.
        targets: ``[1, N, V]`` target logits, or ``[1, N, H_t]`` verifier
            hidden states with ``target_weight``.
        names / name_weights: named losses (``REDUCED_NAMES``) and weights,
            as a resolved ``LossConfig``.
        loss_mask: ``[1, N]``.
        decay: ``[1, N]`` fixed per-row decay (``None``: none). Ignored
            with ``dpace_alpha``, which applies D-PACE per block of
            ``block_size`` rows (the CE-only weighting of ``dpace_loss_decay``).
        extra / extra_weight: optional second linear, ``[1, N, r]`` and
            ``[V, r]``; ``extra_weight`` may be trainable.
        target_weight / logits_dtype / shared_head: as ``fused_linear_compound``.
        metric_terms: primitive terms computed in the same pass, detached.

    Returns:
        ``(loss, outputs)``: the differentiable scalar loss, equal to
        ``losses.utils._weighted_sum`` of the same per-row losses, and
        ``FusedOutputs`` with detached per-row losses per name, metrics and
        both argmaxes.
    """
    if len(names) != len(name_weights):
        raise ValueError("names and name_weights differ in length")
    for name in names:
        if name not in REDUCED_NAMES:
            raise ValueError(f"fused_linear_reduced does not support {name!r}")
    if logits_dtype not in fl.LOGITS_DTYPES:
        raise ValueError(
            f"logits_dtype must be one of {fl.LOGITS_DTYPES}, got {logits_dtype!r}"
        )
    if (
        hidden.dim() != _HIDDEN_DIMS
        or hidden.shape[0] != 1
        or loss_mask.shape != hidden.shape[:2]
    ):
        raise ValueError("expected hidden [1, N, H] and loss_mask [1, N]")
    if (extra is None) != (extra_weight is None):
        raise ValueError("extra and extra_weight go together")
    flat_hidden, weight, flat_targets, target_weight = fl._prepare_inputs(
        hidden, weight, targets, target_weight, shared_head
    )
    rows, vocab = flat_hidden.shape[0], weight.shape[0]
    flat_extra = None
    if extra is not None and extra_weight is not None:
        flat_extra = extra.reshape(rows, -1)
        if torch.is_autocast_enabled(flat_hidden.device.type):
            flat_extra = flat_extra.to(flat_hidden.dtype)
            extra_weight = extra_weight.to(flat_hidden.dtype)
        if extra_weight.shape != (vocab, flat_extra.shape[1]):
            raise ValueError(
                f"extra {tuple(extra.shape)} and extra_weight "
                f"{tuple(extra_weight.shape)} do not match vocab {vocab}"
            )
        if flat_extra.dtype != flat_hidden.dtype or extra_weight.dtype != weight.dtype:
            raise ValueError("extra / extra_weight dtypes must match hidden / weight")
    if dpace_alpha is not None and rows % block_size:
        raise ValueError(f"D-PACE needs whole blocks: {rows} rows, block {block_size}")

    loss_terms = sorted(
        {t for name in names for t in fused.LOSS_TERMS[name]},
        key=lambda t: fused.TERM_BITS[t],
    )
    grad_mask = fused.terms_mask(loss_terms)
    all_terms = grad_mask | fused.terms_mask(metric_terms)
    needs_grad = torch.is_grad_enabled() and (
        flat_hidden.requires_grad
        or (flat_extra is not None and flat_extra.requires_grad)
        or (extra_weight is not None and extra_weight.requires_grad)
    )
    delta = shared_head and bool(all_terms & fused.DIST_MASK)
    # the kernel writes one weighted slab when the row weights are static
    folds = dpace_alpha is None and all(name in _LINEAR_NAMES for name in names)
    if chunk_rows is None:
        chunk_rows = fl.chunk_rows_for(
            vocab,
            flat_hidden.dtype,
            rows,
            recompute_targets=target_weight is not None,
            fp32_logits=logits_dtype == "float32",
            dz_tiles=1 if folds else len(loss_terms),
            shared_head=delta,
        )
    if dpace_alpha is not None:  # D-PACE weights are per block
        chunk_rows = max(chunk_rows // block_size, 1) * block_size
    mask = loss_mask.reshape(-1)
    decay_rows = (
        torch.ones_like(mask, dtype=torch.float32)
        if decay is None
        else decay.reshape(-1)
    )
    denom = mask.sum() + _LOSS_REDUCTION_EPS
    loss, losses, argmax, *_ = _reduced_fwd(
        flat_hidden,
        weight,
        flat_extra,
        extra_weight,
        flat_targets,
        target_weight,
        mask,
        decay_rows,
        denom,
        ",".join(names),
        [float(w) for w in name_weights],
        -1.0 if dpace_alpha is None else float(dpace_alpha),
        block_size,
        all_terms,
        grad_mask if needs_grad else 0,
        bool(extra_weight is not None and extra_weight.requires_grad),
        chunk_rows,
        logits_dtype,
        delta,
    )
    lead = hidden.shape[:-1]
    terms = {t: losses[fused.TERM_SLOTS[t]].view(lead) for t in loss_terms}
    return loss, fused.FusedOutputs(
        losses={name: fused.compose_loss(name, terms) for name in names},
        metrics={t: losses[fused.TERM_SLOTS[t]].view(lead) for t in metric_terms},
        pred_ids=argmax[0].view(lead),
        target_ids=argmax[1].view(lead),
    )
