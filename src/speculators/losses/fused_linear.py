"""Fused linear losses: drafter losses straight from hidden states.

The draft logits ``hidden @ weight.T`` are computed one row chunk at a time
into a scratch tile, reduced by the vocab kernels from ``losses.fused``, and
dropped, so the ``[N, V]`` draft logits and their gradient never exist.
Targets come in either as materialized ``[N, V]`` logits, or as the post-norm
verifier hidden states ``[N, H_t]`` plus the frozen verifier ``lm_head``
weight, in which case each chunk's target logits are computed the same way
and no ``[N, V]`` tensor exists at all. Targets receive no gradient; the
draft weight must be frozen.

Gradients, two modes:

- **deferred** (default for at most two differentiated terms): forward also
  writes each term's unit-weight logits gradient for the chunk and contracts
  it with the weight right away, ``jac_t = (d loss_t / d z) @ W``. Backward is
  ``d hidden = sum_t grad_output_t[:, None] * jac_t``: no GEMM, nothing
  recomputed, and only ``jac`` (``[terms, N, H]``) is kept for backward. This
  is exact because every loss weight -- mask, decay, D-PACE, term weight --
  is a per-row scalar applied after the loss. Same GEMM count as
  materialized logits (draft, target, ``dz @ W``).
- **recompute**: forward computes losses and per-row statistics only;
  backward recomputes each chunk's logits and writes ``dz @ W``.

Logit tiles are in the input dtype by default (bf16 in training, like
materialized logits); ``logits_dtype="float32"`` takes them in fp32 straight
from the GEMM accumulator -- more accurate, ~20% slower GEMMs;
``logits_dtype="float16"`` casts bf16 operands to fp16 (exact for weights
and activations of magnitude >= 2**-14) for fp16 tiles: 3 more mantissa bits
than bf16 for the same bytes (fp16 GEMMs run ~5% slower than bf16 on MI355X;
the fp16 weight copies are cached across calls, see ``_weight_copy``).

**Shared head** (``shared_head=True``, the drafter reuses the verifier's
``lm_head``, as DFlash does): the draft logits are formed as
``z_t + (hidden - verifier_hidden) @ W.T``, both tiles in ``logits_dtype``
(the delta tile in the input dtype for ``"float32"``). The delta tile's
rounding error scales with the small delta instead of the logits, so the
draft-target logit *difference* -- what KL / TV gradients depend on -- is
near-exact: with fp16 tiles KL / TV gradient errors drop to ~0.3-0.4 %
(fp32 tiles: ~0.2 %; bf16: 16-87 % at acceptance 0.88-0.98, V = 152k) at
the GEMM count and tile bytes of the plain path. It is used only with a
distribution term (CE's gradient has no target to cancel against). The
subtraction is exact in fp32 and rounded once. The gradient is unchanged:
``d z / d hidden = W``.

The per-row loss contract is unchanged: unreduced per-term losses, a per-row
``grad_output`` per term, so masking, decay, D-PACE and term weights stay in
``losses.utils``. The chunk loops are ``torch.library.custom_op``s, opaque to
torch.compile (traced, they would unroll per chunk count and recompile when
the row count changes).
"""

import os
import weakref
from collections.abc import Sequence

import torch

from speculators.losses import fused

__all__ = ["chunk_rows_for", "fused_linear_compound"]

# Peak scratch the chunk loop may use, in MiB, and the chunk-size cap.
# Measured on MI355X (hipBLASLt, H = 4096, V = 152k, ms per 1k rows): the
# projection runs at full-GEMM speed from ~768 rows up; the dz @ W
# contraction is shape-sensitive -- 1.80-1.91 at 1024, 1792, 2048, 4096,
# 6144 and 8192 rows but 2.4-3.1 at 1536, 3072, 3584, 3840 and 5120. 2048-row
# chunks are the fastest for both GEMMs and halve the scratch of 4096, so the
# budget only lowers the chunk below that (e.g. fp32 tiles at large V).
_BUDGET_ENV = "SPECULATORS_FUSED_LINEAR_BUDGET_MIB"
_DEFAULT_BUDGET_MIB = 4096
_MIN_CHUNK_ROWS = 64
_MAX_CHUNK_ROWS = 2048
_LOG2 = 0.6931471805599453
GRAD_MODES = ("auto", "deferred", "recompute")
LOGITS_DTYPES = ("input", "float32", "float16")
# auto picks deferred up to this many differentiated terms: jac costs one
# [N, H] tensor and one dz @ W GEMM per term in forward.
_MAX_DEFERRED_TERMS = 2


def chunk_rows_for(
    vocab: int,
    dtype: torch.dtype,
    rows: int,
    recompute_targets: bool = False,
    fp32_logits: bool = False,
    dz_tiles: int = 1,
    shared_head: bool = False,
) -> int:
    """Rows per chunk: the largest power of two whose scratch fits the
    budget, capped at ``_MAX_CHUNK_ROWS``; the last chunk takes the remainder.

    Per row the loop holds a draft-logits row, a target-logits row when
    targets are recomputed (both fp32 or input dtype), and ``dz_tiles``
    input-dtype gradient rows (one per term in deferred mode). With
    ``shared_head`` the draft row is an input-dtype delta row.
    """
    budget = int(os.environ.get(_BUDGET_ENV, _DEFAULT_BUDGET_MIB)) * 1024**2
    elem = torch.empty((), dtype=dtype).element_size()
    tile = 4 if fp32_logits else elem
    draft = elem if shared_head else tile
    target = tile if recompute_targets or shared_head else 0
    per_row = vocab * (draft + target + elem * dz_tiles)
    fit = max(budget // per_row, 1)
    chunk = max(1 << (fit.bit_length() - 1), _MIN_CHUNK_ROWS)
    return min(chunk, _MAX_CHUNK_ROWS, max(rows, 1))


def _out_dtype(hidden: torch.Tensor) -> torch.dtype:
    """Loss / stats dtype: fp32, or fp64 for fp64 inputs (CPU gradcheck)."""
    return torch.promote_types(torch.float32, hidden.dtype)


def _fp16_tile(dtype: torch.dtype, tile: str) -> bool:
    """fp16 tiles apply to bf16 operands only (never a downgrade)."""
    return tile == "float16" and dtype == torch.bfloat16


# Derived copies of the frozen weights, keyed by (id(source), kind): "fp16"
# (fp16 tiles) and "t" (the [H, V] layout of the dz @ W contraction). An
# entry is valid only while its source is alive (weakref, removed with it)
# and unmodified (same version counter); rebuilding a copy of a 1.2 GB head
# costs ~1-3 ms, so they are kept across calls (one head-sized tensor each).
_WEIGHT_COPIES: dict[tuple[int, str], tuple[weakref.ref, int, torch.Tensor]] = {}


def _weight_copy(weight: torch.Tensor, kind: str) -> torch.Tensor:
    key = (id(weight), kind)
    entry = _WEIGHT_COPIES.get(key)
    version = weight._version  # noqa: SLF001 -- bumped by in-place writes
    if entry is not None and entry[0]() is weight and entry[1] == version:
        return entry[2]
    with torch.no_grad():
        if kind == "fp16":
            copy = weight.detach().to(torch.float16)
        else:
            copy = weight.detach().t().contiguous()

    def evict(_ref: weakref.ref, key: tuple[int, str] = key) -> None:
        _WEIGHT_COPIES.pop(key, None)

    _WEIGHT_COPIES[key] = (weakref.ref(weight, evict), version, copy)
    return copy


def _fp16_copy(weight: torch.Tensor) -> torch.Tensor:
    return _weight_copy(weight, "fp16")


def clear_weight_cache() -> None:
    """Drop the cached weight copies (fp16 tiles, contraction layout)."""
    _WEIGHT_COPIES.clear()


def _contract_operand(weight: torch.Tensor) -> torch.Tensor:
    """``weight`` [V, H] for ``dz @ weight``, as a view of the cached
    contiguous ``[H, V]`` copy on accelerators: hipBLASLt runs the
    contraction 6-13% faster from that layout (MI355X, 2048 rows, V = 152k)."""
    if weight.device.type == "cpu":
        return weight
    return _weight_copy(weight, "t").t()


def _tile_weights(weight, target_weight, tile, shared_head):
    """The projection operands of the chunk loop: for fp16 tiles, fp16 copies
    of the bf16 weights (exact for every weight of magnitude >= 2**-14),
    cached across calls while the weight is alive and unmodified; otherwise
    the weights themselves."""

    def cast(w):
        if w is None or not _fp16_tile(w.dtype, tile):
            return w
        return _fp16_copy(w)

    tile_w = cast(weight)
    return tile_w, tile_w if shared_head else cast(target_weight)


def _project(hidden: torch.Tensor, weight: torch.Tensor, tile: str) -> torch.Tensor:
    """``hidden @ weight.T`` as a ``tile`` logits tile: ``"input"`` dtype,
    ``"float32"`` straight from the accumulator, or ``"float16"`` (bf16
    operands cast to fp16, fp16 output: 3 more mantissa bits than bf16 for
    the same bytes; ``weight`` is then the fp16 copy from ``_tile_weights``).

    Autocast is disabled: forward and a backward recompute must produce
    bit-identical logits from the operands as given (the wrapper casts them
    for autocast), or saved statistics would not match recomputed logits.
    """
    with torch.autocast(hidden.device.type, enabled=False):
        if tile == "float32" and hidden.dtype in (torch.float16, torch.bfloat16):
            return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        if _fp16_tile(hidden.dtype, tile):
            return hidden.to(torch.float16) @ weight.to(torch.float16).t()
        return hidden @ weight.t()


def _contract(
    dz: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``dz @ weight`` in ``dz``'s dtype (the input dtype for bf16 / fp16).

    A bf16 ``jac`` is rounded once more than an fp32 one before backward
    scales it, which measured as no change for one differentiated term and
    +0.02-0.04 points of relative gradient error for two (V = 64k), while
    the bf16-output GEMM is ~6% faster and ``jac`` half the size.

    With ``out`` (a contiguous ``[rows, H]`` slice of ``jac``) the GEMM
    writes there directly instead of into a temporary that is then copied.
    """
    operand = _contract_operand(weight)
    with torch.autocast(dz.device.type, enabled=False):
        if out is None:
            return dz @ operand
        if out.dtype != dz.dtype:  # mixed-dtype inputs: cast on the copy
            out.copy_(dz @ operand)
            return out
        return torch.mm(dz, operand, out=out)


def _chunks(rows: int, chunk_rows: int):
    for start in range(0, rows, chunk_rows):
        yield start, min(start + chunk_rows, rows)


def _grad_term_names(grad_terms: int) -> list[str]:
    """Differentiated primitive terms in slot order (the jac / dz order)."""
    return [name for name, bit in fused.TERM_BITS.items() if grad_terms & bit]


# ---------------------------------------------------------------------------
# Torch reference of one chunk (CPU / fp64): same math and stats layout as the
# Triton kernels in losses.fused.
# ---------------------------------------------------------------------------


def _ref_softmax_stats(z: torch.Tensor):
    m = z.max(dim=-1).values
    s = torch.exp(z - m[:, None]).sum(dim=-1)
    lse = m + torch.log(s)
    return m, s, z - lse[:, None]


def _ref_forward_chunk(z, targets, terms, losses, argmax, stats):
    t = targets.to(z.dtype)
    m_d, s_d, ldp = _ref_softmax_stats(z)
    dp = torch.exp(ldp)
    stats[0], stats[1] = m_d, s_d
    argmax[0] = z.argmax(dim=-1)
    argmax[1] = t.argmax(dim=-1)
    if terms & fused.TERM_BITS["ce"]:
        idx = argmax[1]
        losses[3] = m_d + torch.log(s_d) - z.gather(-1, idx[:, None])[:, 0]
        stats[6] = idx.to(stats.dtype)
    if not terms & fused.DIST_MASK:
        return
    m_t, s_t, ltp = _ref_softmax_stats(t)
    tp = torch.exp(ltp)
    stats[2], stats[3] = m_t, s_t
    if terms & fused.TERM_BITS["kl_div"]:
        losses[0] = (tp * (ltp - ldp)).sum(dim=-1)
    if terms & fused.TERM_BITS["rkl"]:
        losses[1] = stats[4] = (dp * (ldp - ltp)).sum(dim=-1)
    if terms & fused.TERM_BITS["jsd"]:
        lmp = torch.logaddexp(ldp, ltp) - _LOG2
        jsd_draft = (dp * (ldp - lmp)).sum(dim=-1)
        losses[2] = 0.5 * (jsd_draft + (tp * (ltp - lmp)).sum(dim=-1))
        stats[5] = jsd_draft
    if terms & fused.TERM_BITS["tv"]:
        losses[4] = 1.0 - torch.minimum(dp, tp).sum(dim=-1)
        stats[7] = torch.where(dp <= tp, dp, 0.0).sum(dim=-1)


def _ref_backward_chunk(z, targets, grad_out, stats, terms) -> torch.Tensor:
    ldp = z - (stats[0] + torch.log(stats[1]))[:, None]
    dp = torch.exp(ldp)
    grad = torch.zeros_like(z)
    if terms & fused.TERM_BITS["ce"]:
        onehot = torch.zeros_like(z).scatter_(-1, stats[6].long()[:, None], 1.0)
        grad += grad_out[3][:, None] * (dp - onehot)
    if terms & fused.DIST_MASK:
        ltp = targets.to(z.dtype) - (stats[2] + torch.log(stats[3]))[:, None]
        tp = torch.exp(ltp)
        if terms & fused.TERM_BITS["kl_div"]:
            grad += grad_out[0][:, None] * (dp - tp)
        if terms & fused.TERM_BITS["rkl"]:
            grad += grad_out[1][:, None] * (dp * (ldp - ltp - stats[4][:, None]))
        if terms & fused.TERM_BITS["jsd"]:
            lmp = torch.logaddexp(ldp, ltp) - _LOG2
            grad += grad_out[2][:, None] * (0.5 * dp * (ldp - lmp - stats[5][:, None]))
        if terms & fused.TERM_BITS["tv"]:
            grad += grad_out[4][:, None] * (
                -dp * ((dp <= tp).to(z.dtype) - stats[7][:, None])
            )
    return grad


# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


def _target_chunk(targets, target_weight, start, end, tile):
    """Target logits for rows ``start:end``: a slice of the materialized
    logits, or computed from the verifier hidden states."""
    rows = targets[start:end]
    return rows if target_weight is None else _project(rows, target_weight, tile)


def _logit_tiles(
    hidden, weight, targets, target_weight, start, end, tile, shared_head, need_t
):
    """``(z, chunk_targets)`` for rows ``start:end``.

    Normally ``z`` is the draft logits tile. With ``shared_head`` it is the
    delta tile ``(hidden - targets) @ weight.T`` (kernels launched with
    ``DELTA``) and the target tile is always computed; on CPU the draft
    logits are materialized as ``target + delta`` for the torch reference.
    The delta tile is fp16 for fp16 tiles and in the input dtype otherwise
    (an fp32 delta tile would not be worth its GEMM).
    """
    if not shared_head:
        z = _project(hidden[start:end], weight, tile)
        if not need_t:
            return z, targets[:0]
        return z, _target_chunk(targets, target_weight, start, end, tile)
    acc = _out_dtype(hidden)
    # Round the exact difference once, to the delta tile's operand dtype.
    fp16 = _fp16_tile(hidden.dtype, tile)
    diff_dtype = torch.float16 if fp16 else hidden.dtype
    diff = (hidden[start:end].to(acc) - targets[start:end].to(acc)).to(diff_dtype)
    z = _project(diff, weight, "float16" if fp16 else "input")
    del diff
    chunk_targets = _project(targets[start:end], weight, tile)
    if hidden.device.type == "cpu":
        dtype = torch.promote_types(acc, chunk_targets.dtype)
        z = z.to(dtype) + chunk_targets.to(dtype)
    return z, chunk_targets


def _forward_deferred_chunk(
    z,
    chunk_targets,
    weight,
    terms,
    grad_terms,
    losses,
    argmax,
    jac,
    start,
    end,
    delta,
):
    """Losses, argmaxes and ``jac[:, start:end]`` for one chunk."""
    names = _grad_term_names(grad_terms)
    rows, vocab = end - start, z.shape[1]
    if z.device.type == "cpu":
        stats = z.new_zeros((fused.N_STATS, rows))
        _ref_forward_chunk(
            z, chunk_targets, terms, losses[:, start:end], argmax[:, start:end], stats
        )
        for k, name in enumerate(names):
            unit = z.new_zeros((fused.N_TERMS, rows))
            unit[fused.TERM_SLOTS[name]] = 1.0
            dz = _ref_backward_chunk(
                z, chunk_targets, unit, stats, fused.TERM_BITS[name]
            ).to(weight.dtype)
            jac[k, start:end] = _contract(dz, weight)
        return
    # dz and jac in the weight (input) dtype, like the materialized grad_in;
    # backward scales and sums the per-term jac in fp32 and rounds once.
    dz = torch.empty((len(names), rows, vocab), device=z.device, dtype=weight.dtype)
    slot = {name: k for k, name in enumerate(names)}
    fused.loss_grad_kernel[(rows,)](
        z,
        chunk_targets,
        losses[:, start:],
        losses.stride(0),
        argmax[:, start:],
        argmax.stride(0),
        dz,
        rows * vocab,
        vocab,
        TERMS=terms,
        GRAD_TERMS=grad_terms,
        DZ_KL=slot.get("kl_div", 0),
        DZ_RKL=slot.get("rkl", 0),
        DZ_JSD=slot.get("jsd", 0),
        DZ_CE=slot.get("ce", 0),
        DZ_TV=slot.get("tv", 0),
        DELTA=delta,
    )
    for k in range(len(names)):
        _contract(dz[k], weight, out=jac[k, start:end])


@torch.library.custom_op("speculators::fused_linear_loss_fwd", mutates_args=())
def _fused_linear_fwd(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor | None,
    terms: int,
    grad_terms: int,
    chunk_rows: int,
    deferred: bool,
    tile: str,
    shared_head: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Losses ``[_N_TERMS, N]``, argmax ``[2, N]``, stats ``[_N_STATS, N]``, jac.

    ``targets`` is ``[N, V]`` logits, or ``[N, H_t]`` verifier hidden states
    when ``target_weight`` (``[V, H_t]``) is given. With ``deferred``, ``jac``
    is ``[n_grad_terms, N, H]`` (one unit-weight ``d loss / d hidden`` per
    differentiated term, slot order) and stats are unused; otherwise ``jac``
    is empty and stats feed the recompute backward. Slots of terms outside
    ``terms`` are zero. ``shared_head`` requires ``target_weight is weight``
    (see the module docstring).
    """
    rows, vocab = hidden.shape[0], weight.shape[0]
    dtype = _out_dtype(hidden)
    losses = hidden.new_zeros((fused.N_TERMS, rows), dtype=dtype)
    argmax = hidden.new_zeros((2, rows), dtype=torch.int64)
    stats = hidden.new_zeros((fused.N_STATS, rows), dtype=dtype)
    n_jac = len(_grad_term_names(grad_terms)) if deferred else 0
    jac = hidden.new_empty(
        (n_jac, rows if n_jac else 0, hidden.shape[1]), dtype=hidden.dtype
    )
    tile_w, tile_tw = _tile_weights(weight, target_weight, tile, shared_head)
    for start, end in _chunks(rows, chunk_rows):
        z, chunk_targets = _logit_tiles(
            hidden,
            tile_w,
            targets,
            tile_tw,
            start,
            end,
            tile,
            shared_head,
            need_t=True,
        )
        if deferred:
            _forward_deferred_chunk(
                z,
                chunk_targets,
                weight,
                terms,
                grad_terms,
                losses,
                argmax,
                jac,
                start,
                end,
                shared_head,
            )
        elif hidden.device.type == "cpu":
            _ref_forward_chunk(
                z,
                chunk_targets,
                terms,
                losses[:, start:end],
                argmax[:, start:end],
                stats[:, start:end],
            )
        else:
            fused.loss_forward_kernel[(end - start,)](
                z,
                chunk_targets,
                losses[:, start:],
                losses.stride(0),
                argmax[:, start:],
                argmax.stride(0),
                stats[:, start:],
                stats.stride(0),
                vocab,
                TERMS=terms,
                DELTA=shared_head,
            )
        # Release the tiles before the next chunk allocates its own.
        del z, chunk_targets
    return losses, argmax, stats, jac


@_fused_linear_fwd.register_fake
def _(
    hidden,
    weight,
    targets,
    target_weight,
    terms,
    grad_terms,
    chunk_rows,
    deferred,
    tile,
    shared_head,
):
    del weight, targets, target_weight, terms, chunk_rows, tile, shared_head
    rows = hidden.shape[0]
    dtype = _out_dtype(hidden)
    n_jac = len(_grad_term_names(grad_terms)) if deferred else 0
    return (
        hidden.new_empty((fused.N_TERMS, rows), dtype=dtype),
        hidden.new_empty((2, rows), dtype=torch.int64),
        hidden.new_empty((fused.N_STATS, rows), dtype=dtype),
        hidden.new_empty(
            (n_jac, rows if n_jac else 0, hidden.shape[1]), dtype=hidden.dtype
        ),
    )


@torch.library.custom_op("speculators::fused_linear_loss_bwd", mutates_args=())
def _fused_linear_bwd(
    grad_losses: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    target_weight: torch.Tensor | None,
    stats: torch.Tensor,
    grad_terms: int,
    chunk_rows: int,
    tile: str,
    shared_head: bool,
) -> torch.Tensor:
    """Recompute backward: per chunk, recompute the logits, form ``dz``,
    write ``dz @ weight``. Target logits are only needed for distribution
    terms (and to rebuild the draft logits with ``shared_head``); CE reads
    its cached target argmax from ``stats``."""
    rows, vocab = hidden.shape[0], weight.shape[0]
    needs_targets = shared_head or bool(grad_terms & fused.DIST_MASK)
    d_hidden = torch.empty_like(hidden)
    tile_w, tile_tw = _tile_weights(weight, target_weight, tile, shared_head)
    for start, end in _chunks(rows, chunk_rows):
        z, chunk_targets = _logit_tiles(
            hidden,
            tile_w,
            targets,
            tile_tw,
            start,
            end,
            tile,
            shared_head,
            need_t=needs_targets,
        )
        if hidden.device.type == "cpu":
            dz = _ref_backward_chunk(
                z,
                chunk_targets,
                grad_losses[:, start:end],
                stats[:, start:end],
                grad_terms,
            ).to(hidden.dtype)
        else:
            dz = torch.empty((end - start, vocab), device=z.device, dtype=hidden.dtype)
            fused.loss_backward_kernel[(end - start,)](
                z,
                chunk_targets,
                dz,
                grad_losses[:, start:],
                grad_losses.stride(0),
                stats[:, start:],
                stats.stride(0),
                vocab,
                TERMS=grad_terms,
                DELTA=shared_head,
            )
        del z, chunk_targets
        with torch.autocast(hidden.device.type, enabled=False):
            torch.mm(dz, _contract_operand(weight), out=d_hidden[start:end])
        del dz
    return d_hidden


@_fused_linear_bwd.register_fake
def _(grad_losses, hidden, weight, targets, target_weight, stats, *_):
    del grad_losses, weight, targets, target_weight, stats
    return torch.empty_like(hidden)


@torch.library.custom_op("speculators::fused_linear_deferred_bwd", mutates_args=())
def _fused_linear_deferred_bwd(
    grad_losses: torch.Tensor,
    jac: torch.Tensor,
    slots: list[int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Deferred backward: ``d hidden = sum_t grad_losses[slot_t][:, None] * jac_t``,
    accumulated in fp32 (or wider) and rounded once to ``out_dtype``."""
    acc_dtype = torch.promote_types(torch.float32, jac.dtype)
    acc = torch.zeros(jac.shape[1:], device=jac.device, dtype=acc_dtype)
    for k, slot in enumerate(slots):
        acc.addcmul_(grad_losses[slot].to(acc_dtype)[:, None], jac[k])
    return acc.to(out_dtype)


@_fused_linear_deferred_bwd.register_fake
def _(grad_losses, jac, slots, out_dtype):
    del grad_losses, slots
    return jac.new_empty(jac.shape[1:], dtype=out_dtype)


def _setup_context(ctx, inputs, output):
    (
        hidden,
        weight,
        targets,
        target_weight,
        _terms,
        grad_terms,
        chunk_rows,
        deferred,
        tile,
        shared_head,
    ) = inputs
    _, argmax, stats, jac = output
    ctx.mark_non_differentiable(argmax, stats, jac)
    ctx.grad_terms = grad_terms
    ctx.chunk_rows = chunk_rows
    ctx.deferred = deferred
    ctx.tile = tile
    ctx.shared_head = shared_head
    if deferred:
        # Everything backward needs is already contracted into jac.
        ctx.grad_slots = [fused.TERM_SLOTS[n] for n in _grad_term_names(grad_terms)]
        ctx.hidden_dtype = hidden.dtype
        ctx.save_for_backward(jac)
        return
    # Backward reads the targets only for distribution terms; a CE-only loss
    # releases them after forward (CE caches the target argmax in stats),
    # unless the shared head rebuilds the draft logits from them.
    if not shared_head and not grad_terms & fused.DIST_MASK:
        targets = targets.new_empty((0, targets.shape[-1]))
        target_weight = None
    ctx.save_for_backward(hidden, weight, targets, target_weight, stats)


def _backward(ctx, grad_losses, _grad_argmax, _grad_stats, _grad_jac):
    d_hidden = None
    if ctx.deferred and ctx.needs_input_grad[0]:
        (jac,) = ctx.saved_tensors
        d_hidden = _fused_linear_deferred_bwd(
            grad_losses.contiguous(), jac, ctx.grad_slots, ctx.hidden_dtype
        )
    elif ctx.grad_terms and ctx.needs_input_grad[0]:
        hidden, weight, targets, target_weight, stats = ctx.saved_tensors
        d_hidden = _fused_linear_bwd(
            grad_losses.contiguous(),
            hidden,
            weight,
            targets,
            target_weight,
            stats,
            ctx.grad_terms,
            ctx.chunk_rows,
            ctx.tile,
            ctx.shared_head,
        )
    return d_hidden, None, None, None, None, None, None, None, None, None


_fused_linear_fwd.register_autograd(_backward, setup_context=_setup_context)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _prepare_inputs(hidden, weight, targets, target_weight, shared_head=False):  # noqa: C901
    """Validate and flatten the inputs; cast them for autocast.

    Returns ``(flat_hidden, weight, flat_targets, target_weight)``.
    """
    if shared_head and (target_weight is None or target_weight.shape != weight.shape):
        raise ValueError(
            "shared_head needs target_weight (the verifier lm_head) shaped like weight"
        )
    if weight.requires_grad:
        raise ValueError(
            "fused_linear_compound supports frozen draft weights only; "
            "use the 'fused' loss implementation for a trainable lm_head"
        )
    if target_weight is not None and target_weight.requires_grad:
        raise ValueError("target_weight (verifier lm_head) must be frozen")
    # The kernels index targets as contiguous rows; both are no-ops when
    # the inputs already are.
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    flat_targets = targets.reshape(-1, targets.shape[-1]).contiguous()
    device_type = flat_hidden.device.type
    if torch.is_autocast_enabled(device_type):
        # The operands autocast would give lm_head (the same bf16 weight copy);
        # the ops run with autocast off, so forward and a backward recompute
        # see identical logits.
        compute_dtype = torch.get_autocast_dtype(device_type)
        flat_hidden = flat_hidden.to(compute_dtype)
        weight = weight.to(compute_dtype)
        if target_weight is not None:
            flat_targets = flat_targets.to(compute_dtype)
            target_weight = target_weight.to(compute_dtype)
    elif weight.dtype != flat_hidden.dtype:
        raise ValueError(
            f"hidden ({flat_hidden.dtype}) and weight ({weight.dtype}) dtypes "
            "differ; cast them or run under torch.autocast"
        )
    elif target_weight is not None and target_weight.dtype != flat_targets.dtype:
        raise ValueError(
            f"verifier hidden ({flat_targets.dtype}) and target_weight "
            f"({target_weight.dtype}) dtypes differ; cast them or run under "
            "torch.autocast"
        )
    total_rows, vocab = flat_hidden.shape[0], weight.shape[0]
    if target_weight is not None and (
        target_weight.shape[0] != vocab
        or flat_targets.shape != (total_rows, target_weight.shape[1])
    ):
        raise ValueError(
            f"verifier hidden {tuple(targets.shape)} and target_weight "
            f"{tuple(target_weight.shape)} do not match hidden rows {total_rows} "
            f"and vocab {vocab}"
        )
    if target_weight is None and flat_targets.shape != (total_rows, vocab):
        raise ValueError(
            f"targets {tuple(targets.shape)} do not match hidden rows "
            f"{total_rows} and vocab {vocab}"
        )
    if shared_head:
        # One weight serves both projections (the caller asserts the heads
        # are equal); this also avoids a second autocast copy.
        target_weight = weight
    return flat_hidden, weight, flat_targets, target_weight


def fused_linear_compound(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    names: Sequence[str],
    metric_terms: Sequence[str] = (),
    chunk_rows: int | None = None,
    target_weight: torch.Tensor | None = None,
    grad_mode: str = "auto",
    logits_dtype: str = "input",
    rows: torch.Tensor | None = None,
    shared_head: bool = False,
) -> "fused.FusedOutputs":
    """``fused.fused_compound`` without materializing ``hidden @ weight.T``.

    Args:
        hidden: ``[..., H]`` final (post-norm) draft hidden states.
        weight: ``[V, H]`` frozen draft ``lm_head`` weight.
        targets: rows aligned with ``hidden``: ``[..., V]`` target logits, or
            ``[..., H_t]`` post-norm verifier hidden states when
            ``target_weight`` is given.
        names: named losses (keys of ``fused.LOSS_TERMS``) to differentiate.
        metric_terms: primitive terms computed in the same pass, detached.
        chunk_rows: rows per chunk; ``None`` sizes chunks from the budget
            (``SPECULATORS_FUSED_LINEAR_BUDGET_MIB``, default 4096).
        target_weight: ``[V, H_t]`` frozen verifier ``lm_head`` weight; the
            target logits are then computed per chunk, never materialized.
        grad_mode: ``"deferred"``, ``"recompute"`` or ``"auto"`` (deferred
            for at most two differentiated terms). Without a backward pass
            (no grad) only the losses are computed.
        logits_dtype: ``"input"`` (logit tiles in the input dtype, as
            materialized logits), ``"float32"`` (fp32 from the GEMM
            accumulator: more accurate, slower GEMMs) or ``"float16"``
            (bf16 operands cast to fp16, fp16 tiles: 8x finer rounding than
            bf16 at bf16 cost; plus one fp16 copy of each weight per call).
        rows: optional 1-D index of the rows to compute; every other row is
            treated as fully masked (zero loss, zero gradient, ids 0). Use it
            to skip rows that are masked by construction.
        shared_head: ``weight`` and ``target_weight`` are the same head
            (the caller guarantees equal values). With a distribution term,
            draft logits are then ``target + (hidden - verifier_hidden) @ W.T``
            (module docstring): near-fp32 KL / TV gradients with fp16 tiles.
            Requires ``target_weight``.

    Returns:
        ``FusedOutputs`` whose tensors have ``hidden``'s leading shape.
    """
    if grad_mode not in GRAD_MODES:
        raise ValueError(f"grad_mode must be one of {GRAD_MODES}, got {grad_mode!r}")
    if logits_dtype not in LOGITS_DTYPES:
        raise ValueError(
            f"logits_dtype must be one of {LOGITS_DTYPES}, got {logits_dtype!r}"
        )
    flat_hidden, weight, flat_targets, target_weight = _prepare_inputs(
        hidden, weight, targets, target_weight, shared_head
    )
    lead = hidden.shape[:-1]
    total_rows, vocab = flat_hidden.shape[0], weight.shape[0]
    if rows is not None:
        flat_hidden = flat_hidden.index_select(0, rows)
        flat_targets = flat_targets.index_select(0, rows)

    loss_terms = sorted(
        {t for name in names for t in fused.LOSS_TERMS[name]},
        key=lambda t: fused.TERM_BITS[t],
    )
    grad_mask = fused.terms_mask(loss_terms)
    needs_grad = torch.is_grad_enabled() and flat_hidden.requires_grad
    deferred = needs_grad and (
        grad_mode == "deferred"
        or (grad_mode == "auto" and len(loss_terms) <= _MAX_DEFERRED_TERMS)
    )
    fp32_logits = logits_dtype == "float32"
    all_terms = grad_mask | fused.terms_mask(metric_terms)
    # The delta form helps only against a target distribution (CE's gradient,
    # softmax(z_d) - onehot, has no p_t to cancel the rounding) and is free
    # exactly then: any distribution term already reads the target row in
    # the gradient pass, while CE-only would read it just to rebuild z_d.
    # (The recompute backward reads targets only for differentiated ones.)
    dist_reads = grad_mask if needs_grad and not deferred else all_terms
    delta = shared_head and bool(dist_reads & fused.DIST_MASK)
    if chunk_rows is None:
        chunk_rows = chunk_rows_for(
            vocab,
            flat_hidden.dtype,
            flat_hidden.shape[0],
            recompute_targets=target_weight is not None,
            fp32_logits=fp32_logits,
            dz_tiles=len(loss_terms) if deferred else 1,
            shared_head=delta,
        )
    losses, argmax, _, _ = _fused_linear_fwd(
        flat_hidden,
        weight,
        flat_targets,
        target_weight,
        all_terms,
        grad_mask if needs_grad else 0,
        chunk_rows,
        deferred,
        logits_dtype,
        delta,
    )
    if rows is not None:
        losses = losses.new_zeros((losses.shape[0], total_rows)).index_copy(
            1, rows, losses
        )
        argmax = argmax.new_zeros((2, total_rows)).index_copy(1, rows, argmax)
    terms = {t: losses[fused.TERM_SLOTS[t]].view(lead) for t in loss_terms}
    return fused.FusedOutputs(
        losses={name: fused.compose_loss(name, terms) for name in names},
        metrics={
            t: losses[fused.TERM_SLOTS[t]].view(lead).detach() for t in metric_terms
        },
        pred_ids=argmax[0].view(lead),
        target_ids=argmax[1].view(lead),
    )
