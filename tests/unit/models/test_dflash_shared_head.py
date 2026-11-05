"""DFlash fused_linear shared-head mode: ``lm_head == verifier_lm_head``.

DFlash loads both heads from the verifier. The model records whether they
are equal at weight-load time (never per step) and, with recomputed targets,
forms the draft logits as target + delta. The step must match the
materialized fused path, and a model whose heads differ must not use it.
"""

import pytest
import torch

from speculators.models.dflash import core as dflash_core
from speculators.models.dflash.core import DFlashDraftModel
from tests.unit.models.test_dflash_fused_linear import (
    _compare,
    _inputs,
    _model,
    _step,
    requires_gpu,
)


def _share_heads(model):
    with torch.no_grad():
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)
    return model


def _spy(monkeypatch):
    """Record the shared_head flag each fused_linear forward passes."""
    calls = []
    real = dflash_core.compute_metrics_from_hidden

    def spy(*args, **kwargs):
        calls.append(kwargs.get("shared_head"))
        return real(*args, **kwargs)

    monkeypatch.setattr(dflash_core, "compute_metrics_from_hidden", spy)
    return calls


@requires_gpu
def test_flag_follows_weight_loads():
    model = _model(False, torch.float32)
    assert not model._shared_lm_heads  # constructed: not verified
    assert not model.refresh_shared_lm_heads()  # heads differ (see _model)
    _share_heads(model)
    assert not model._shared_lm_heads  # in-place edits are not tracked...
    assert model.refresh_shared_lm_heads()  # ...until refreshed

    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["lm_head.weight"] = state["lm_head.weight"] + 1.0
    model.load_state_dict(state)  # post-hook re-checks
    assert not model._shared_lm_heads
    state["lm_head.weight"] = state["verifier_lm_head.weight"].clone()
    model.load_state_dict(state)
    assert model._shared_lm_heads


def test_flag_false_for_meta_or_mismatched_heads():
    with torch.device("meta"):
        lm = torch.nn.Linear(4, 10, bias=False)
        ver = torch.nn.Linear(4, 10, bias=False)
    holder = torch.nn.Module()
    holder.lm_head, holder.verifier_lm_head = lm, ver
    assert not DFlashDraftModel.refresh_shared_lm_heads(holder)  # type: ignore[arg-type]
    holder.lm_head = torch.nn.Linear(4, 10, bias=False)
    holder.verifier_lm_head = torch.nn.Linear(4, 12, bias=False)
    assert not DFlashDraftModel.refresh_shared_lm_heads(holder)  # type: ignore[arg-type]


@requires_gpu
@pytest.mark.parametrize("spec", ["kl_div", '{"kl_div": 0.5, "tv": 0.5}', "ce"])
@pytest.mark.parametrize("weighting", ["fixed-exp-decay", "dpace"])
def test_shared_head_step_matches_fused(spec, weighting, monkeypatch):
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    model = _share_heads(_model(False, torch.float32))
    assert model.refresh_shared_lm_heads()
    calls = _spy(monkeypatch)
    with torch.compiler.set_stance("force_eager"):
        _compare(model, _inputs(torch.float32), spec, weighting, 1e-5, 1e-3)
    assert calls == [True]


@requires_gpu
def test_shared_head_off_and_unequal_heads(monkeypatch):
    calls = _spy(monkeypatch)
    model = _share_heads(_model(False, torch.float32))
    model.refresh_shared_lm_heads()
    inputs = _inputs(torch.float32)
    with torch.compiler.set_stance("force_eager"):
        train_kwargs, _ = DFlashDraftModel.get_trainer_kwargs(
            loss_fn="kl_div",
            loss_implementation="fused_linear",
            max_anchors=32,
            fused_linear_shared_head="off",
        )
        model(**inputs, **train_kwargs)
        _step(_model(False, torch.float32), inputs, "kl_div", "fused_linear", "dpace")
        # materialized targets never use the shared head
        _step(model, inputs, "kl_div", "fused_linear", "dpace", "materialize")
    assert calls == [False, False, False]


@requires_gpu
@pytest.mark.parametrize("logits", ["input", "float16", "float32"])
def test_shared_head_bf16_autocast_compiled(logits):
    """As the trainer runs it (compiled, autocast, fp32 params): finite and
    aligned with the materialized bf16-logit path."""
    torch._dynamo.reset()
    model = _share_heads(_model(False, torch.float32))
    model.refresh_shared_lm_heads()
    inputs = _inputs(torch.bfloat16)
    results = {}
    for implementation in ("fused", "fused_linear"):
        train_kwargs, _ = DFlashDraftModel.get_trainer_kwargs(
            loss_fn='{"kl_div": 0.5, "tv": 0.5}',
            loss_implementation=implementation,
            max_anchors=32,
            fused_linear_logits=logits,
        )
        model.zero_grad(set_to_none=True)
        torch.manual_seed(2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(**inputs, **train_kwargs)
        loss.backward()
        results[implementation] = (
            loss.detach().float(),
            {
                n: p.grad.float()
                for n, p in model.named_parameters()
                if p.grad is not None
            },
        )
    loss, grads = results["fused"]
    loss_fl, grads_fl = results["fused_linear"]
    torch.testing.assert_close(loss_fl, loss, rtol=2e-2, atol=1e-4)
    for name, grad in grads.items():
        assert torch.isfinite(grads_fl[name]).all(), name
        cosine = torch.nn.functional.cosine_similarity(
            grads_fl[name].flatten(), grad.flatten(), dim=0
        )
        assert cosine > 0.99, (name, float(cosine))


@requires_gpu
def test_reload_drops_cached_head_copies(monkeypatch):
    """FSDP2 refills all-gathered parameters without bumping their version
    counter, which the weight-copy cache checks; a weight load must therefore
    drop the cached fp16 / [H, V] copies (seen stale in an FSDP2 probe)."""
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    model = _share_heads(_model(False, torch.bfloat16))
    model.refresh_shared_lm_heads()
    inputs = _inputs(torch.bfloat16)
    spec = '{"kl_div": 0.5, "tv": 0.5}'

    def losses():
        out = {}
        for impl in ("fused", "fused_linear"):
            kwargs, _ = DFlashDraftModel.get_trainer_kwargs(
                loss_fn=spec, loss_implementation=impl, max_anchors=32
            )
            torch.manual_seed(2)
            with torch.no_grad(), torch.compiler.set_stance("force_eager"):
                out[impl] = float(model(**inputs, **kwargs)[1])
        return out

    losses()  # fills the cache
    heads = (model.lm_head.weight, model.verifier_lm_head.weight)
    with torch.no_grad():
        for weight in heads:
            with torch.autograd._unsafe_preserve_version_counter(weight):
                weight.mul_(1.7).add_(0.05)
    model.refresh_shared_lm_heads()  # what the load hooks run
    out = losses()
    torch.testing.assert_close(out["fused_linear"], out["fused"], rtol=2e-2, atol=0)
