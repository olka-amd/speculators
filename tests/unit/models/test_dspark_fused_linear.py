"""DSpark with loss_implementation="fused_linear": one real training step.

The Markov bias ``markov_w2(u)`` is added inside the loss's GEMMs and the
trainable ``markov_w2`` gets its gradient from the reduced op's forward. A
tiny DSpark model is run through the materialized fused path and through
fused_linear, configured by ``get_trainer_kwargs`` as the trainer does; the
loss, every metric and every trainable gradient must agree.
"""

from typing import Literal

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from tests.unit.models.test_dflash_fused_linear import (
    BLOCK,
    HIDDEN,
    MAX_ANCHORS,
    VOCAB,
    _inputs,
    requires_gpu,
)

RANK = 16


def _model(
    head_type: Literal["vanilla", "gated", "rnn"], dtype: torch.dtype, shared: bool
) -> DSparkDraftModel:
    tl_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=VOCAB,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DSparkSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        markov_rank=RANK,
        markov_head_type=head_type,
    )
    torch.manual_seed(0)
    model = DSparkDraftModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight, std=0.3)
        noise = 0.0 if shared else 0.1
        model.verifier_lm_head.weight.copy_(
            model.lm_head.weight + noise * torch.randn_like(model.lm_head.weight)
        )
        assert model.markov_head is not None
        torch.nn.init.normal_(model.markov_head.markov_w1.weight, std=0.5)
        torch.nn.init.normal_(model.markov_head.markov_w2.weight, std=0.3)
        torch.nn.init.ones_(model.verifier_norm.weight)
    model = model.to(device="cuda", dtype=dtype).train()  # type: ignore[call-arg]
    model.refresh_shared_lm_heads()
    assert model._shared_lm_heads == shared
    return model


def _step(model, inputs, spec, implementation, weighting, **extra):
    train_kwargs, _ = DSparkDraftModel.get_trainer_kwargs(
        loss_fn=spec,
        loss_implementation=implementation,
        per_position_loss_weight=weighting,
        max_anchors=MAX_ANCHORS,
        **extra,
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(2)  # anchor sampling
    _, loss, metrics = model(**inputs, **train_kwargs)
    loss.backward()
    grads = {
        name: p.grad.detach().float().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach().float(), metrics, grads


def _compare(model, inputs, spec, weighting, rtol, grad_rtol, **extra):
    loss, metrics, grads = _step(model, inputs, spec, "fused", weighting)
    loss_fl, metrics_fl, grads_fl = _step(
        model, inputs, spec, "fused_linear", weighting, **extra
    )
    torch.testing.assert_close(loss_fl, loss, rtol=rtol, atol=1e-6)
    assert metrics_fl.keys() == metrics.keys()
    for key, value in metrics.items():
        if "acc" in key and key != "accept_rate_sum" or key.startswith("eal"):
            torch.testing.assert_close(metrics_fl[key], value, msg=key)
        else:
            torch.testing.assert_close(
                metrics_fl[key].float(), value.float(), rtol=rtol, atol=1e-5, msg=key
            )
    assert grads_fl.keys() == grads.keys()
    assert "markov_head.markov_w2.weight" in grads
    assert "lm_head.weight" not in grads
    for name, grad in grads.items():
        atol = grad_rtol * float(grad.abs().max()) + 1e-12
        torch.testing.assert_close(
            grads_fl[name], grad, rtol=grad_rtol, atol=atol, msg=name
        )


CASES = [
    ("kl_div", "fixed-exp-decay"),
    ('{"kl_div": 0.5, "tv": 0.5}', "fixed-exp-decay"),
    ("ce", "dpace"),
    ("nla", "fixed-exp-decay"),
]


@requires_gpu
@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
@pytest.mark.parametrize(("spec", "weighting"), CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("targets", ["recompute", "materialize"])
@pytest.mark.parametrize("shared", [False, True], ids=["heads", "shared"])
def test_training_step_matches_fused(
    head_type, spec, weighting, targets, shared, monkeypatch
):
    # 1 MiB budget -> chunks of 64 rows: the 128 loss rows span two chunks
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    model = _model(head_type, torch.float32, shared)
    with torch.compiler.set_stance("force_eager"):
        _compare(
            model,
            _inputs(torch.float32),
            spec,
            weighting,
            1e-5,
            1e-3,
            fused_linear_targets=targets,
            fused_linear_logits="input",
        )


@requires_gpu
@pytest.mark.parametrize("head_type", ["vanilla", "rnn"])
def test_training_step_compiled(head_type, monkeypatch):
    """DSpark.forward is torch.compile'd on GPU; the op traces as one node."""
    monkeypatch.setenv("SPECULATORS_FUSED_LINEAR_BUDGET_MIB", "1")
    torch._dynamo.reset()
    model = _model(head_type, torch.float32, True)
    _compare(
        model,
        _inputs(torch.float32),
        '{"kl_div": 0.5, "tv": 0.5}',
        "fixed-exp-decay",
        1e-5,
        1e-3,
        fused_linear_logits="input",
    )


@requires_gpu
@pytest.mark.parametrize("params", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("logits", ["input", "float16", "float32"])
def test_training_step_under_autocast(params, logits):
    """As the trainer runs it: autocast(bf16) with fp32 (DDP) or bf16 (FSDP)
    parameters; gradients align with the materialized bf16 path."""
    model = _model("gated", params, True)
    inputs = _inputs(torch.bfloat16)
    results = {}
    for implementation in ("fused", "fused_linear"):
        with (
            torch.compiler.set_stance("force_eager"),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            results[implementation] = _step(
                model,
                inputs,
                "kl_div",
                implementation,
                "fixed-exp-decay",
                fused_linear_logits=logits,
            )
    loss, _, grads = results["fused"]
    loss_fl, _, grads_fl = results["fused_linear"]
    torch.testing.assert_close(loss_fl, loss, rtol=2e-2, atol=1e-4)
    assert grads_fl.keys() == grads.keys()
    for name, grad in grads.items():
        assert torch.isfinite(grads_fl[name]).all(), name
        cosine = torch.nn.functional.cosine_similarity(
            grads_fl[name].flatten(), grad.flatten(), dim=0
        )
        assert cosine > 0.99, (name, float(cosine))
