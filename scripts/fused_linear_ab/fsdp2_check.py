"""FSDP2 check for fused_linear (design v3, PR 7 gate).

For a tiny DFlash and DSpark model sharded with ``apply_fully_sharded`` (as the
trainer does), one training step with ``loss_implementation="fused_linear"``
must match the materialized ``fused`` path: same loss, same gradient for every
trainable parameter after reduce-scatter (``markov_w2`` included). It also
checks that the shared-head delta form is taken under FSDP2, i.e. that the
unsharded ``verifier_lm_head.weight`` is one tensor for both arguments.

Run on 2 GPUs:
    torchrun --nproc_per_node 2 scripts/fused_linear_ab/fsdp2_check.py
    torchrun --nproc_per_node 2 scripts/fused_linear_ab/fsdp2_check.py --replicate-frozen
    torchrun --nproc_per_node 2 scripts/fused_linear_ab/fsdp2_check.py --compile
Prints PASS / FAIL per model on rank 0; exit code 1 on any failure.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import fused_linear, fused_linear_reduced
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.train.distributed import apply_fully_sharded

HIDDEN, VOCAB, BLOCK, SEQ, ANCHORS = 256, 4096, 8, 1024, 64
MODELS = {
    "dflash": (DFlashDraftModel, DFlashSpeculatorConfig, {}),
    "dspark": (DSparkDraftModel, DSparkSpeculatorConfig, {"markov_rank": 16}),
}

# (wrapper, shared head?) per fused_linear call, recorded by _record
CALLS: list[tuple[str, bool]] = []


def _record(module, name):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        CALLS.append((name, kwargs.get("target_weight") is args[1]))
        return original(*args, **kwargs)

    setattr(module, name, wrapper)


def build(kind):
    model_cls, config_cls, extra = MODELS[kind]
    tl_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=VOCAB,
    )
    config = config_cls(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=BLOCK,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        **extra,
    )
    torch.manual_seed(0)
    model = model_cls(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
        torch.nn.init.normal_(model.lm_head.weight, std=0.3)
        # full vocab: lm_head is the verifier head, as after load_verifier_weights
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)
        torch.nn.init.ones_(model.verifier_norm.weight)
    return model.cuda().train()


def inputs():
    gen = torch.Generator(device="cuda").manual_seed(1)
    bf16 = torch.bfloat16
    return {
        "hidden_states": torch.randn(
            1, SEQ, 2 * HIDDEN, device="cuda", generator=gen
        ).to(bf16),
        "input_ids": torch.randint(0, VOCAB, (1, SEQ), device="cuda", generator=gen),
        "loss_mask": torch.ones(1, SEQ, device="cuda"),
        "verifier_last_hidden_states": torch.randn(
            1, SEQ, HIDDEN, device="cuda", generator=gen
        ).to(bf16),
        "document_ids": torch.zeros(1, SEQ, device="cuda", dtype=torch.long),
    }


def step(kind, implementation, args):
    model = build(kind)
    apply_fully_sharded(model, replicate_frozen=args.replicate_frozen)
    kwargs, _ = MODELS[kind][0].get_trainer_kwargs(
        loss_fn="kl_div", loss_implementation=implementation, max_anchors=ANCHORS
    )
    torch.manual_seed(2)  # anchor sampling, identical on every rank
    with torch.compiler.set_stance("default" if args.compile else "force_eager"):
        _, loss, _ = model(**inputs(), **kwargs)
    loss.backward()
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grads[name] = (
                grad.full_tensor() if hasattr(grad, "full_tensor") else grad
            ).float()
    return loss.detach().float(), grads


def check(kind, args):
    failures = []
    loss, grads = step(kind, "fused", args)
    CALLS.clear()
    loss_fl, grads_fl = step(kind, "fused_linear", args)
    if not CALLS:
        failures.append("fused_linear was not called")
    elif not all(shared for _, shared in CALLS):
        failures.append(f"shared head not detected under FSDP2: {CALLS}")
    if abs(float(loss_fl - loss)) > 1e-2 * abs(float(loss)) + 1e-4:
        failures.append(f"loss {float(loss_fl):.6f} vs fused {float(loss):.6f}")
    if grads_fl.keys() != grads.keys():
        failures.append(
            f"gradient sets differ: {sorted(grads.keys() ^ grads_fl.keys())}"
        )
    if kind == "dspark" and "markov_head.markov_w2.weight" not in grads_fl:
        failures.append("markov_w2 received no gradient")
    for name in grads.keys() & grads_fl.keys():
        a, b = grads_fl[name].flatten(), grads[name].flatten()
        if not torch.isfinite(a).all():
            failures.append(f"{name}: non-finite gradient")
            continue
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        if cosine < 0.99:
            failures.append(f"{name}: cosine {cosine:.4f} vs fused")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate-frozen", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    _record(fused_linear, "fused_linear_compound")
    _record(fused_linear_reduced, "fused_linear_reduced")
    failed = False
    for kind in MODELS:
        failures = check(kind, args)
        failed |= bool(failures)
        if dist.get_rank() == 0:
            status = "FAIL" if failures else "PASS"
            print(f"{status} {kind} (world {dist.get_world_size()}, {vars(args)})")
            for failure in failures:
                print(f"  {failure}")
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
