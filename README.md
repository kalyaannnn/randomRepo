# AgentRL

Single-GPU rollout/runtime and post-training stack for verifier-based RL: continuous batching, chunked prefill, KV-aware admission, runtime telemetry, supervised bootstrap, GRPO with verifier reward, LoRA adapters, and strict evaluation. Tasks plug in via a small environment/verifier interface; optional BYOD helpers cover common single-turn setups.

**Constraints:** one CUDA GPU (CPU OK for smoke tests). No multi-node or multi-GPU orchestration.

**Design stance:** not aimed at cold-start RL from a base model alone. Sparse verifier signal is usually insufficient without a bootstrap policy that already lands some correct trajectories; the stack assumes supervised warm-start, rollout diagnosis, optional shaped reward during training only when useful, and strict evaluation at the end.

## Components

| Area | Contents |
|------|----------|
| Runtime | Standard and continuous batching, chunked prefill, persistent-KV decode where supported, controller signals, bottleneck tooling |
| Training | TRL-compatible clipped GRPO path, verifier-driven reward, adapter reuse and checkpointing |
| Tasks | Dataset/verifier/environment not tied to one benchmark; full control via `BaseEnvironment` / `BaseVerifier`, or high-level BYOD (`BYODRecord`, `make_single_turn_task`) for single-turn `prompt → response → verify` flows |

Custom tasks extend `BaseEnvironment.reset`, `step`, `state`, and `BaseVerifier.verify(response, env_state)`.

## Install

```bash
pip install -e .
pip install -e ".[benchmark]"
```

Optional:

```bash
pip install -e ".[dev]"
```

## Example

```python
from agentrl import GRPOConfig, GRPOTrainer
from examples.math_env import MathEnvironment, MathVerifier

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    group_size=8,
    batch_size=4,
    max_new_tokens=128,
    steps=100,
    output_dir="./checkpoints",
)

trainer = GRPOTrainer(
    config=config,
    environment=MathEnvironment(split="smoke"),
    verifier=MathVerifier(),
)

trainer.train()
```

## Repository layout

- `agentrl/` — runtime, rollout, training, public APIs
- `examples/` — small runnable entry points
- `notebooks/` — demos (MBPP BYOD code task)
- `docs/` — integration and demo guides
- `tests/` — runtime, trainer, task checks

## V2 Runtime Benchmark

The runtime benchmark now supports a short-horizon task-backed multi-turn stub
in addition to the legacy math workload. This is the intended `v2` finish-line
benchmark: a deterministic tool-use task that stresses uneven decode lengths,
growing transcript context, and scheduler / KV behavior without hardbaking any
agent schema into the library.

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --task tool-use \
  --split easy \
  --steps 5 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --max-episode-steps 4 \
  --output-dir ./systems_benchmark_compare \
  --compare-runtime-modes
```

This compares:

- standard rollout
- legacy continuous batching
- paged-KV continuous batching

and preserves both systems metrics and task reward so the runtime comparison
stays grounded in a real multi-turn workload.


## Out of scope (current open-source shape)

Not a production serving engine, multi-node trainer, hosted task platform, or hardened sandbox. Model-generated code in demos runs in a subprocess for evaluation only. Advanced runtime items (e.g. paged-KV-style work) are tracked separately from the validated v1 runtime story.

## Documentation

- [Bring your own task](docs/bring_your_own_task.md)
- [Open-source demo](docs/open_source_demo.md)
- [MBPP BYOD notebook](notebooks/codeDemo.ipynb)

## Security

- Replay trajectories use `torch.load(..., weights_only=False)` — do not load untrusted trajectory files.
- Execution-based demos are demo-grade isolation, not a security boundary for untrusted code.
