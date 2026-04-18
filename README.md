# AgentRL

AgentRL is a single-GPU rollout/runtime and post-training stack for verifier-based RL.

It combines three ideas in one repo:

- **systems** — continuous batching, chunked prefill, KV-aware admission, runtime headroom tracking, and rollout telemetry on one GPU
- **post-training** — supervised bootstrap, verifier-based GRPO, strict evaluation, and adapter lifecycle management
- **task abstraction** — bring your own dataset, verifier, and environment through a small task interface, or use a lightweight high-level BYOD helper for common single-turn tasks

The intended workflow is **not** cold-start RL. For real tasks, AgentRL is designed around:

1. bootstrap a task-specific adapter with supervised data
2. check whether the bootstrap policy already samples some correct trajectories
3. continue with verifier-based RL from that adapter
4. evaluate the saved adapter strictly

**Hardware:** one CUDA GPU (or CPU for small tests). No multi-node or multi-GPU orchestration.

## What It Is

AgentRL is best understood as three connected layers:

1. **Systems proof** — a research-grade single-GPU rollout runtime with standard and continuous batching, chunked prefill, persistent-KV decode where supported, runtime controller signals, and bottleneck diagnosis.
2. **Post-training proof** — a reference GRPO stack with supervised bootstrap, verifier-driven reward, LoRA adapter reuse, checkpointing, and strict final evaluation.
3. **Task abstraction proof** — tasks are not hard-wired to one benchmark; users can bring their own dataset, verifier, and environment.

## Workflow

The supported methodology is:

- **bootstrap first**
- **diagnose before RL**
- **use shaped reward during RL only when needed**
- **keep final evaluation strict**

This matters because sparse verifier reward is often too weak for cold-start RL on real tasks. AgentRL is built around a practical post-training workflow rather than the assumption that strict RL will work from scratch.

## Quick Start

Install:

```bash
pip install -e .
pip install -e ".[benchmark]"
```

Optional development extras:

```bash
pip install -e ".[dev]"
```

Minimal GRPO example:

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

## Task Integration

AgentRL supports two task integration paths.

### High-level BYOD path

Use this when the task is a common single-turn `prompt -> response -> verifier` setup.

- bring your dataset or in-memory records
- represent records with `BYODRecord`
- use `make_single_turn_task(...)`
- provide:
  - prompt formatting
  - verifier / reward logic
  - optional supervised targets for bootstrap

Start here:

- [Bring your own task guide](docs/bring_your_own_task.md)
- [MBPP BYOD notebook](notebooks/codeDemo.ipynb)

### Low-level custom task path

Use this when you need more control, especially for:

- custom environment state
- task-specific step transitions
- multi-turn tasks
- richer verifier logic

Implement:

- `BaseEnvironment.reset()`
- `BaseEnvironment.step(action)`
- `BaseEnvironment.state()`
- `BaseVerifier.verify(response, env_state)`

This is the primary extension point for custom environments and custom verifiers.

## Demo Path

The featured public demo is the MBPP BYOD notebook:

- [notebooks/codeDemo.ipynb](notebooks/codeDemo.ipynb)

It demonstrates:

- a real external code dataset
- a custom execution-based verifier
- shaped reward during RL
- strict functional final evaluation
- standard vs continuous batching comparison on the same workload

## Validation Snapshot

On the MBPP BYOD code-task demo with a real execution-based verifier:

- continuous batching was **1.54x faster per step**
- continuous batching delivered **1.55x higher throughput**
- continuous batching used **2842 MB less rollout VRAM**
- continuous batching achieved **+0.158 mean reward** over standard rollout in that small run

These are project validation results, not broad benchmark or SOTA claims.

## Repository Structure

- `agentrl/` — core runtime, rollout, training, and public APIs
- `examples/` — runnable scripts and small example entry points
- `notebooks/` — public demo notebooks
- `docs/` — focused guides and walkthroughs
- `tests/` — automated validation of runtime, trainer, and task behavior

## What AgentRL Is Not

AgentRL is not, in its current open-source shape:

- a production serving engine
- a multi-node training stack
- a hosted task platform
- a hardened sandbox for executing untrusted code

The current validated story is a research-grade single-GPU rollout/runtime and post-training stack. More advanced runtime-engine work, including paged-KV-style improvements, belongs to the v2 track rather than the validated v1 story.

## Documentation

- [Bring your own task guide](docs/bring_your_own_task.md)
- [Open-source demo guide](docs/open_source_demo.md)
- [MBPP BYOD notebook](notebooks/codeDemo.ipynb)

## Security Note

- Replay trajectories are loaded with `torch.load(..., weights_only=False)`. Do not load untrusted trajectory files.
- Execution-based code-task demos may run model-generated Python in a subprocess for evaluation. This is a demo-grade setup, not a hardened security boundary.
