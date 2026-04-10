# AgentRL

AgentRL is a lightweight single-GPU framework for GRPO-style post-training with
pluggable text environments and deterministic verifiers.

This repository is being built in the implementation order defined by the
project prompt. The current checkpoint includes the public task-side contracts
and trainer configuration surface.

## Minimal Usage

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
    environment=MathEnvironment(split="train"),
    verifier=MathVerifier(),
)

trainer.train()
```

CLI example:

```bash
python -m examples.train_math --model Qwen/Qwen2.5-1.5B-Instruct --steps 5
```

## Single-GPU Playbook

- Start with `dtype="float16"` on CUDA and `batch_size * group_size` as the main VRAM dial.
- Continuous batching helps when responses finish at uneven lengths; it matters less when every rollout is nearly the same length.
- Environments used with `group_size > 1` must be `deepcopy`-safe after `reset()`.
- `chunk_size` controls continuous-batching sub-batches when active sequence count grows too large.
- `pad_to_multiple_of` trades a bit of extra padding for more regular tensor shapes.

## Current Constraints

- AgentRL currently requires `use_lora=True`; full-model GRPO is not wired yet.
- Gradient checkpointing is opt-in. Enable it only when VRAM pressure requires it.
- `prefill_chunk_size` is implemented at the mixin level, but the live generation paths are still converging on one long-prompt strategy.
- The framework is single-device only.

## Security Note

- Replay trajectories are loaded with `torch.load(..., weights_only=False)`. Do not load untrusted trajectory files.
