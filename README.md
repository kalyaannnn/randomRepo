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
    environment=MathEnvironment(split="smoke"),
    verifier=MathVerifier(),
)

trainer.train()
```

CLI example:

```bash
python -m examples.train_math --model Qwen/Qwen2.5-1.5B-Instruct --steps 5 --split smoke
```

## Phase 2 Benchmark Harness

The next validation phase uses a real filtered GSM8K subset loaded through the
HuggingFace `datasets` package. AgentRL keeps only shorter GSM8K questions with
integer final answers, and the default `easy` curriculum ranks examples toward
the simplest real problems first so the first benchmark run stays manageable on
a single GPU while still using authentic dataset examples.

Example:

```bash
python -m examples.benchmark_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 10 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --split train \
  --output-dir ./checkpoints_gsm8k_subset
```

Recommended progression:

- use `smoke` to verify your runtime and reward pipeline
- use `easy` and `train` to check that a small model gets non-degenerate rewards
- use `examples.benchmark_gsm8k_subset` as the first realistic benchmark harness

Install the extra dependency before running the GSM8K benchmark:

```bash
pip install datasets
```

If the standard GSM8K slice collapses to all-zero rewards at initialization,
start with `--curriculum easy` first. It still uses real GSM8K examples, but it
prefers shorter and simpler questions before expanding to the broader filtered
subset.

## Canonical Smoke Config

This is the validated first Colab smoke configuration for small models:

```python
from agentrl import GRPOConfig, GRPOTrainer
from examples.math_env import MathEnvironment, MathVerifier

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    steps=3,
    batch_size=1,
    group_size=4,
    max_new_tokens=32,
    output_dir="./checkpoints",
    dtype="float16",
    sdpa_backend="auto",
    use_continuous_batching=False,
    use_gradient_checkpointing=False,
    do_sample=True,
    temperature=0.8,
)

trainer = GRPOTrainer(
    config=config,
    environment=MathEnvironment(split="smoke"),
    verifier=MathVerifier(),
)

trainer.train()
```

## Validated Smoke Run

AgentRL was smoke-tested end to end on Colab with `Qwen/Qwen2.5-0.5B-Instruct`
on the bundled `smoke` split. In a 3-step run with the config above:

- step 0: `mean_reward=0.25`, `reward_std=0.433`
- step 1: `mean_reward=0.75`, `reward_std=0.433`, `policy_loss=-0.5215`
- step 2: `mean_reward=0.50`, `reward_std=0.50`, `policy_loss=-0.8253`
- peak VRAM stayed around `2.5 GB`
- generation dominated runtime at roughly `15 tokens/sec`

This is a smoke result, not a benchmark. The goal was to verify non-degenerate
group rewards, replay artifacts, and real-model GRPO updates on a single GPU.

## Validated Train-Split Run

The same `Qwen/Qwen2.5-0.5B-Instruct` setup was also exercised on the bundled
`train` split for 10 steps with:

```python
config = GRPOConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    steps=10,
    batch_size=1,
    group_size=4,
    max_new_tokens=32,
    output_dir="./checkpoints_train",
    dtype="float16",
    sdpa_backend="auto",
    use_continuous_batching=False,
    use_gradient_checkpointing=False,
    do_sample=True,
    temperature=0.8,
)
```

Observed behavior from the Colab run:

- non-degenerate rewards on most steps
- `mean_reward` commonly between `0.5` and `0.75`
- stable peak VRAM around `2.35 GB`
- generation remained the dominant runtime cost
- later steps reached all-correct groups (`mean_reward=1.0`, `reward_std=0.0`)

This is still a prototype validation result, not a benchmark claim, but it
shows that the harder synthetic `train` split is viable on a small model.

## Synthetic Ladder

- `smoke`: ultra-easy addition-only problems for the first non-degenerate reward batch
- `easy`: slightly harder synthetic arithmetic with small subtraction and 3-term expressions
- `train` / `eval`: stricter general arithmetic splits that are harder for small models
- `gsm8k_subset`: filtered real GSM8K examples for the first benchmark-style run

## Single-GPU Playbook

- Start with `dtype="float16"` on CUDA and `batch_size * group_size` as the main VRAM dial.
- Continuous batching helps when responses finish at uneven lengths; it matters less when every rollout is nearly the same length.
- Environments used with `group_size > 1` must be `deepcopy`-safe after `reset()`.
- `chunk_size` controls continuous-batching sub-batches when active sequence count grows too large.
- `pad_to_multiple_of` trades a bit of extra padding for more regular tensor shapes.
- If you want the next rung after `smoke`, use `MathEnvironment(split="easy")` before jumping to GSM8K-style tasks.

## Current Constraints

- AgentRL currently requires `use_lora=True`; full-model GRPO is not wired yet.
- Gradient checkpointing is opt-in. Enable it only when VRAM pressure requires it.
- `prefill_chunk_size` is implemented at the mixin level, but the live generation paths are still converging on one long-prompt strategy.
- The framework is single-device only.

## Security Note

- Replay trajectories are loaded with `torch.load(..., weights_only=False)`. Do not load untrusted trajectory files.
