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
a single GPU while still using authentic dataset examples. The GSM8K verifier is
strict binary only: the last non-empty line must be exactly
`Final answer: <integer>` and the integer must match the gold answer.

The current GSM8K path is:

1. bootstrap a LoRA adapter with rationale-plus-final-answer supervised targets
2. run strict pass@k diagnostics on the bootstrap adapter
3. continue to strict-binary GRPO only if pass@k is nonzero
4. evaluate the post-RL adapter separately with strict exact-match

The exact prompt format emitted by `GSM8KSubsetEnvironment` is:

```text
Solve the following GSM8K math word problem.
Reply with exactly one line and nothing else:
Final answer: <integer>

Problem: {question}
```

During rollout, AgentRL wraps that inside its canonical transcript prompt:

```text
Observation:
Solve the following GSM8K math word problem.
Reply with exactly one line and nothing else:
Final answer: <integer>

Problem: {question}

Assistant:
```

The main next model is `Qwen/Qwen2.5-1.5B-Instruct`. The scripts stay fully
configurable through `--model`, so the same workflow can be reused later with
`Qwen/Qwen2.5-3B-Instruct` without code changes.

Example:

```bash
python -m examples.bootstrap_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3 \
  --batch-size 4 \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --adapter-dir ./bootstrap_gsm8k_adapter_15b
```

Then run a strict diagnostic pass@k check on the bootstrap adapter:

```bash
python -m examples.eval_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --init-adapter-path ./bootstrap_gsm8k_adapter_15b \
  --output-dir ./eval_gsm8k_subset_15b_diag \
  --split train \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --max-new-tokens 96 \
  --num-samples 8
```

Only if the diagnostic run shows nonzero pass@k, launch strict-binary GRPO from
that saved adapter:

```bash
python -m examples.benchmark_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 10 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 96 \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --reward-mode strict \
  --init-adapter-path ./bootstrap_gsm8k_adapter_15b \
  --split train \
  --output-dir ./checkpoints_gsm8k_subset_15b
```

Strict evaluation is a separate step:

```bash
python -m examples.eval_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --init-adapter-path ./checkpoints_gsm8k_subset_15b/checkpoint_final \
  --output-dir ./eval_gsm8k_subset_15b_final \
  --split train \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --max-new-tokens 96
```

The diagnostic evaluator reports:

- `pass_at_1`
- `pass_at_k`
- `fraction_with_any_correct`
- `mean_reward`
- `mean_response_tokens`
- `fraction_hitting_max_new_tokens`

It also writes `predictions.jsonl` with raw sampled responses, parsed terminal
predictions, per-sample rewards, and per-sample token lengths for inspection.

The rollout settings used in the working GSM8K path are:

- `group_size=4`
- `batch_size=1`
- `temperature=0.8`
- `do_sample=True`
- `top_p=0.95`
- `max_new_tokens=96` is the recommended starting budget for GSM8K

Recommended progression:

- use `smoke` to verify your runtime and reward pipeline
- use `examples.bootstrap_gsm8k_subset` to warm-start a real GSM8K subset
- use `examples.eval_gsm8k_subset --num-samples 8` to check whether the bootstrap adapter produces any correct strict trajectories
- use `examples.benchmark_gsm8k_subset --reward-mode strict` only if bootstrap pass@k is nonzero

Install the extra dependency before running the GSM8K benchmark:

```bash
pip install datasets
```

Interpretation:

- if `pass_at_8` is near zero, RL still has no signal and the next step is a stronger model such as `Qwen/Qwen2.5-3B-Instruct` or an easier bridge task
- if `pass_at_8` is nontrivial but `pass_at_1` is low, RL may still be useful
- if `fraction_hitting_max_new_tokens` stays high, increase the decoding budget or improve stopping before blaming RL

## Real GSM8K Status

The current hypothesis for real GSM8K is:

- the strict binary verifier is behaving correctly
- the remaining failure mode is dominated by truncation plus model capability floor
- `Qwen/Qwen2.5-1.5B-Instruct` is the next model to test before considering `Qwen/Qwen2.5-3B-Instruct`

This means the next decision point is whether the bootstrap adapter on `1.5B`
has nontrivial pass@k. If it does not, more strict RL is unlikely to help.

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
- `prefill_chunk_size` now affects both standard rollout and continuous batching for long prompts.
- Continuous batching now keeps persistent KV caches for cache-capable models instead of replaying full histories each decode step.
- Trainer startup now reports parameter VRAM plus device headroom when CUDA is available.
- GRPO now saves periodic adapter checkpoints and a final adapter alias under `output_dir/`.
- If you want the next rung after `smoke`, use `MathEnvironment(split="easy")` before jumping to GSM8K-style tasks.

## Current Constraints

- AgentRL currently requires `use_lora=True`; full-model GRPO is not wired yet.
- Gradient checkpointing is opt-in. Enable it only when VRAM pressure requires it.
- `top_p` sampling is not exposed yet.
- Post-GRPO strict evaluation is possible, but the strongest comparison path still needs explicit eval runs against saved GRPO adapters.
- Continuous batching now has persistent KV decode, but it is not yet a paged-KV / vLLM-style runtime.
- The framework is single-device only.

## Security Note

- Replay trajectories are loaded with `torch.load(..., weights_only=False)`. Do not load untrusted trajectory files.
