# Open-Source Demo Guide

This guide packages AgentRL around the three claims the project should make publicly:

1. it shows **real single-GPU rollout/runtime understanding**
2. it shows a **coherent post-training RL workflow**
3. it shows a **task abstraction** that can extend beyond the bundled examples

The target is not frontier scale. The target is a strong, honest engineering demo with measurable runtime behavior and a correct post-training story.

## What This Demo Should Prove

AgentRL should read as:

- a research-grade single-GPU rollout runtime with clear bottleneck diagnosis
- a verifier-based post-training stack with mandatory SFT bootstrap for serious tasks
- a task abstraction that can support both bundled examples and external backends

It should **not** read as:

- a distributed RL platform
- a production serving engine
- a paged-KV or vLLM-class inference system
- a cold-start RL benchmark that happens to reward formatting tricks

## Stable Demo Setup

Use one stable hardware context and one model family throughout the demo.

Recommended public framing:

- one CUDA GPU
- one instruct-tuned model family
- one systems comparison workload
- one primary end-to-end task lifecycle
- one secondary task portability example under the same abstraction

If using a larger GPU such as an A100, use the extra headroom to keep the demo stable and interpretable, not to make broader claims than the repo supports.

## Part 1: Systems Proof

Start with a controlled runtime comparison on the same workload, same model, same decode budget, same `batch_size`, and same `group_size`.

Run:

```bash
python -m examples.benchmark_systems \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 5 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 64 \
  --split easy \
  --output-dir ./systems_benchmark_compare \
  --compare-standard-vs-continuous
```

This produces:

- `./systems_benchmark_compare/standard/summary.json`
- `./systems_benchmark_compare/continuous/summary.json`
- `./systems_benchmark_compare/comparison.json`

Preserve these metrics in the writeup:

- `mean_step_time_ms`
- `mean_tokens_per_second`
- `mean_padding_ratio`
- `mean_cache_reuse_effectiveness`
- `mean_scheduler_kv_pressure`
- `mean_scheduler_deferred_sequences`
- `mean_scheduler_max_concurrent_sequences`
- `peak_vram_mb`
- `min_rollout_runtime_headroom_mb`
- `dominant_runtime_bottleneck`
- `efficiency_diagnosis`
- `comparison_verdict`

Interpret them explicitly:

- say what bottleneck dominated
- say why continuous batching did or did not help
- say whether the workload was decode-limited, padding-limited, or KV-budget-limited
- say that the runtime is a research-grade single-GPU rollout engine, not a production serving stack

Why this matters for RL:

- rollout cost sets practical training speed
- padding waste reduces usable throughput
- KV pressure and headroom constrain stable concurrency
- telemetry tells you whether to optimize decode, prefill, or admission behavior next

## Part 2: Post-Training Proof

Use one primary task to show the intended lifecycle:

1. SFT bootstrap
2. diagnostic evaluation from the bootstrap adapter
3. GRPO initialized from the bootstrap adapter
4. final evaluation from the saved RL adapter

The project should **not** be demoed as cold-start RL. For serious tasks, SFT bootstrap is mandatory because the verifier only becomes useful once the policy can already sample some correct trajectories.

### Primary Task: GSM8K-Style Benchmark

Bootstrap:

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

Diagnostic evaluation:

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

GRPO from the bootstrap adapter:

```bash
python -m examples.benchmark_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 10 \
  --batch-size 1 \
  --group-size 4 \
  --max-new-tokens 128 \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --reward-mode strict \
  --init-adapter-path ./bootstrap_gsm8k_adapter_15b \
  --split train \
  --output-dir ./checkpoints_gsm8k_subset_15b_rl
```

Final strict eval:

```bash
python -m examples.eval_gsm8k_subset \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --init-adapter-path ./checkpoints_gsm8k_subset_15b_rl/checkpoint_final \
  --output-dir ./eval_gsm8k_subset_15b_rl_final \
  --split train \
  --subset-size 128 \
  --max-question-words 45 \
  --curriculum easy \
  --max-new-tokens 128
```

### What To Preserve

Artifacts:

- bootstrap adapter path
- diagnostic `summary.json`
- diagnostic `predictions.jsonl`
- RL `metrics.jsonl`
- RL checkpoint directory
- final eval `summary.json`
- final eval `predictions.jsonl`

Metrics:

- pre-RL `pass@1`
- pre-RL `pass@k`
- fraction of prompts with any correct sample
- post-RL strict evaluation metric
- runtime telemetry from the RL run

Interpretation:

- explain that SFT bootstrap is mandatory because cold RL was not the intended workflow
- explain that the strict verifier rewards exact terminal correctness, not formatting alone
- explain whether RL improved selection/sharpening from an already-competent bootstrap policy
- do not claim broad SOTA behavior from the filtered subset benchmark

## Part 3: Task Abstraction Proof

The public abstraction claim should be:

> Any task can plug in if it provides supervised bootstrap samples, an environment contract, and a deterministic verifier.

The repo’s concrete task interface is the pair of contracts in [agentrl/core/base.py](agentrl/core/base.py:1):

- `BaseEnvironment.reset()`
- `BaseEnvironment.step(action)`
- `BaseEnvironment.state()`
- `BaseVerifier.verify(response, env_state)`

### External Task API Contract

For public demos, use one shared API shape for external tasks:

#### `POST /episodes/reset`

Request:

```json
{
  "task_id": "gsm8k",
  "split": "train",
  "seed": 0
}
```

Response:

```json
{
  "episode_id": "ep_123",
  "observation": "Solve the following problem...",
  "env_state": {
    "answer": 42,
    "task_metadata": {}
  }
}
```

#### `POST /episodes/step`

Request:

```json
{
  "episode_id": "ep_123",
  "action": "Final answer: 42"
}
```

Response:

```json
{
  "observation": "done",
  "done": true,
  "env_state": {
    "answer": 42,
    "task_metadata": {}
  }
}
```

#### `POST /verifier/verify`

Request:

```json
{
  "task_id": "gsm8k",
  "response": "Final answer: 42",
  "env_state": {
    "answer": 42,
    "task_metadata": {}
  }
}
```

Response:

```json
{
  "reward": 1.0,
  "diagnostics": {
    "matched_exact_answer": true
  }
}
```

#### `POST /sft/samples`

Request:

```json
{
  "task_id": "gsm8k",
  "split": "train",
  "limit": 128
}
```

Response:

```json
{
  "samples": [
    {
      "prompt": "Solve the following problem...",
      "target": "Reasoning...\n\nFinal answer: 42"
    }
  ]
}
```

The key contract rule is that `env_state` must be deterministic enough for verification and replay. The backend can be external; the reward contract cannot be fuzzy.

### Secondary Task Example: Coding

The portability story is stronger if a second task uses the same lifecycle:

1. bootstrap on supervised prompt/target pairs
2. evaluate the bootstrap adapter
3. continue with verifier-based RL
4. evaluate the saved RL adapter

For coding, the environment may be multi-turn while GSM8K remains single-turn. That is fine. The abstraction only requires that the task provide:

- supervised bootstrap samples
- `reset`
- `step`
- deterministic `state`
- deterministic `verify`

A coding backend can, for example, expose:

- a prompt describing the problem and current workspace state
- iterative observations after each model action
- remote unit-test-based verification at the end

The important point is not that AgentRL ships that service today. The important point is that the runtime/trainer abstraction can support it cleanly.

## Demo Deliverables

The final public package should include:

- tightened [README.md](README.md)
- this standalone guide
- one systems comparison artifact set
- one end-to-end primary-task artifact set
- clear references to metrics JSONL, benchmark summaries, eval summaries, predictions, and saved adapters/checkpoints

## Acceptance Checklist

The demo is complete when a reader can see all of this without guessing:

- what AgentRL is
- what it is not
- why the systems layer matters
- why SFT bootstrap is mandatory
- how adapters move through the workflow
- how evaluation is performed
- how a new task fits the same abstraction

If any of those points require verbal explanation outside the repo, the public docs are still too implicit.
