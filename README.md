# AgentRL

AgentRL is a **single-GPU rollout/runtime and post-training stack** for verifier-based RL. It is built to show two things together:

- **systems understanding** — grouped generation, scheduling, KV-aware admission, runtime headroom tracking, and rollout telemetry on one GPU
- **post-training workflow** — SFT bootstrap, LoRA adapter reuse, verifier-based **GRPO**, and strict evaluation

The intended benchmark story is **not cold-start RL**. For real tasks, the supported workflow is:

1. bootstrap a task-specific adapter with supervised data
2. evaluate whether the bootstrap policy already samples some correct trajectories
3. continue with verifier-based RL from that adapter
4. evaluate the saved RL adapter strictly

**Hardware:** one CUDA GPU (or CPU for small tests). No multi-node or multi-GPU orchestration.

---

## Project story

AgentRL is best understood as three connected layers:

1. **Systems proof** — a research-grade single-GPU rollout runtime with continuous batching, chunked prefill, persistent-KV decode where supported, runtime controller signals, and measurable bottleneck diagnosis.
2. **Post-training proof** — a reference GRPO stack with mandatory SFT bootstrap for serious tasks, strict verifier-driven reward, and first-class adapter lifecycle.
3. **Task abstraction proof** — the trainer is not tied to one bundled benchmark; tasks plug in through `BaseEnvironment`, `BaseVerifier`, and supervised bootstrap samples.

The public demo shape for this repo should preserve all three. See [docs/open_source_demo.md](docs/open_source_demo.md) for the end-to-end walkthrough and the external task API contract used to explain portability beyond bundled examples.

If you want to bring your own dataset rather than use the bundled examples, start with the official v1 onboarding guide in [docs/bring_your_own_task.md](docs/bring_your_own_task.md). The public BYOD surface stays small: use `BYODRecord` and `make_single_turn_task` for the common single-turn path, and drop to `BaseEnvironment` / `BaseVerifier` only when you need lower-level control.

Two notebooks show the intended demo shape:

- [notebooks/gsm8k_end_to_end.ipynb](notebooks/gsm8k_end_to_end.ipynb)
- [notebooks/byod_onboarding.ipynb](notebooks/byod_onboarding.ipynb)

---

## Rollout engine and systems

The framework optimizes **how** rollouts are executed on one device:

- **Standard and continuous batching** — continuous path drops finished sequences during decode to cut wasted work when lengths differ.
- **Chunked prefill** — long prompts are prefilled in chunks in both rollout paths.
- **Persistent KV decode** — when the Hugging Face model supports cache-style forwards, continuous batching reuses per-sequence KV across decode steps (with a fallback path for models that do not).
- **Execution-policy-aware scheduling** — `safe` / `balanced` / `max_throughput` policies; token and **KV-budgeted admission** for prefill and decode waves.
- **Runtime controller** — startup-style KV estimates vs free VRAM, headroom-based adjustments, OOM-aware retries, bottleneck hints (`decode`, `prefill`, `padding`, `kv_budget`).
- **Observability** — metrics logging, `SystemsProfiler` (per-phase time and VRAM), debugger hooks, trajectory buffer and replay.

This is **not** a paged-KV or vLLM-class production serving engine; it is a **research-grade** single-GPU rollout runtime with first-class telemetry. Optional integration with external engines (for example `experimental_vllm_rollout`) is reserved for a future release and is not implemented yet.

---

## Post-training RL stack (GRPO)

AgentRL ships a full **GRPO** loop as the main consumer of the rollout engine:

- **Pluggable tasks** — implement `BaseEnvironment` (multi-turn via `reset` / `step` / `state`) and `BaseVerifier` (deterministic scalar reward in `[0, 1]`).
- **LoRA-only policy and reference** — shared-weight layout, adapter checkpoints and a final alias.
- **SFT bootstrap** — the intended warm-start path for serious tasks; cold-start RL is not the recommended demo story.
- **Trajectory buffering, replay, checkpointing** — practical training hygiene on one machine.

The **systems benchmark** (`examples.benchmark_systems`) is the quickest way to compare standard vs continuous rollouts on a fixed workload before you spend RL compute.

---

## Installation

```bash
pip install -e .
pip install -e ".[benchmark]"
```

Optional development extras:

```bash
pip install -e ".[dev]"
```

## Bring Your Own Task / Dataset

AgentRL is **not** a dataset platform. The official v1 onboarding path is documented in [docs/bring_your_own_task.md](docs/bring_your_own_task.md), and the lightweight reference implementation remains [examples/byod_task.py](examples/byod_task.py:1).

For most single-turn tasks, the intended public API is `BYODRecord` plus `make_single_turn_task(...)`. Use `BaseEnvironment` and `BaseVerifier` directly only when the task genuinely needs custom multi-turn state or another lower-level contract.

## Minimal usage (GRPO)

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
python -m examples.train_math --model Qwen/Qwen2.5-1.5B-Instruct --steps 5 \
  --split smoke
```

## Systems benchmark (quick demo)

Use this when you want to compare rollout implementations, runtime bottlenecks, VRAM pressure, cache reuse, and scheduler KV pressure on a fixed workload.

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

This writes:

- `./systems_benchmark_compare/standard/summary.json`
- `./systems_benchmark_compare/continuous/summary.json`
- `./systems_benchmark_compare/comparison.json`

The main signals to preserve in a demo writeup are:

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

One measured Colab T4 run on the `easy` math workload produced:

- Standard rollout: `22.47s` mean step time, `11.28` tokens/s, diagnosis `decode-limited`, bottleneck `decode_without_cache_reuse`
- Continuous batching: `8.71s` mean step time, `29.20` tokens/s, diagnosis `decode-limited`, bottleneck `decode`

Continuous batching was about **2.6×** faster on mean step time with near-zero padding waste in both modes. The benchmark is intentionally **systems-focused**: the same run can still show zero reward under a strict verifier while exposing decode as the dominant cost and the benefit of cache reuse.

In other words, the benchmark is useful even when task reward is uninteresting: it isolates rollout behavior, shows where time went, and makes clear whether the run was decode-limited, padding-limited, or KV-budget-limited.

Recent runtime instrumentation also reports:

- Scheduler prefill and decode passes
- Deferred sequences and peak concurrent active sequences
- Scheduler KV-budget estimates, admitted KV load, and normalized KV pressure
- `kv_budget` as a distinct runtime bottleneck when admission runs close to the configured KV-cache budget

---

## Current project status

### Implemented

- Rollout engine: standard and continuous batching, chunked prefill, persistent KV decode where supported, execution-policy-aware scheduler with token- and KV-aware admission
- Runtime controller, startup VRAM / headroom reporting, trajectory replay, logging, debugger, profiler, buffer eviction
- Core GRPO loop end to end; shared-weight LoRA policy/reference layout; adapter checkpoints and final alias
- Synthetic task ladder; real filtered GSM8K subset path; rationale-based SFT bootstrap; strict binary verifier and pass@k diagnostics
- Test suite is green

### Validated experimentally

- Single-GPU Colab smoke runs on real Hugging Face models
- T4 comparison above (continuous vs standard on the same workload)
- Synthetic tasks with non-degenerate reward; multi-step GRPO updates are real training updates, not plumbing-only
- Filtered GSM8K subset: bootstrap, strict verification, diagnostics, GRPO from a bootstrap adapter
- Persistent-KV continuous batching path is unit tested

### Still in progress

- Runtime is not a paged-KV or vLLM-style production decode engine
- Scheduling is more memory-aware than early prototypes but simpler than full serving stacks
- Strongest benchmark story remains explicit bootstrap-vs-post-RL evaluation on saved adapters
- GSM8K training quality depends on bootstrap strength, decoding budget, and subset difficulty

---

## Recommended demo shape

If you are showing the project publicly, the cleanest story is:

1. **Systems proof** — compare standard vs continuous rollouts on the same workload with the same model and decode budget.
2. **Post-training proof** — run `bootstrap -> diagnostic eval -> GRPO -> final eval` on one primary task.
3. **Task abstraction proof** — show that the same lifecycle can apply to an external task backend that provides supervised bootstrap data, an environment contract, and a deterministic verifier.

The standalone walkthrough in [docs/open_source_demo.md](docs/open_source_demo.md) is written around that structure.

---

## GSM8K workflow

The current GSM8K path is:

1. Bootstrap a LoRA adapter with rationale-based supervised targets derived from GSM8K solutions.
2. Run diagnostic evaluation to measure `pass@1` and `pass@k` before RL.
3. Run GRPO with a strict binary verifier only if the bootstrap model already produces some correct trajectories.
4. Evaluate the saved GRPO adapter with strict exact match.

This is the intended benchmark workflow because **cold-start strict RL on GSM8K was not effective**. The bootstrap adapter is the prerequisite that turns the verifier into a useful training signal.

The GSM8K verifier is intentionally strict:

- reward is `1.0` only if the last non-empty line is exactly `Final answer: <integer>` and the integer matches the gold answer
- reward is `0.0` otherwise
- there is no partial reward for formatting, near misses, or fallback integer extraction

Bootstrap is rationale-based rather than answer-only. The diagnostic stage answers: does the bootstrap adapter produce any correct strict trajectories at all?

## Recommended GSM8K commands

Bootstrap SFT:

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

Diagnostic evaluation before RL:

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

GRPO training:

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

This run writes runtime telemetry to `./checkpoints_gsm8k_subset_15b_rl/metrics.jsonl` alongside checkpoints and replay artifacts. The RL run should be interpreted together with the pre-RL diagnostic eval, not in isolation.

Strict final eval:

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

## How to interpret GSM8K diagnostics

- If `pass@k` is near zero, RL likely has no signal.
- If `pass@k` is nontrivial but `pass@1` is lower, RL may help sharpen the policy toward trajectories the model can already sample but not yet select reliably.
- If many responses hit `max_new_tokens`, increase decoding budget before concluding the model is too weak.

The diagnostic evaluator writes both a compact `summary.json` and a `predictions.jsonl` file with raw sampled responses, parsed terminal predictions, per-sample rewards, and response lengths for inspection.

The end-to-end artifact flow for a serious benchmark run is:

- bootstrap adapter directory
- diagnostic `summary.json` and `predictions.jsonl`
- GRPO `metrics.jsonl` and checkpoints
- final strict eval `summary.json` and `predictions.jsonl`

That adapter lifecycle is the core post-training story: bootstrap writes an adapter, eval loads it, GRPO starts from it with `init_adapter_path`, and final eval loads the saved RL adapter.

## Real GSM8K status

The current story is:

- Cold-start strict RL on GSM8K did not work on small models.
- Answer-only SFT plus shaped reward produced superficially better rewards, but much of that signal came from formatting behavior rather than robust problem solving.
- Rationale-based SFT bootstrap plus a strict binary verifier produced a meaningful bootstrap policy on the filtered GSM8K subset.
- On `Qwen/Qwen2.5-1.5B-Instruct`, bootstrap diagnostics showed:
  - `pass@1 = 0.5156`
  - `pass@8 = 0.7734`
  - `fraction_with_any_correct = 0.7734`
- After strict binary GRPO, strict greedy evaluation reached:
  - `pass@1 = 0.6875`

This result is on a filtered easy-curriculum training subset. It is a project benchmark result, not a broad GSM8K SOTA claim.

## Benchmark snapshot

| Model | Setup | Subset | Metric | Value |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Rationale SFT bootstrap | GSM8K filtered train easy subset | pass@1 | 0.5156 |
| Qwen2.5-1.5B-Instruct | Rationale SFT bootstrap | GSM8K filtered train easy subset | pass@8 | 0.7734 |
| Qwen2.5-1.5B-Instruct | Bootstrap + strict GRPO | GSM8K filtered train easy subset | pass@1 | 0.6875 |

## Synthetic task ladder

- `smoke`: ultra-easy arithmetic for the first non-degenerate reward batch
- `easy`: slightly harder synthetic arithmetic before moving to benchmark-style tasks
- `train` / `eval`: stricter arithmetic splits with more room for policy improvement
- `gsm8k_subset`: filtered real GSM8K examples for benchmark-style experiments

## Single-GPU playbook

- `batch_size * group_size` is the main VRAM dial.
- Enable gradient checkpointing only when needed.
- Continuous batching helps most when sequence lengths vary across active samples.
- Persistent KV decode helps decode cost but is not a full paged-KV runtime.
- Larger `max_new_tokens` can be necessary for real reasoning tasks.
- If many prompt groups are all-correct or all-wrong, increase prompt batch size so each optimizer step sees more than one prompt group.

## Current constraints

- `use_lora=True` is required.
- The framework is single-device only.
- Continuous batching is not a full paged-KV or vLLM-style runtime; optional vLLM-class rollout remains future work (`experimental_vllm_rollout` is reserved and not implemented).
- `top_p` support may still be limited depending on the code path.
- The strongest benchmark comparison still comes from explicit bootstrap-vs-post-RL evaluation on saved adapters.
- External task portability is part of the project story, but the repo does **not** ship a production external task service. The external API-backed integration pattern is documented in [docs/open_source_demo.md](docs/open_source_demo.md).

## Security note

- Replay trajectories are loaded with `torch.load(..., weights_only=False)`. Do not load untrusted trajectory files.
