
› # AgentRL

  AgentRL is a lightweight single-GPU framework for verifier-based GRPO style
  post-training of language models.

  It is designed for small-scale single-device experimentation with pluggable
  text environments, deterministic verifiers, LoRA-based policy updates, and a
  practical path from synthetic tasks to real benchmarks.

  ## What AgentRL currently supports

  - GRPO style post-training with LoRA adapters
  - Pluggable text environments and deterministic verifiers
  - Shared-weight LoRA policy/reference setup
  - Standard rollout and continuous batching
  - Chunked prefill in both rollout paths
  - Persistent KV decode for cache-capable models in continuous batching
  - CPU-backed trajectory buffering and replay
  - Checkpoint saving during training plus a final adapter alias
  - Metrics logging, debugger hooks, profiler utilities, and VRAM headroom
  reporting
  - Synthetic arithmetic task ladder and a real filtered GSM8K subset workflow

  ## Current Project Status

  ### Implemented

  - Core GRPO loop works end to end
  - Shared-weight LoRA policy/reference layout
  - Continuous batching with persistent KV decode for cache-capable models
  - Chunked prefill in standard rollout and continuous batching
  - Trajectory replay, logging, debugger, profiler, and buffer eviction
  - Adapter checkpoint saving and final alias
  - Startup VRAM and headroom reporting
  - Synthetic task ladder
  - Real GSM8K subset path
  - Rationale-based SFT bootstrap
  - Strict binary verifier path for GSM8K
  - Strict evaluation and pass@k diagnostics
  - Test suite is green

  ### Validated Experimentally

  - Single-GPU Colab smoke runs worked on real Hugging Face models
  - On a Tesla T4 with `Qwen/Qwen2.5-1.5B-Instruct`, the systems benchmark
  showed continuous batching reducing mean step time from `22.47s` to `8.71s`
  and increasing throughput from `11.28` to `29.20` tokens/s on the same
  workload
  - Synthetic tasks produce non-degenerate reward signal
  - Multi-step GRPO updates on synthetic tasks are real training updates, not
  just plumbing
  - Real filtered GSM8K subset runs work with rationale-based SFT bootstrap,
  strict exact-match style verification, pass@k diagnostics, and GRPO fine-
  tuning from a bootstrap adapter
  - Persistent-KV continuous batching is implemented and unit tested

  ### Still In Progress

  - Runtime is better than the initial prototype, but it is not yet a paged-KV
  or vLLM-style production decode engine
  - Cache management and scheduling are still simpler than production serving
  runtimes
  - The strongest benchmark story still comes from explicit bootstrap-vs-post-
  RL evaluation on saved adapters
  - GSM8K training quality still depends heavily on bootstrap strength,
  decoding budget, and subset difficulty

  ## Installation

  ```bash
  pip install -e .
  pip install datasets
  ```

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
  python -m examples.train_math --model Qwen/Qwen2.5-1.5B-Instruct --steps 5
  --split smoke
  ```

  ## Systems Benchmark Example

  Use the systems benchmark when you want to compare rollout implementations,
  runtime bottlenecks, VRAM pressure, and cache reuse on a fixed workload.

  Comparison command:

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

  One measured Colab T4 run on the `easy` math workload produced:

  - Standard rollout: `22.47s` mean step time, `11.28` tokens/s, diagnosis
  `decode-limited`, bottleneck `decode_without_cache_reuse`
  - Continuous batching: `8.71s` mean step time, `29.20` tokens/s, diagnosis
  `decode-limited`, bottleneck `decode`

  In that comparison, continuous batching was about `2.6x` faster on mean step
  time with near-zero padding waste in both modes. The benchmark verdict is
  intentionally systems-focused: the same run still had zero reward under the
  strict verifier, but it was useful for identifying decode as the dominant
  bottleneck and quantifying the benefit of cache reuse.

  ## GSM8K Workflow

  The current GSM8K path is:

  1. Bootstrap a LoRA adapter with rationale-based supervised targets derived
  from GSM8K solutions.
  2. Run diagnostic evaluation to measure `pass@1` and `pass@k` before RL.
  3. Run GRPO with a strict binary verifier only if the bootstrap model already
  produces some correct trajectories.
  4. Evaluate the saved GRPO adapter with strict exact match.

  The GSM8K verifier is intentionally strict:

  - reward is `1.0` only if the last non-empty line is exactly `Final answer:
  <integer>` and the integer matches the gold answer
  - reward is `0.0` otherwise
  - there is no partial reward for formatting, near misses, or fallback integer
  extraction

  Bootstrap is rationale-based rather than answer-only. The point of the
  diagnostic stage is to answer a concrete question before spending RL compute:
  does the bootstrap adapter produce any correct strict trajectories at all?

  ## Recommended GSM8K Commands

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

  ## How To Interpret GSM8K Diagnostics

  - If `pass@k` is near zero, RL likely has no signal.
  - If `pass@k` is nontrivial but `pass@1` is lower, RL may help sharpen the
  policy toward trajectories the model can already sample but not yet select
  reliably.
  - If many responses hit `max_new_tokens`, increase decoding budget before
  concluding the model is too weak.

  The diagnostic evaluator writes both a compact `summary.json` and a
  `predictions.jsonl` file with raw sampled responses, parsed terminal
  predictions, per-sample rewards, and response lengths for inspection.

  ## Real GSM8K Status

  The current story is:

  - Cold-start strict RL on GSM8K did not work on small models.
  - Answer-only SFT plus shaped reward produced superficially better rewards,
  but much of that signal came from formatting behavior rather than robust
  problem solving.
  - Rationale-based SFT bootstrap plus a strict binary verifier produced a
  meaningful bootstrap policy on the filtered GSM8K subset.
  - On `Qwen/Qwen2.5-1.5B-Instruct`, bootstrap diagnostics showed:
    - `pass@1 = 0.5156`
    - `pass@8 = 0.7734`
    - `fraction_with_any_correct = 0.7734`
  - After strict binary GRPO, strict greedy evaluation reached:
    - `pass@1 = 0.6875`

  This result is on a filtered easy-curriculum training subset. It is a project
  benchmark result, not a broad GSM8K SOTA claim.

  ## Benchmark Snapshot

  | Model | Setup | Subset | Metric | Value |
  |---|---|---|---|---|
  | Qwen2.5-1.5B-Instruct | Rationale SFT bootstrap | GSM8K filtered train easy
  subset | pass@1 | 0.5156 |
  | Qwen2.5-1.5B-Instruct | Rationale SFT bootstrap | GSM8K filtered train easy
  subset | pass@8 | 0.7734 |
  | Qwen2.5-1.5B-Instruct | Bootstrap + strict GRPO | GSM8K filtered train easy
  subset | pass@1 | 0.6875 |

  ## Synthetic Task Ladder

  - `smoke`: ultra-easy arithmetic for the first non-degenerate reward batch
  - `easy`: slightly harder synthetic arithmetic before moving to benchmark-
  style tasks
  - `train` / `eval`: stricter arithmetic splits with more room for policy
  improvement
  - `gsm8k_subset`: filtered real GSM8K examples for benchmark-style
  experiments

  ## Single-GPU Playbook

  - `batch_size * group_size` is the main VRAM dial.
  - Enable gradient checkpointing only when needed.
  - Continuous batching helps most when sequence lengths vary across active
  samples.
  - Persistent KV decode is useful, but it is not yet a full paged-KV runtime.
  - Larger `max_new_tokens` can be necessary for real reasoning tasks.
  - If many prompt groups are all-correct or all-wrong, increase prompt batch
  size so each optimizer step sees more than one prompt group.

  ## Current Constraints

  - `use_lora=True` is required.
  - The framework is single-device only.
  - Continuous batching is not yet a full paged-KV or vLLM-style runtime.
  - `top_p` support may still be limited depending on the code path.
  - The strongest benchmark comparison still comes from explicit bootstrap-vs-
  post-RL evaluation on saved adapters.

  ## Security Note

  - Replay trajectories are loaded with `torch.load(..., weights_only=False)`.
  Do not load untrusted trajectory files.
