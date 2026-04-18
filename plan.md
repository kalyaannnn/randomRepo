# Paged-KV Branch Plan

## Summary

Keep the current repo as OSS v1: balanced runtime plus trainer, with the existing continuous batching path as the baseline. Build full paged-KV continuous batching on a separate branch as a v2 runtime-engine track.

The branch should have one clear goal: materially improve the single-GPU rollout engine while preserving the post-training workflow already proven in v1.

## Key Changes

- Add a real paged-KV runtime path.
- Keep the trainer and task API stable.
- Add engine-level observability for block allocation and KV utilization.
- Reposition the repo only after the paged-KV path is correct and benchmarked.

## Implementation Changes

### Allocator and cache model

- Define a fixed-size KV block abstraction and a per-sequence block table.
- Implement deterministic allocation, append, release, and reuse semantics.
- Choose one block sizing policy and keep it fixed for the first branch version.

### Continuous decode integration

- Refactor the cached continuous batching path in `agentrl/generation/continuous.py` so active decode no longer depends on stacking and splitting per-sequence cache objects.
- Preserve the existing fallback path for models that do not support the required decode behavior.
- Make scheduler KV accounting block-aware rather than only token-length-aware.

### Metrics and runtime controller

- Add paged-KV-specific runtime counters:
  - free block count
  - used block count
  - allocator occupancy
  - block reuse
  - allocator pressure
- Keep existing metrics:
  - throughput
  - step latency
  - headroom
  - KV pressure
  - bottleneck diagnosis
- Extend benchmark reporting so it can compare:
  - standard rollout
  - current persistent-KV continuous batching
  - paged-KV continuous batching

### Benchmark matrix

- Hold model, task, batch size, group size, and decode budget constant.
- Benchmark workloads with meaningful sequence-length variance.
- Compare:
  - latency
  - tokens/sec
  - padding waste
  - KV pressure
  - active concurrency
  - peak VRAM
  - runtime recommendations
  - correctness parity

## Test Plan

### Correctness tests

- Block allocation and free behavior for one and many sequences.
- Reuse of freed blocks without cross-sequence corruption.
- Sequence growth across block boundaries.
- Cleanup for finished and truncated sequences.
- Deterministic logical-to-physical mapping under seeded runs.

### Runtime integration tests

- Paged-KV continuous batching matches baseline outputs and rewards on controlled toy environments.
- Decode remains correct when active sequence lengths diverge sharply.
- KV-budgeted admission respects allocator limits and reports pressure correctly.

### Benchmark validation

- Standard rollout vs current continuous batching vs paged-KV continuous batching.
- Validate qualitative improvements in:
  - allocator waste
  - concurrency under equal budget
  - effective KV utilization
- Do not merge if correctness regresses, even if throughput improves.

## Assumptions

- OSS v1 ships before paged-KV lands.
- Public API remains stable:
  - `BaseEnvironment`
  - `BaseVerifier`
  - `GRPOConfig`
  - bootstrap trainer contract
- The branch is a single-GPU RL runtime-engine upgrade, not a pivot to a general production serving stack.
- README positioning should only change after the new runtime path is proven and benchmarked.
