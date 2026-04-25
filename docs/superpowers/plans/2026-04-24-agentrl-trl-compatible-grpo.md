# AgentRL TRL-Compatible GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AgentRL's current GRPO-like policy/reference delta objective with a TRL-compatible old-policy clipped GRPO path while preserving the shared-base single-GPU LoRA runtime story and separating external parity from internal systems benchmarking.

**Architecture:** Keep AgentRL adapter-only and single-GPU. Change the rollout contract to store `old_policy_logprobs` plus `completion_mask`, upgrade the shared layout from "adapter on vs adapter off" to a shared-base dual-adapter `policy` / `reference` design, and rewrite the trainer loss around current-vs-old PPO-style clipping with optional sampled-token KL. Runtime backends remain distinct implementations but must emit the same batch semantics.

**Tech Stack:** AgentRL, PyTorch, Transformers, PEFT/LoRA, pytest, existing example benchmark scripts.

---

## File Structure

- `agentrl/core/config.py`
  - Add TRL-compatible objective config and reject mixed legacy semantics.
- `agentrl/core/rollout.py`
  - Rename rollout tensors and stop computing reference rollout logprobs.
- `agentrl/generation/continuous.py`
  - Emit `completion_mask` and `old_policy_logprobs` for continuous and paged-KV paths.
- `agentrl/generation/speculative.py`
  - Preserve speculative policy logprob caching but export it as `old_policy_logprobs`.
- `agentrl/core/trainer.py`
  - Replace current sequence-delta loss with current-vs-old clipped GRPO and optional sampled-token KL.
- `agentrl/memory/layout.py`
  - Introduce shared-base dual-adapter `policy` / `reference` switching.
- `agentrl/memory/buffer.py`
  - Persist renamed fields and support one migration window for legacy trajectory payloads.
- `agentrl/observability/debugger.py`
  - Render debug information from the new batch contract and trainer metrics.
- `agentrl/observability/replay.py`
  - Replay remains mostly stable but must consume migrated trajectory payloads.
- `tests/test_rollout.py`
  - Validate rollout batch semantics and mask behavior.
- `tests/test_speculative.py`
  - Validate speculative export semantics under the renamed batch contract.
- `tests/test_trainer.py`
  - Validate trainer-step metrics and the new clipped loss path.
- `tests/test_buffer.py`
  - Validate trajectory serialization roundtrips and legacy compatibility.
- `tests/test_observability.py`
  - Validate debugger and replay rendering under the renamed fields.
- `tests/test_base.py`
  - Validate config invariants for the new GRPO path.
- `tests/test_grpo_objective.py`
  - New analytical objective tests for ratio, clipping, and KL.
- `examples/benchmark_systems.py`
  - Keep internal runtime benchmarking focused on AgentRL runtime modes only.
- `docs/superpowers/plans/2026-04-23-phase1-agentrl-vs-trl-single-turn.md`
  - Existing external parity execution plan; reuse it after this GRPO core migration lands.
- `examples/train_math.py`
  - Update defaults and CLI docs for the new objective semantics if needed.
- `README.md`
  - Clarify the new TRL-compatible objective wording and benchmark split.

## Task 1: Lock the config and batch-schema migration

**Files:**
- Modify: `agentrl/core/config.py`
- Modify: `agentrl/core/rollout.py`
- Modify: `tests/test_base.py`
- Modify: `tests/test_rollout.py`

- [ ] **Step 1: Write failing config tests for the new objective knobs and rejected legacy combinations**

Add focused tests in `tests/test_base.py`:

```python
def test_config_defaults_to_trl_compatible_grpo() -> None:
    config = GRPOConfig(model_name="fake/model")
    assert config.beta == 0.0
    assert config.epsilon == 0.2
    assert config.num_iterations == 1


def test_config_rejects_num_iterations_other_than_one_for_now() -> None:
    with pytest.raises(ConfigurationError):
        GRPOConfig(model_name="fake/model", num_iterations=2)


def test_config_rejects_legacy_kl_adaptation_in_trl_mode() -> None:
    with pytest.raises(ConfigurationError):
        GRPOConfig(model_name="fake/model", use_adaptive_kl=True, kl_target=0.1)
```

- [ ] **Step 2: Write a failing rollout-batch test for renamed fields**

Update `tests/test_rollout.py` to expect:

```python
assert hasattr(batch, "completion_mask")
assert hasattr(batch, "old_policy_logprobs")
assert not hasattr(batch, "ref_logprobs")
assert batch.completion_mask.shape == batch.input_ids.shape
assert batch.old_policy_logprobs.shape == batch.input_ids.shape
```

- [ ] **Step 3: Run the targeted tests to confirm the current code fails**

Run:

```bash
pytest tests/test_base.py tests/test_rollout.py -k "trl_compatible or num_iterations or completion_mask or old_policy_logprobs" -v
```

Expected: FAIL because `GRPOConfig` lacks the new fields and `RolloutBatch` still exposes `action_mask` / `policy_logprobs` / `ref_logprobs`.

- [ ] **Step 4: Implement the config migration in `agentrl/core/config.py`**

Make the dataclass surface look like:

```python
epsilon: float = 0.2
num_iterations: int = 1
grpo_mode: str = "trl"
beta: float = 0.0
```

Add validation:

```python
if self.num_iterations != 1:
    raise ConfigurationError("num_iterations != 1 is out of scope for this pass.")
if self.grpo_mode != "trl":
    raise ConfigurationError("Only grpo_mode='trl' is supported after the redesign.")
if self.use_adaptive_kl or self.kl_target is not None:
    raise ConfigurationError("Adaptive KL controls are not supported in the TRL-compatible path.")
```

- [ ] **Step 5: Implement the batch-schema rename in `agentrl/core/rollout.py`**

Change the dataclass:

```python
@dataclass(slots=True)
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_policy_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    metadata: dict[str, Any]
```

Rename local variables and helper signatures so `action_mask` becomes `completion_mask` consistently.

- [ ] **Step 6: Run the targeted tests again**

Run:

```bash
pytest tests/test_base.py tests/test_rollout.py -k "trl_compatible or num_iterations or completion_mask or old_policy_logprobs" -v
```

Expected: PASS

- [ ] **Step 7: Commit the schema/config foundation**

```bash
git add agentrl/core/config.py agentrl/core/rollout.py tests/test_base.py tests/test_rollout.py
git commit -m "Rename rollout batch fields for TRL-compatible GRPO"
```

## Task 2: Upgrade the shared layout to a dual-adapter policy/reference design

**Files:**
- Modify: `agentrl/memory/layout.py`
- Modify: `agentrl/core/trainer.py`
- Create: `tests/test_layout.py`

- [ ] **Step 1: Write a failing layout test for named adapter switching**

Create `tests/test_layout.py` with a focused fake-PEFT test:

```python
def test_shared_weight_layout_uses_named_policy_and_reference_adapters() -> None:
    layout = SharedWeightLayout.__new__(SharedWeightLayout)
    calls = []

    class FakeModel:
        def set_adapter(self, name):
            calls.append(("set_adapter", name))
        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(logits=torch.zeros((input_ids.shape[0], input_ids.shape[1], 4)))

    layout.model = FakeModel()
    layout._sdpa_context = lambda: nullcontext()
    layout._activate_adapter = SharedWeightLayout._activate_adapter.__get__(layout, SharedWeightLayout)

    layout.policy_forward(torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long))
    layout.reference_forward(torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long))

    assert calls == [("set_adapter", "policy"), ("set_adapter", "reference")]
```

- [ ] **Step 2: Run the test to confirm the current layout fails**

Run:

```bash
pytest tests/test_layout.py::test_shared_weight_layout_uses_named_policy_and_reference_adapters -v
```

Expected: FAIL because `SharedWeightLayout` still toggles adapter layers globally instead of selecting named adapters.

- [ ] **Step 3: Refactor `agentrl/memory/layout.py` to create and freeze `policy` and `reference` adapters**

The implementation target is:

```python
self.policy_adapter_name = "policy"
self.reference_adapter_name = "reference"

# Create/load policy adapter
# Copy policy adapter weights into frozen reference adapter

def _activate_adapter(self, name: str) -> None:
    set_adapter = getattr(self.model, "set_adapter", None)
    if set_adapter is None:
        raise AttributeError("PEFT model does not expose set_adapter().")
    set_adapter(name)
```

Freeze policy/reference params correctly:

```python
for name, parameter in self.model.named_parameters():
    if f".{self.policy_adapter_name}." in name:
        parameter.requires_grad = True
    else:
        parameter.requires_grad = False
```

- [ ] **Step 4: Update trainer construction to rely on the layout for reference creation**

Ensure `_build_layout()` in `agentrl/core/trainer.py` continues to pass one init adapter path only, and let `SharedWeightLayout` own the policy/reference snapshot logic rather than duplicating that logic in the trainer.

- [ ] **Step 5: Run the targeted layout tests**

Run:

```bash
pytest tests/test_layout.py tests/test_trainer.py -k "adapter or startup_vram_report" -v
```

Expected: PASS

- [ ] **Step 6: Commit the layout refactor**

```bash
git add agentrl/memory/layout.py agentrl/core/trainer.py tests/test_layout.py tests/test_trainer.py
git commit -m "Add shared-base policy and reference adapters"
```

## Task 3: Migrate standard and continuous rollouts to `old_policy_logprobs`

**Files:**
- Modify: `agentrl/core/rollout.py`
- Modify: `agentrl/generation/continuous.py`
- Modify: `tests/test_rollout.py`

- [ ] **Step 1: Add failing tests that rollout no longer computes reference logprobs**

Update `tests/test_rollout.py` with:

```python
def test_rollout_collects_old_policy_logprobs_only() -> None:
    batch = orchestrator.collect()
    assert batch.old_policy_logprobs.abs().sum().item() > 0.0
    assert "ref_logprobs" not in batch.__dict__
```

Add a continuous-batching version if the file already covers continuous collection.

- [ ] **Step 2: Run the targeted rollout tests and confirm failure**

Run:

```bash
pytest tests/test_rollout.py -k "old_policy_logprobs_only or multi_turn_grouped_batch" -v
```

Expected: FAIL because both standard and continuous collection still compute and store `ref_logprobs`.

- [ ] **Step 3: Update standard rollout collection**

In `agentrl/core/rollout.py`, keep:

```python
old_policy_sequences = self._compute_logprobs(
    self.layout.policy_forward,
    flat_input_ids,
    flat_attention_mask,
    flat_completion_mask,
)
```

Delete the rollout-time reference branch entirely.

Return:

```python
return RolloutBatch(
    input_ids=input_ids,
    attention_mask=attention_mask,
    completion_mask=completion_mask,
    old_policy_logprobs=old_policy_sequences.view_as(input_ids),
    rewards=rewards,
    advantages=advantages,
    metadata=metadata,
)
```

- [ ] **Step 4: Remove parity-path advantage clipping**

In `_compute_advantages(...)`, change:

```python
return normalized
```

and delete the final `clamp(...)` in the parity path.

- [ ] **Step 5: Update continuous rollout collection to match the same contract**

In `agentrl/generation/continuous.py`, make the same structural change:

```python
old_policy_sequences = self._compute_logprobs(
    self.layout.policy_forward,
    flat_input_ids,
    flat_attention_mask,
    flat_completion_mask,
)
return RolloutBatch(
    input_ids=input_ids,
    attention_mask=attention_mask,
    completion_mask=completion_mask,
    old_policy_logprobs=old_policy_sequences.view_as(input_ids),
    rewards=rewards,
    advantages=advantages,
    metadata=metadata,
)
```

- [ ] **Step 6: Run standard and continuous rollout tests**

Run:

```bash
pytest tests/test_rollout.py tests/test_examples.py -k "rollout or benchmark_systems" -v
```

Expected: PASS for the rollout contract and no immediate benchmark-script breakage from the renamed fields.

- [ ] **Step 7: Commit the rollout migration**

```bash
git add agentrl/core/rollout.py agentrl/generation/continuous.py tests/test_rollout.py tests/test_examples.py
git commit -m "Export old policy logprobs from standard and continuous rollouts"
```

## Task 4: Migrate speculative rollout without losing cached rollout semantics

**Files:**
- Modify: `agentrl/generation/speculative.py`
- Modify: `tests/test_speculative.py`

- [ ] **Step 1: Write a failing speculative test against the new field names**

Update `tests/test_speculative.py`:

```python
def test_speculative_orchestrator_collects_old_policy_logprobs() -> None:
    batch = orchestrator.collect()
    assert batch.old_policy_logprobs.abs().sum().item() > 0.0
    assert batch.completion_mask.sum().item() > 0
```

- [ ] **Step 2: Run the speculative tests and confirm failure**

Run:

```bash
pytest tests/test_speculative.py -v
```

Expected: FAIL because speculative still returns `action_mask`, `policy_logprobs`, and `ref_logprobs`.

- [ ] **Step 3: Rename speculative export fields without changing the cached meaning**

In `agentrl/generation/speculative.py`, keep the per-turn policy logprob caching and rename the exported tensors:

```python
input_ids, attention_mask, completion_mask, old_policy_logprobs = self._pack_speculative_sequences(episodes)
```

Return:

```python
return RolloutBatch(
    input_ids=input_ids,
    attention_mask=attention_mask,
    completion_mask=completion_mask,
    old_policy_logprobs=old_policy_logprobs,
    rewards=rewards,
    advantages=advantages,
    metadata=metadata,
)
```

Delete the rollout-time reference scoring block entirely.

- [ ] **Step 4: Run speculative tests again**

Run:

```bash
pytest tests/test_speculative.py -v
```

Expected: PASS

- [ ] **Step 5: Commit the speculative migration**

```bash
git add agentrl/generation/speculative.py tests/test_speculative.py
git commit -m "Rename speculative rollout outputs to old policy semantics"
```

## Task 5: Rewrite the trainer around clipped current-vs-old GRPO

**Files:**
- Modify: `agentrl/core/trainer.py`
- Modify: `tests/test_trainer.py`
- Create: `tests/test_grpo_objective.py`

- [ ] **Step 1: Add failing analytical tests for ratio, clipping, and sampled-token KL**

Create `tests/test_grpo_objective.py`:

```python
def test_clipped_surrogate_matches_hand_calculation() -> None:
    current = torch.tensor([[0.0, 0.3]])
    old = torch.tensor([[0.0, 0.0]])
    advantages = torch.tensor([1.0])
    mask = torch.tensor([[False, True]])

    ratio = torch.exp(current[:, 1:] - old[:, 1:])
    clipped = torch.clamp(ratio, 0.8, 1.2)
    expected = -torch.min(ratio * advantages.view(-1, 1), clipped * advantages.view(-1, 1)).mean()
    assert expected.item() == pytest.approx(-1.2, rel=1e-5)


def test_sampled_token_kl_matches_trl_style_approximator() -> None:
    current = torch.tensor([-0.2])
    ref = torch.tensor([-0.5])
    kl = torch.exp(ref - current) - (ref - current) - 1
    assert kl.item() == pytest.approx(float(torch.exp(torch.tensor(-0.3)) + 0.3 - 1.0), rel=1e-5)
```

- [ ] **Step 2: Add a failing trainer-step test for clip metrics and beta skipping**

Add to `tests/test_trainer.py`:

```python
def test_trainer_step_logs_clip_metrics_and_skips_reference_when_beta_zero(monkeypatch) -> None:
    calls = {"reference": 0}

    class Layout(TrainableLayout):
        def reference_forward(self, input_ids, attention_mask):
            calls["reference"] += 1
            return super().reference_forward(input_ids, attention_mask)

    config = GRPOConfig(model_name="fake/model", beta=0.0, steps=1, batch_size=1, group_size=2)
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=Layout(),
        rollout_orchestrator=StaticRollout(batch),
    )
    _loss, metrics = trainer.step(batch)

    assert calls["reference"] == 0
    assert "clip_ratio/region_mean" in metrics
    assert "mean_ratio" in metrics
```

- [ ] **Step 3: Run the new analytical and trainer tests**

Run:

```bash
pytest tests/test_grpo_objective.py tests/test_trainer.py -k "clip or ratio or beta_zero" -v
```

Expected: FAIL because the trainer still computes the old sequence-delta loss.

- [ ] **Step 4: Rewrite `GRPOTrainer.step()`**

The target structure is:

```python
current_logprobs = self._gather_sampled_logprobs(flat_input_ids, policy_logits)
old_logprobs = flat_old_policy[:, 1:]
masked_completion = flat_completion_mask[:, 1:].to(dtype=policy_logits.dtype)

ratio = torch.exp(current_logprobs - old_logprobs)
advantages = flat_advantages.unsqueeze(-1).expand_as(ratio)

surrogate_unclipped = ratio * advantages
surrogate_clipped = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
surrogate = torch.min(surrogate_unclipped, surrogate_clipped)
policy_loss = -(surrogate * masked_completion).sum() / masked_completion.sum().clamp(min=1.0)
```

Compute KL only when `beta > 0.0`:

```python
if self.config.beta > 0.0:
    ref_logprobs = self._gather_sampled_logprobs(flat_input_ids, ref_logits)
    token_kl = torch.exp(ref_logprobs - current_logprobs) - (ref_logprobs - current_logprobs) - 1.0
else:
    token_kl = torch.zeros_like(current_logprobs)
```

- [ ] **Step 5: Replace `_token_statistics(...)` with sampled-token helpers**

Split helpers so the code is easier to test:

```python
def _gather_sampled_logprobs(
    self,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    token_logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return token_logprobs.gather(dim=-1, index=targets).squeeze(-1)

def _compute_clipped_surrogate(
    self,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    masked_completion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(current_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
    surrogate = torch.min(unclipped, clipped)
    return surrogate, ratio, clipped

def _compute_sampled_token_kl(
    self,
    current_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    delta = ref_logprobs - current_logprobs
    return torch.exp(delta) - delta - 1.0
```

- [ ] **Step 6: Update metrics to expose parity-relevant signals**

Metrics should include:

```python
{
    "clip_ratio/region_mean": float(clipped_region.mean().item()),
    "clip_ratio/low_mean": float((ratio < 1 - self.config.epsilon).float().mean().item()),
    "clip_ratio/high_mean": float((ratio > 1 + self.config.epsilon).float().mean().item()),
    "mean_ratio": float((ratio * masked_completion).sum().item() / masked_completion.sum().item()),
    "mean_token_kl": float((token_kl * masked_completion).sum().item() / masked_completion.sum().item()),
}
```

Delete:

```python
if self.config.use_adaptive_kl:
    self._update_beta(...)
```

and remove `_update_beta(...)` once nothing calls it.

- [ ] **Step 7: Run the trainer and analytical tests**

Run:

```bash
pytest tests/test_grpo_objective.py tests/test_trainer.py -v
```

Expected: PASS

- [ ] **Step 8: Commit the trainer rewrite**

```bash
git add agentrl/core/trainer.py tests/test_trainer.py tests/test_grpo_objective.py
git commit -m "Rewrite GRPO trainer around clipped old policy ratios"
```

## Task 6: Migrate trajectory persistence and observability

**Files:**
- Modify: `agentrl/memory/buffer.py`
- Modify: `agentrl/observability/debugger.py`
- Modify: `agentrl/observability/replay.py`
- Modify: `tests/test_buffer.py`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Write failing buffer tests for renamed fields and legacy payload support**

Update `tests/test_buffer.py`:

```python
def test_buffer_roundtrips_new_completion_and_old_policy_fields(tmp_path: Path) -> None:
    buffer = TrajectoryBuffer(output_dir=str(tmp_path))
    batch = make_batch()
    buffer.add(batch, step=12)
    path = buffer.save(12)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["completion_mask"].dtype == torch.bool
    assert payload["old_policy_logprobs"].dtype == torch.float16


def test_buffer_loads_legacy_action_mask_payload(tmp_path: Path) -> None:
    payload = {
        "input_ids": torch.tensor([[[1, 2, 3]]], dtype=torch.int16),
        "attention_mask": torch.tensor([[[1, 1, 1]]], dtype=torch.uint8),
        "action_mask": torch.tensor([[[0, 1, 1]]], dtype=torch.bool),
        "policy_logprobs": torch.tensor([[[0.0, -0.1, -0.2]]], dtype=torch.float16),
        "ref_logprobs": torch.tensor([[[0.0, -0.2, -0.3]]], dtype=torch.float16),
        "rewards": torch.tensor([[1.0]], dtype=torch.float32),
        "advantages": torch.tensor([[1.0]], dtype=torch.float32),
        "metadata": {},
    }
    torch.save(payload, tmp_path / "trajectory_000001.pt")
    loaded = TrajectoryBuffer(output_dir=str(tmp_path)).load(1, device="cpu")
    assert torch.equal(loaded.completion_mask, payload["action_mask"])
    assert torch.equal(loaded.old_policy_logprobs, payload["policy_logprobs"].to(dtype=torch.float32))
```

- [ ] **Step 2: Write a failing debugger test against the new metrics**

Update `tests/test_observability.py`:

```python
assert "old_policy_logprob" in debug_text or "ratio=" in debug_text
assert "clip_ratio/region_mean" in debug_text or "Metrics:" in debug_text
```

- [ ] **Step 3: Run the persistence and observability tests**

Run:

```bash
pytest tests/test_buffer.py tests/test_observability.py -v
```

Expected: FAIL because serialization and debugger logic still assume `action_mask`, `policy_logprobs`, and `ref_logprobs`.

- [ ] **Step 4: Implement buffer serialization with legacy fallback**

In `agentrl/memory/buffer.py`, write new payload keys:

```python
"completion_mask": batch.completion_mask.detach().to(device="cpu", dtype=torch.bool),
"old_policy_logprobs": batch.old_policy_logprobs.detach().to(device="cpu", dtype=torch.float16),
```

Load with a compatibility shim:

```python
completion_mask = payload.get("completion_mask", payload["action_mask"])
old_policy_logprobs = payload.get("old_policy_logprobs", payload["policy_logprobs"])
```

Ignore legacy `ref_logprobs` during reconstruction.

- [ ] **Step 5: Update debugger snapshots and rendering**

Make the debugger consume:

```python
flat_old = batch.old_policy_logprobs.view(-1, batch.old_policy_logprobs.shape[-1])
flat_completion = batch.completion_mask.view(-1, batch.completion_mask.shape[-1])
```

and render something like:

```python
old_lp = float(flat_old[sequence_index, token_position].item())
lines.append(f"  token={token_id} | log_prob(old_policy)={old_lp:.4f}{marker}")
```

Do not depend on `ref_logprobs` being stored in the batch.

- [ ] **Step 6: Run the tests again**

Run:

```bash
pytest tests/test_buffer.py tests/test_observability.py -v
```

Expected: PASS

- [ ] **Step 7: Commit the migration**

```bash
git add agentrl/memory/buffer.py agentrl/observability/debugger.py agentrl/observability/replay.py tests/test_buffer.py tests/test_observability.py
git commit -m "Migrate replay and debugger to old policy batch fields"
```

## Task 7: Separate external parity from internal systems benchmarking

**Files:**
- Modify: `examples/benchmark_systems.py`
- Modify: `examples/train_math.py`
- Modify: `README.md`
- Modify: `tests/test_examples.py`
- Reference: `docs/superpowers/plans/2026-04-23-phase1-agentrl-vs-trl-single-turn.md`

- [ ] **Step 1: Add a failing examples test that keeps runtime comparisons internal to AgentRL**

Add to `tests/test_examples.py`:

```python
def test_benchmark_systems_only_compares_agentrl_runtime_modes(monkeypatch, tmp_path) -> None:
    seen = []

    def stubbed_run_one(args, use_continuous_batching, output_dir, *, use_paged_kv_continuous=False):
        seen.append((use_continuous_batching, use_paged_kv_continuous))
        return {"mode_name": "stub", "mean_step_time_ms": 1.0, "efficiency_diagnosis": "balanced"}

    monkeypatch.setattr(benchmark_systems, "_run_one", stubbed_run_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_systems.py",
            "--model",
            "fake/model",
            "--output-dir",
            str(tmp_path),
            "--compare-runtime-modes",
        ],
    )
    benchmark_systems_main()
    assert seen == [(False, False), (True, False), (True, True)]
```

- [ ] **Step 2: Run the targeted example tests**

Run:

```bash
pytest tests/test_examples.py -k "benchmark_systems" -v
```

Expected: Either PASS already or reveal any assumptions that now depend on the renamed metrics.

- [ ] **Step 3: Update docs and example defaults to the new objective terminology**

In `README.md` and `examples/train_math.py`, align wording with:

```markdown
Training uses a TRL-compatible GRPO objective with rollout-time old-policy logprobs,
PPO-style clipping, and optional KL against a frozen reference adapter.
```

Keep `examples/benchmark_systems.py` explicitly framed as:

```python
description="Run an AgentRL-only runtime comparison across standard, continuous, paged-KV, and speculative modes."
```

Document in `README.md` that the external AgentRL-vs-TRL parity run is handled by the separate single-turn comparison plan, while `benchmark_systems.py` remains AgentRL-only.

- [ ] **Step 4: Run the example and smoke-doc tests**

Run:

```bash
pytest tests/test_examples.py -v
```

Expected: PASS

- [ ] **Step 5: Commit the benchmark/doc split**

```bash
git add examples/benchmark_systems.py examples/train_math.py README.md tests/test_examples.py
git commit -m "Clarify external parity versus internal runtime benchmarks"
```

## Task 8: Run the final verification set

**Files:**
- Modify: none
- Test: `tests/test_base.py`
- Test: `tests/test_rollout.py`
- Test: `tests/test_speculative.py`
- Test: `tests/test_trainer.py`
- Test: `tests/test_grpo_objective.py`
- Test: `tests/test_buffer.py`
- Test: `tests/test_observability.py`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Run the full targeted GRPO migration suite**

Run:

```bash
pytest \
  tests/test_base.py \
  tests/test_rollout.py \
  tests/test_speculative.py \
  tests/test_trainer.py \
  tests/test_grpo_objective.py \
  tests/test_buffer.py \
  tests/test_observability.py \
  tests/test_examples.py -v
```

Expected: PASS

- [ ] **Step 2: Run a quick grep audit to ensure legacy batch-field names are gone from code paths**

Run:

```bash
rg -n "action_mask|policy_logprobs|ref_logprobs" agentrl tests examples
```

Expected:

- no remaining production-code uses of `action_mask`
- no remaining production-code uses of `policy_logprobs` as the batch field name
- any remaining `ref_logprobs` references are limited to trainer-local reference KL computation, not rollout batch storage

- [ ] **Step 3: Record the final status in a short summary note**

Use this checklist in the implementation PR or commit summary:

```text
- rollout batches store completion_mask + old_policy_logprobs
- trainer computes current-vs-old clipped surrogate
- reference scoring is update-time only and skipped when beta == 0
- shared-base dual-adapter layout is active
- replay/debugger paths support the migrated contract
- runtime benchmark remains internal to AgentRL
```

- [ ] **Step 4: Commit any final cleanup if needed**

```bash
git add -A
git commit -m "Finish TRL-compatible GRPO migration"
```
