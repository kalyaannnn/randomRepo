from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.core.rollout import RolloutBatch
from agentrl.core.trainer import GRPOTrainer


class MinimalEnvironment(BaseEnvironment):
    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> tuple[str, bool]:
        del action
        return ("done", True)

    def state(self) -> dict[str, str]:
        return {"expected": "x"}


class MinimalVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, str]) -> float:
        return 1.0 if response == env_state["expected"] else 0.0


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"


class TrainableLayout:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.config = SimpleNamespace(use_cache=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.ref_bias = 0.25
        self.saved_adapters = []
        self.reference_forward_calls = 0

    def trainable_parameters(self):
        yield self.logit_scale

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 4
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32)
        logits[:, :, 1] = self.logit_scale
        return logits

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        self.reference_forward_calls += 1
        batch, seq = input_ids.shape
        vocab = 4
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32)
        logits[:, :, 1] = self.ref_bias
        return logits

    def save_adapter(self, path):
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.saved_adapters.append(output)
        return output

    def vram_report(self) -> dict[str, float]:
        return {"base_mb": 10.0, "adapter_mb": 1.5, "total_mb": 11.5}


class StaticRollout:
    def __init__(self, batch: RolloutBatch) -> None:
        self.batch = batch

    def collect(self) -> RolloutBatch:
        return self.batch


class ClosingLogger:
    def __init__(self) -> None:
        self.closed = False
        self.rows = []

    def log(self, step: int, metrics: dict[str, float]) -> str:
        self.rows.append((step, metrics))
        return "logged"

    def close(self) -> None:
        self.closed = True


class RecordingTrajectoryBuffer:
    def __init__(self) -> None:
        self.added_steps: list[int] = []
        self.saved_steps: list[int] = []

    def add(self, batch: RolloutBatch, step: int) -> None:
        del batch
        self.added_steps.append(step)

    def save(self, step: int) -> None:
        self.saved_steps.append(step)


def make_batch(group_size: int = 2) -> RolloutBatch:
    input_rows = [[0, 1, 1], [0, 1, 2]]
    reward_row = [1.0, 0.0]
    advantage_row = [1.0, -1.0]
    response_row = ["x", "y"]
    if group_size == 4:
        input_rows = [[0, 1, 1], [0, 1, 2], [0, 1, 1], [0, 1, 2]]
        reward_row = [1.0, 0.0, 1.0, 0.0]
        advantage_row = [1.0, -1.0, 1.0, -1.0]
        response_row = ["x", "y", "x", "y"]

    input_ids = torch.tensor([input_rows], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.tensor(
        [[[0, 1, 1] for _ in range(group_size)]],
        dtype=torch.bool,
    )
    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_policy_logprobs=torch.zeros_like(input_ids, dtype=torch.float32),
        rewards=torch.tensor([reward_row], dtype=torch.float32),
        advantages=torch.tensor([advantage_row], dtype=torch.float32),
        metadata={"unique_response_ratio": 1.0, "responses": [response_row]},
    )


def test_trainer_step_updates_trainable_parameter() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        max_new_tokens=4,
    )
    batch = make_batch()
    layout = TrainableLayout()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=layout,
        rollout_orchestrator=StaticRollout(batch),
    )

    before = layout.logit_scale.detach().clone()
    loss, metrics = trainer.step(batch)
    after = layout.logit_scale.detach().clone()

    assert loss.item() != 0.0
    assert after.item() != before.item()
    assert metrics["mean_reward"] == 0.5
    assert "mean_token_kl" in metrics
    assert metrics["learning_rate"] == config.lr
    assert metrics["beta"] == config.beta
    assert metrics["unique_response_ratio"] == 1.0


def test_trainer_closes_metrics_logger_after_train() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        use_continuous_batching=False,
    )
    batch = make_batch()
    logger = ClosingLogger()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=StaticRollout(batch),
        metrics_logger=logger,
        trajectory_buffer=RecordingTrajectoryBuffer(),
    )

    trainer.train()

    assert logger.closed is True
    assert len(logger.rows) == 1
    logged_metrics = logger.rows[0][1]
    assert "prefill_time_ms" in logged_metrics
    assert "decode_time_ms" in logged_metrics
    assert "cache_reuse_effectiveness" in logged_metrics
    assert "rollout_peak_vram_mb" in logged_metrics


def test_trainer_saves_periodic_and_final_adapters(tmp_path) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=2,
        save_every=1,
        output_dir=str(tmp_path),
        use_continuous_batching=False,
    )
    batch = make_batch()
    layout = TrainableLayout()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=layout,
        rollout_orchestrator=StaticRollout(batch),
        metrics_logger=ClosingLogger(),
        trajectory_buffer=RecordingTrajectoryBuffer(),
    )

    trainer.train()

    saved_names = [path.name for path in layout.saved_adapters]
    assert saved_names == [
        "checkpoint_000001",
        "checkpoint_000002",
        "checkpoint_final",
    ]


def test_trainer_exposes_startup_vram_report() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        device="cpu",
    )
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=StaticRollout(make_batch()),
    )

    assert trainer.startup_report["device"] == "cpu"
    assert trainer.startup_report["parameter_total_mb"] == 11.5
    assert trainer.startup_report["runtime_headroom_mb"] is None


def test_trainer_enables_gradient_checkpointing_hooks() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        use_gradient_checkpointing=True,
        device="cpu",
    )
    layout = TrainableLayout()
    flags = {"checkpointing": 0, "input_grads": 0}

    def _enable_checkpointing():
        flags["checkpointing"] += 1

    def _enable_input_grads():
        flags["input_grads"] += 1

    layout.model.gradient_checkpointing_enable = _enable_checkpointing
    layout.model.enable_input_require_grads = _enable_input_grads

    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=layout,
        rollout_orchestrator=StaticRollout(make_batch()),
    )

    assert flags["checkpointing"] == 1
    assert flags["input_grads"] == 1
    assert trainer.gradient_checkpointing_enabled is True
    assert trainer.startup_report["gradient_checkpointing_requested"] is True
    assert trainer.startup_report["gradient_checkpointing_enabled"] is True


def test_trainer_creates_profile_trace(tmp_path) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        use_continuous_batching=False,
        profile_steps=1,
        profile_dir=str(tmp_path / "profiles"),
    )
    batch = make_batch()
    logger = ClosingLogger()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=StaticRollout(batch),
        metrics_logger=logger,
        trajectory_buffer=RecordingTrajectoryBuffer(),
    )

    trainer.train()

    trace_path = Path(logger.rows[0][1]["profile_trace_path"])
    assert trace_path.exists()
    assert trace_path.stat().st_size > 0


def test_trainer_retries_after_rollout_oom() -> None:
    class OOMThenSuccessRollout:
        def __init__(self, config: GRPOConfig, batch: RolloutBatch) -> None:
            self.config = config
            self.batch = batch
            self.calls = 0

        def collect(self) -> RolloutBatch:
            self.calls += 1
            if self.calls == 1 and (self.config.chunk_size or 0) > 1:
                raise RuntimeError("CUDA out of memory while collecting rollout")
            return self.batch

    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=4,
        chunk_size=4,
        steps=1,
        use_continuous_batching=True,
        oom_retry_budget=1,
    )
    batch = make_batch(group_size=4)
    rollout = OOMThenSuccessRollout(config=config, batch=batch)
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=rollout,
        metrics_logger=ClosingLogger(),
        trajectory_buffer=RecordingTrajectoryBuffer(),
    )

    trainer.train()

    assert rollout.calls == 2
    assert config.chunk_size == 2


def test_trainer_logs_scheduler_and_static_beta() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=2,
        use_continuous_batching=False,
        lr=1e-3,
        lr_scheduler="cosine",
        min_lr_ratio=0.5,
    )
    batch = make_batch()
    logger = ClosingLogger()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=StaticRollout(batch),
        metrics_logger=logger,
        trajectory_buffer=RecordingTrajectoryBuffer(),
    )

    trainer.train()

    first_metrics = logger.rows[0][1]
    second_metrics = logger.rows[1][1]
    assert first_metrics["learning_rate"] == config.lr
    assert second_metrics["learning_rate"] < config.lr
    assert first_metrics["beta"] == config.beta
    assert second_metrics["beta"] == config.beta
    assert "mean_token_kl" in second_metrics


def test_trainer_skips_reference_forward_when_beta_is_zero() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        beta=0.0,
    )
    layout = TrainableLayout()
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=layout,
        rollout_orchestrator=StaticRollout(make_batch()),
    )

    _, metrics = trainer.step(make_batch())

    assert layout.reference_forward_calls == 0
    assert metrics["mean_token_kl"] == 0.0
    assert metrics["kl_loss"] == 0.0


def test_trainer_logs_clip_metrics() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        beta=0.0,
    )
    trainer = GRPOTrainer(
        config=config,
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        tokenizer=DummyTokenizer(),
        layout=TrainableLayout(),
        rollout_orchestrator=StaticRollout(make_batch()),
    )

    _, metrics = trainer.step(make_batch())

    assert "clip_ratio/region_mean" in metrics
    assert "clip_ratio/low_mean" in metrics
    assert "clip_ratio/high_mean" in metrics
    assert "mean_ratio" in metrics


def test_trainer_rejects_unimplemented_async_and_vllm_flags() -> None:
    base_kwargs = {
        "environment": MinimalEnvironment(),
        "verifier": MinimalVerifier(),
        "tokenizer": DummyTokenizer(),
        "layout": TrainableLayout(),
        "rollout_orchestrator": StaticRollout(make_batch()),
    }

    for flag_name in ("use_async_rollout_workers", "use_async_trajectory_copy", "experimental_vllm_rollout"):
        config = GRPOConfig(model_name="fake/model", **{flag_name: True})
        try:
            GRPOTrainer(config=config, **base_kwargs)
        except NotImplementedError:
            continue
        raise AssertionError(f"{flag_name} should raise NotImplementedError when enabled.")


def test_trainer_build_layout_passes_single_init_adapter_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class CapturingLayout(TrainableLayout):
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            super().__init__()

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(GRPOTrainer, "_build_tokenizer", lambda self: DummyTokenizer())
    monkeypatch.setattr("agentrl.core.trainer.SharedWeightLayout", CapturingLayout)

    init_adapter_path = "/tmp/init-adapter"
    trainer = GRPOTrainer(
        config=GRPOConfig(
            model_name="fake/model",
            steps=1,
            init_adapter_path=init_adapter_path,
            device="cpu",
        ),
        environment=MinimalEnvironment(),
        verifier=MinimalVerifier(),
        rollout_orchestrator=StaticRollout(make_batch()),
    )

    assert isinstance(trainer.layout, CapturingLayout)
    assert captured["adapter_path"] == init_adapter_path
