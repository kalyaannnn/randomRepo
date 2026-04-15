from __future__ import annotations

from types import SimpleNamespace

from agentrl.core.config import GRPOConfig
from agentrl.runtime.controller import ExecutionController


def test_execution_controller_builds_preflight_report() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=2,
        group_size=8,
        max_new_tokens=256,
        chunk_size=4,
        execution_policy="safe",
    )
    controller = ExecutionController(config=config, device=SimpleNamespace(type="cuda"))
    startup_report = {"device_free_mb": 512.0}
    model_config = SimpleNamespace(
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_size=2048,
    )

    report = controller.build_preflight_report(startup_report, model_config)

    assert report["execution_policy"] == "safe"
    assert report["estimated_kv_cache_mb"] is not None
    assert report["kv_cache_fraction_of_free_vram"] is not None
    assert report["preflight_risk"] in {"medium", "high"}


def test_execution_controller_reduces_chunk_size_on_low_headroom() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        group_size=8,
        chunk_size=8,
        min_runtime_headroom_mb=1024.0,
    )
    controller = ExecutionController(config=config, device=SimpleNamespace(type="cuda"))

    metrics = controller.observe({"rollout_runtime_headroom_mb": 256.0})

    assert config.chunk_size == 4
    assert metrics["runtime_adjustments"] == 1.0
    assert metrics["runtime_low_headroom"] == 1.0
    assert metrics["last_runtime_adjustment_reason"] == "low_headroom_chunk_size"


def test_execution_controller_reports_decode_recommendation() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        use_continuous_batching=False,
    )
    controller = ExecutionController(config=config, device=SimpleNamespace(type="cpu"))

    metrics = controller.observe(
        {
            "prefill_time_ms": 10.0,
            "decode_time_ms": 40.0,
            "cache_reuse_effectiveness": 0.1,
            "padding_ratio": 0.05,
        }
    )

    assert metrics["dominant_runtime_bottleneck"] == "decode_without_cache_reuse"
    assert "continuous batching" in str(metrics["runtime_recommendation"])


def test_execution_controller_reduces_chunk_size_after_padding_streak() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        group_size=8,
        chunk_size=8,
        min_runtime_headroom_mb=1.0,
    )
    controller = ExecutionController(config=config, device=SimpleNamespace(type="cuda"))

    first = controller.observe(
        {
            "rollout_runtime_headroom_mb": 2048.0,
            "padding_ratio": 0.45,
            "generation_padding_ratio": 0.45,
            "prefill_time_ms": 10.0,
            "decode_time_ms": 12.0,
            "cache_reuse_effectiveness": 0.8,
        }
    )
    second = controller.observe(
        {
            "rollout_runtime_headroom_mb": 2048.0,
            "padding_ratio": 0.50,
            "generation_padding_ratio": 0.50,
            "prefill_time_ms": 10.0,
            "decode_time_ms": 12.0,
            "cache_reuse_effectiveness": 0.8,
        }
    )

    assert first["padding_pressure_streak"] == 1.0
    assert config.chunk_size == 4
    assert second["last_runtime_adjustment_reason"] == "high_padding_chunk_size"
    assert second["dominant_runtime_bottleneck"] == "padding"


def test_execution_controller_reports_kv_budget_pressure() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        use_continuous_batching=True,
    )
    controller = ExecutionController(config=config, device=SimpleNamespace(type="cpu"))

    metrics = controller.observe(
        {
            "prefill_time_ms": 10.0,
            "decode_time_ms": 12.0,
            "cache_reuse_effectiveness": 0.8,
            "padding_ratio": 0.05,
            "scheduler_prefill_kv_pressure": 0.25,
            "scheduler_decode_kv_pressure": 0.96,
        }
    )

    assert metrics["dominant_runtime_bottleneck"] == "kv_budget"
    assert metrics["scheduler_kv_pressure"] == 0.96
    assert "KV-cache budget" in str(metrics["runtime_recommendation"])
