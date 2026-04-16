"""Memory-aware runtime control for single-device AgentRL execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agentrl.generation.scheduler import dtype_bytes, estimate_kv_cache_bytes, kv_cache_geometry


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionController:
    """Track runtime risk and apply conservative fallback adjustments."""

    config: Any
    device: Any
    _oom_retries: int = 0
    _adjustments: int = 0
    _last_reason: str = "none"
    _high_padding_streak: int = 0
    _high_kv_pressure_streak: int = 0

    def __post_init__(self) -> None:
        self._oom_retries = 0
        self._adjustments = 0
        self._last_reason = "none"
        self._high_padding_streak = 0
        self._high_kv_pressure_streak = 0

    def build_preflight_report(
        self,
        startup_report: dict[str, float | str | None],
        model_config: Any | None,
    ) -> dict[str, float | str | None]:
        """Estimate rollout pressure and expose the initial execution policy."""

        report: dict[str, float | str | None] = {
            "execution_policy": self.config.execution_policy,
            "auto_tune_chunk_size": bool(self.config.auto_tune_chunk_size),
            "auto_reduce_on_oom": bool(self.config.auto_reduce_on_oom),
            "oom_retry_budget": float(self.config.oom_retry_budget),
            "runtime_adjustments": float(self._adjustments),
            "current_chunk_size": float(self.config.chunk_size or self.config.group_size),
            "current_prefill_chunk_size": float(self.config.prefill_chunk_size),
            "preflight_risk": "unknown",
            "estimated_kv_cache_mb": None,
            "kv_cache_fraction_of_free_vram": None,
            "expected_bottleneck": self._expected_bottleneck(),
            "execution_recommendation": "No preflight recommendation available.",
            "last_runtime_adjustment_reason": self._last_reason,
        }

        if model_config is None:
            return report

        try:
            estimate_mb = self._estimated_kv_cache_mb(model_config)
        except AttributeError:
            return report

        free_mb = startup_report.get("device_free_mb")
        fraction = None
        if isinstance(free_mb, float) and free_mb > 0:
            fraction = estimate_mb / free_mb

        report["estimated_kv_cache_mb"] = estimate_mb
        report["kv_cache_fraction_of_free_vram"] = fraction
        report["preflight_risk"] = self._risk_level(fraction)
        report["execution_recommendation"] = self._recommendation(fraction)
        return report

    def observe(self, metrics: dict[str, float]) -> dict[str, float | str]:
        """Adjust runtime knobs when observed headroom gets too tight."""

        if getattr(self.device, "type", None) != "cuda":
            return {
                "runtime_adjustments": float(self._adjustments),
                "runtime_low_headroom": 0.0,
                "dominant_runtime_bottleneck": self._classify_bottleneck(metrics),
                "runtime_recommendation": self._recommend_from_metrics(metrics),
                "scheduler_kv_pressure": self._scheduler_kv_pressure(metrics),
            }

        bottleneck = self._classify_bottleneck(metrics)
        recommendation = self._recommend_from_metrics(metrics)
        headroom = float(
            metrics.get(
                "rollout_runtime_headroom_mb",
                metrics.get("generation_runtime_headroom_mb", metrics.get("runtime_headroom_mb", 0.0)),
            )
        )
        if headroom >= self.config.min_runtime_headroom_mb:
            update = self._maybe_reduce_proactively(metrics)
            return {
                "runtime_adjustments": float(self._adjustments),
                "runtime_low_headroom": 0.0,
                "dominant_runtime_bottleneck": bottleneck,
                "runtime_recommendation": recommendation,
                "scheduler_kv_pressure": self._scheduler_kv_pressure(metrics),
                **update,
            }

        adjusted = False
        if self._reduce_chunk_size():
            adjusted = True
            self._last_reason = "low_headroom_chunk_size"
        elif self._reduce_prefill_chunk_size():
            adjusted = True
            self._last_reason = "low_headroom_prefill_chunk_size"

        if adjusted:
            LOGGER.warning(
                "Runtime headroom is low (%.1f MB < %.1f MB). Reduced settings to chunk_size=%s, "
                "prefill_chunk_size=%s.",
                headroom,
                self.config.min_runtime_headroom_mb,
                self.config.chunk_size,
                self.config.prefill_chunk_size,
            )

        return {
            "runtime_adjustments": float(self._adjustments),
            "runtime_low_headroom": 1.0,
            "last_runtime_adjustment_reason": self._last_reason,
            "active_chunk_size": float(self.config.chunk_size or self.config.group_size),
            "active_prefill_chunk_size": float(self.config.prefill_chunk_size),
            "dominant_runtime_bottleneck": bottleneck,
            "runtime_recommendation": recommendation,
            "scheduler_kv_pressure": self._scheduler_kv_pressure(metrics),
        }

    def handle_oom(self, stage: str) -> bool:
        """Apply a conservative runtime fallback after an out-of-memory failure."""

        if not self.config.auto_reduce_on_oom:
            return False
        if self._oom_retries >= self.config.oom_retry_budget:
            return False

        adjusted = self._reduce_chunk_size() or self._reduce_prefill_chunk_size()
        if not adjusted:
            return False

        self._oom_retries += 1
        self._last_reason = f"oom_{stage}"
        LOGGER.warning(
            "OOM during %s. Retrying with chunk_size=%s, prefill_chunk_size=%s (%s/%s retries used).",
            stage,
            self.config.chunk_size,
            self.config.prefill_chunk_size,
            self._oom_retries,
            self.config.oom_retry_budget,
        )
        return True

    def _expected_bottleneck(self) -> str:
        if self.config.max_new_tokens >= self.config.prefill_chunk_size:
            return "decode"
        if self.config.batch_size * self.config.group_size >= 8:
            return "padding"
        return "prefill"

    def _classify_bottleneck(self, metrics: dict[str, float]) -> str:
        padding_ratio = float(metrics.get("padding_ratio", 0.0))
        generation_padding_ratio = float(metrics.get("generation_padding_ratio", 0.0))
        prefill_time_ms = float(metrics.get("prefill_time_ms", 0.0))
        decode_time_ms = float(metrics.get("decode_time_ms", 0.0))
        cache_reuse = float(metrics.get("cache_reuse_effectiveness", 0.0))
        scheduler_pressure = self._scheduler_kv_pressure(metrics)

        if max(padding_ratio, generation_padding_ratio) >= 0.35:
            return "padding"
        if scheduler_pressure >= 0.9:
            return "kv_budget"
        if decode_time_ms > prefill_time_ms * 1.5:
            if cache_reuse < 0.3:
                return "decode_without_cache_reuse"
            return "decode"
        if prefill_time_ms > decode_time_ms * 1.5:
            return "prefill"
        return "balanced"

    def _recommend_from_metrics(self, metrics: dict[str, float]) -> str:
        bottleneck = self._classify_bottleneck(metrics)
        scheduler_pressure = self._scheduler_kv_pressure(metrics)
        if bottleneck == "kv_budget":
            return (
                "Scheduler is near its KV-cache budget; reduce chunk_size, max_new_tokens, "
                "or prompt length before scaling up."
            )
        if bottleneck == "padding":
            return "Padding waste is high; reduce chunk_size or group together similar prompt lengths."
        if bottleneck == "decode_without_cache_reuse":
            return "Decode dominates with weak cache reuse; prefer continuous batching or shorter max_new_tokens."
        if bottleneck == "decode":
            return "Decode dominates; reduce max_new_tokens or scale group_size conservatively."
        if bottleneck == "prefill":
            return "Prefill dominates; trim prompts, lower max_prompt_tokens, or reduce prefill_chunk_size."
        return "Runtime phases look balanced for the current workload."

    def _maybe_reduce_proactively(self, metrics: dict[str, float]) -> dict[str, float | str]:
        """Tighten runtime knobs preemptively when pressure persists on CUDA runs."""

        kv_update = self._maybe_reduce_for_kv_pressure(metrics)
        if float(kv_update.get("runtime_adjustments", 0.0)) > 0.0:
            return kv_update
        padding_update = self._maybe_reduce_for_padding(metrics)
        return {
            **kv_update,
            **padding_update,
        }

    def _maybe_reduce_for_kv_pressure(self, metrics: dict[str, float]) -> dict[str, float | str]:
        """Reduce scheduler breadth when KV pressure remains near the configured budget."""

        scheduler_pressure = self._scheduler_kv_pressure(metrics)
        if scheduler_pressure >= 0.9:
            self._high_kv_pressure_streak += 1
        else:
            self._high_kv_pressure_streak = 0

        if self._high_kv_pressure_streak < 2:
            return {
                "kv_pressure_streak": float(self._high_kv_pressure_streak),
            }

        if self._reduce_chunk_size():
            self._last_reason = "high_kv_pressure_chunk_size"
            self._high_kv_pressure_streak = 0
            LOGGER.warning(
                "Scheduler KV pressure stayed high across steps. Reduced chunk_size to %s to lower admission cost.",
                self.config.chunk_size,
            )
            return {
                "kv_pressure_streak": 0.0,
                "runtime_adjustments": float(self._adjustments),
                "last_runtime_adjustment_reason": self._last_reason,
                "active_chunk_size": float(self.config.chunk_size or self.config.group_size),
            }

        if self._reduce_prefill_chunk_size():
            self._last_reason = "high_kv_pressure_prefill_chunk_size"
            self._high_kv_pressure_streak = 0
            LOGGER.warning(
                "Scheduler KV pressure stayed high across steps. Reduced prefill_chunk_size to %s.",
                self.config.prefill_chunk_size,
            )
            return {
                "kv_pressure_streak": 0.0,
                "runtime_adjustments": float(self._adjustments),
                "last_runtime_adjustment_reason": self._last_reason,
                "active_chunk_size": float(self.config.chunk_size or self.config.group_size),
                "active_prefill_chunk_size": float(self.config.prefill_chunk_size),
            }

        return {
            "kv_pressure_streak": float(self._high_kv_pressure_streak),
        }

    def _maybe_reduce_for_padding(self, metrics: dict[str, float]) -> dict[str, float | str]:
        """Tighten chunking when padding waste remains high on CUDA runs."""

        padding_ratio = max(
            float(metrics.get("padding_ratio", 0.0)),
            float(metrics.get("generation_padding_ratio", 0.0)),
        )
        if padding_ratio >= 0.35:
            self._high_padding_streak += 1
        else:
            self._high_padding_streak = 0

        if self._high_padding_streak < 2:
            return {
                "padding_pressure_streak": float(self._high_padding_streak),
            }
        if not self._reduce_chunk_size():
            return {
                "padding_pressure_streak": float(self._high_padding_streak),
            }

        self._last_reason = "high_padding_chunk_size"
        self._high_padding_streak = 0
        LOGGER.warning(
            "Padding ratio stayed high across steps. Reduced chunk_size to %s to cut redundant decode work.",
            self.config.chunk_size,
        )
        return {
            "padding_pressure_streak": 0.0,
            "runtime_adjustments": float(self._adjustments),
            "last_runtime_adjustment_reason": self._last_reason,
            "active_chunk_size": float(self.config.chunk_size or self.config.group_size),
        }

    def _estimated_kv_cache_mb(self, model_config: Any) -> float:
        num_layers, num_heads, head_dim = kv_cache_geometry(model_config)
        chunk_size = int(self.config.chunk_size or self.config.group_size)
        estimate = estimate_kv_cache_bytes(
            batch_size=int(self.config.batch_size),
            group_size=chunk_size,
            max_new_tokens=int(self.config.max_new_tokens),
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=int(head_dim),
            dtype_bytes=dtype_bytes(self.config.dtype),
        )
        return estimate / (1024 * 1024)

    def _scheduler_kv_pressure(self, metrics: dict[str, float]) -> float:
        prefill_pressure = float(metrics.get("scheduler_prefill_kv_pressure", 0.0))
        decode_pressure = float(metrics.get("scheduler_decode_kv_pressure", 0.0))
        return max(prefill_pressure, decode_pressure)

    def _risk_level(self, fraction: float | None) -> str:
        if fraction is None:
            return "unknown"
        if self.config.execution_policy == "safe":
            if fraction >= 0.4:
                return "high"
            if fraction >= 0.2:
                return "medium"
            return "low"
        if fraction >= 0.7:
            return "high"
        if fraction >= 0.4:
            return "medium"
        return "low"

    def _recommendation(self, fraction: float | None) -> str:
        if fraction is None:
            return "CUDA free-memory data unavailable; keeping conservative defaults."
        if fraction >= 0.7:
            return "Reduce group_size or max_new_tokens before scaling up."
        if fraction >= 0.4:
            return "Prefer continuous batching with conservative chunking."
        return "Current rollout shape looks safe on the detected device."

    def _reduce_chunk_size(self) -> bool:
        current = int(self.config.chunk_size or self.config.group_size)
        if current <= 1:
            return False
        reduced = max(1, current // 2)
        if reduced == current:
            reduced = current - 1
        self.config.chunk_size = reduced
        self._adjustments += 1
        return True

    def _reduce_prefill_chunk_size(self) -> bool:
        current = int(self.config.prefill_chunk_size)
        if current <= 32:
            return False
        reduced = max(32, current // 2)
        if reduced == current:
            return False
        self.config.prefill_chunk_size = reduced
        self._adjustments += 1
        return True

    def _require_attr(self, obj: Any, name: str) -> Any:
        value = getattr(obj, name, None)
        if value is None:
            raise AttributeError(f"model_config must define `{name}`.")
        return value
