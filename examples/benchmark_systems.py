"""Systems-focused runtime benchmark for AgentRL."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
from statistics import mean

import torch

from agentrl import GRPOConfig, GRPOTrainer
from examples.math_env import MathEnvironment, MathVerifier


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the systems benchmark."""

    parser = argparse.ArgumentParser(
        description="Run a fixed AgentRL workload and summarize runtime metrics.",
    )
    parser.add_argument("--model", required=True, help="Transformers model name or local path.")
    parser.add_argument("--steps", type=int, default=5, help="Number of benchmarked training steps.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts sampled per training step.")
    parser.add_argument("--group-size", type=int, default=4, help="Responses sampled per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens per turn.")
    parser.add_argument(
        "--split",
        default="easy",
        choices=["smoke", "easy", "train", "eval"],
        help="Math benchmark split to sample from.",
    )
    parser.add_argument(
        "--output-dir",
        default="./systems_benchmark",
        help="Directory to write the runtime summary.",
    )
    parser.add_argument(
        "--compare-standard-vs-continuous",
        action="store_true",
        help="Run both rollout modes on the same workload and emit a comparison summary.",
    )
    return parser


def _hardware_string() -> str:
    """Describe the current execution device briefly."""

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"cpu:{platform.processor() or platform.machine()}"


def _config_hash(config: GRPOConfig) -> str:
    """Hash a stable subset of config fields for reproducibility notes."""

    payload = {
        "model_name": config.model_name,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "max_new_tokens": config.max_new_tokens,
        "split": getattr(config, "split", None),
        "use_continuous_batching": config.use_continuous_batching,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _summarize_run(history: list[dict[str, float]], config: GRPOConfig, split: str) -> dict[str, object]:
    """Aggregate a per-step history into a compact systems summary."""

    generation_fractions = [float(row.get("generation_time_ms", 0.0)) / max(float(row.get("total_step_time_ms", 1e-6)), 1e-6) for row in history]
    training_fractions = [float(row.get("training_time_ms", 0.0)) / max(float(row.get("total_step_time_ms", 1e-6)), 1e-6) for row in history]
    bottleneck_counts = Counter(str(row.get("dominant_runtime_bottleneck", "unknown")) for row in history)
    adjustment_reason_counts = Counter(
        str(row.get("last_runtime_adjustment_reason", "none"))
        for row in history
        if str(row.get("last_runtime_adjustment_reason", "none")) != "none"
    )
    recommendation_counts = Counter(str(row.get("runtime_recommendation", "")) for row in history if row.get("runtime_recommendation"))
    dominant_bottleneck = bottleneck_counts.most_common(1)[0][0] if bottleneck_counts else "unknown"
    top_recommendation = recommendation_counts.most_common(1)[0][0] if recommendation_counts else ""
    efficiency_diagnosis = _diagnose_run(history, dominant_bottleneck, adjustment_reason_counts)
    benchmark_verdict = _single_run_verdict(
        use_continuous_batching=config.use_continuous_batching,
        efficiency_diagnosis=efficiency_diagnosis,
        dominant_bottleneck=dominant_bottleneck,
        steps_with_runtime_adjustment=sum(1 for row in history if float(row.get("runtime_adjustments", 0.0)) > 0.0),
        top_runtime_recommendation=top_recommendation,
    )
    return {
        "model_name": config.model_name,
        "split": split,
        "hardware": _hardware_string(),
        "config_hash": _config_hash(config),
        "steps": len(history),
        "use_continuous_batching": config.use_continuous_batching,
        "mean_step_time_ms": mean(float(row.get("total_step_time_ms", 0.0)) for row in history),
        "mean_generation_fraction": mean(generation_fractions),
        "mean_training_fraction": mean(training_fractions),
        "mean_tokens_per_second": mean(float(row.get("tokens_per_second", 0.0)) for row in history),
        "mean_prefill_tokens_per_second": mean(float(row.get("prefill_tokens_per_second", 0.0)) for row in history),
        "mean_decode_tokens_per_second": mean(float(row.get("decode_tokens_per_second", 0.0)) for row in history),
        "mean_padding_ratio": mean(float(row.get("padding_ratio", 0.0)) for row in history),
        "mean_generation_padding_ratio": mean(float(row.get("generation_padding_ratio", 0.0)) for row in history),
        "mean_sequence_padding_ratio": mean(float(row.get("sequence_padding_ratio", 0.0)) for row in history),
        "mean_cache_reuse_effectiveness": mean(float(row.get("cache_reuse_effectiveness", 0.0)) for row in history),
        "mean_scheduler_prefill_passes": mean(float(row.get("scheduler_prefill_passes", 0.0)) for row in history),
        "mean_scheduler_decode_passes": mean(float(row.get("scheduler_decode_passes", 0.0)) for row in history),
        "mean_scheduler_prefill_kv_budget_mb": mean(
            float(row.get("scheduler_prefill_kv_budget_mb", 0.0)) for row in history
        ),
        "mean_scheduler_decode_kv_budget_mb": mean(
            float(row.get("scheduler_decode_kv_budget_mb", 0.0)) for row in history
        ),
        "mean_scheduler_prefill_admitted_kv_mb": mean(
            float(row.get("scheduler_prefill_admitted_kv_mb", 0.0)) for row in history
        ),
        "mean_scheduler_decode_admitted_kv_mb": mean(
            float(row.get("scheduler_decode_admitted_kv_mb", 0.0)) for row in history
        ),
        "mean_scheduler_kv_pressure": mean(
            max(
                float(row.get("scheduler_prefill_kv_pressure", 0.0)),
                float(row.get("scheduler_decode_kv_pressure", 0.0)),
            )
            for row in history
        ),
        "mean_scheduler_deferred_sequences": mean(
            float(row.get("scheduler_deferred_sequences", 0.0)) for row in history
        ),
        "mean_scheduler_max_concurrent_sequences": mean(
            float(row.get("scheduler_max_concurrent_sequences", 0.0)) for row in history
        ),
        "peak_vram_mb": max(float(row.get("peak_vram_mb", 0.0)) for row in history),
        "rollout_peak_vram_mb": max(float(row.get("rollout_peak_vram_mb", 0.0)) for row in history),
        "min_rollout_runtime_headroom_mb": min(float(row.get("rollout_runtime_headroom_mb", 0.0)) for row in history),
        "mean_runtime_adjustments": mean(float(row.get("runtime_adjustments", 0.0)) for row in history),
        "steps_with_runtime_adjustment": sum(1 for row in history if float(row.get("runtime_adjustments", 0.0)) > 0.0),
        "steps_with_low_headroom": sum(1 for row in history if float(row.get("runtime_low_headroom", 0.0)) > 0.0),
        "dominant_runtime_bottleneck": dominant_bottleneck,
        "bottleneck_step_counts": dict(sorted(bottleneck_counts.items())),
        "runtime_adjustment_reason_counts": dict(sorted(adjustment_reason_counts.items())),
        "top_runtime_recommendation": top_recommendation,
        "efficiency_diagnosis": efficiency_diagnosis,
        "benchmark_verdict": benchmark_verdict,
    }


def _diagnose_run(
    history: list[dict[str, float]],
    dominant_bottleneck: str,
    adjustment_reason_counts: Counter[str],
) -> str:
    """Return a compact run-level diagnosis from per-step controller signals."""

    if any("oom_" in reason for reason in adjustment_reason_counts):
        return "memory-constrained but recoverable"
    if dominant_bottleneck == "padding":
        return "padding-limited"
    if dominant_bottleneck == "kv_budget":
        return "kv-budget-limited"
    if dominant_bottleneck in {"decode", "decode_without_cache_reuse"}:
        return "decode-limited"
    if dominant_bottleneck == "prefill":
        return "prefill-limited"
    if any(float(row.get("runtime_adjustments", 0.0)) > 0.0 for row in history):
        return "runtime-adjusted"
    return "balanced"


def _single_run_verdict(
    *,
    use_continuous_batching: bool,
    efficiency_diagnosis: str,
    dominant_bottleneck: str,
    steps_with_runtime_adjustment: int,
    top_runtime_recommendation: str,
) -> str:
    """Render a short human-readable verdict for one run."""

    mode = "continuous batching" if use_continuous_batching else "standard rollout"
    verdict = f"{mode} was {efficiency_diagnosis}"
    if dominant_bottleneck != "balanced":
        verdict += f", dominated by {dominant_bottleneck.replace('_', ' ')}"
    if steps_with_runtime_adjustment > 0:
        verdict += f", and needed {steps_with_runtime_adjustment} runtime adjustment"
        if steps_with_runtime_adjustment != 1:
            verdict += "s"
    verdict += "."
    if top_runtime_recommendation:
        verdict += f" {top_runtime_recommendation}"
    return verdict


def _comparison_verdict(summaries: list[dict[str, object]]) -> str:
    """Render a compact head-to-head conclusion for benchmark comparisons."""

    if len(summaries) != 2:
        return "Comparison verdict unavailable."

    ordered = {("continuous" if summary["use_continuous_batching"] else "standard"): summary for summary in summaries}
    standard = ordered.get("standard")
    continuous = ordered.get("continuous")
    if standard is None or continuous is None:
        return "Comparison verdict unavailable."

    faster = continuous if float(continuous["mean_step_time_ms"]) < float(standard["mean_step_time_ms"]) else standard
    slower = standard if faster is continuous else continuous
    faster_name = "continuous batching" if faster is continuous else "standard rollout"
    slower_name = "standard rollout" if faster is continuous else "continuous batching"

    verdict = (
        f"{faster_name} was faster on mean step time than {slower_name}"
        f" ({float(faster['mean_step_time_ms']):.2f} ms vs {float(slower['mean_step_time_ms']):.2f} ms)"
        f" but was {faster['efficiency_diagnosis']}."
    )
    if int(faster["steps_with_runtime_adjustment"]) > 0:
        verdict += (
            f" It needed {int(faster['steps_with_runtime_adjustment'])} runtime adjustment"
            f"{'' if int(faster['steps_with_runtime_adjustment']) == 1 else 's'}."
        )
    elif int(slower["steps_with_runtime_adjustment"]) == 0:
        verdict += " Both modes stayed stable without runtime adjustment."
    return verdict


def _run_one(args: argparse.Namespace, use_continuous_batching: bool, output_dir: Path) -> dict[str, object]:
    """Run one systems benchmark configuration and write its summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = GRPOConfig(
        model_name=args.model,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=str(output_dir),
        use_continuous_batching=use_continuous_batching,
    )
    trainer = GRPOTrainer(
        config=config,
        environment=MathEnvironment(split=args.split),
        verifier=MathVerifier(),
    )
    history = trainer.train()
    summary = _summarize_run(history, config=config, split=args.split)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _render_comparison_table(summaries: list[dict[str, object]]) -> str:
    """Render a tiny markdown comparison table for stdout."""

    lines = [
        "| Mode | Mean step ms | Tokens/s | Padding ratio | KV pressure | Deferred seqs | Max concurrent | Runtime diagnosis | Bottleneck | Adjusted steps | Peak VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for summary in summaries:
        mode = "continuous" if summary["use_continuous_batching"] else "standard"
        lines.append(
            "| "
            f"{mode} | {summary['mean_step_time_ms']:.2f} | {summary['mean_tokens_per_second']:.2f} | "
            f"{summary['mean_padding_ratio']:.4f} | {summary['mean_scheduler_kv_pressure']:.2f} | "
            f"{summary['mean_scheduler_deferred_sequences']:.2f} | "
            f"{summary['mean_scheduler_max_concurrent_sequences']:.2f} | {summary['efficiency_diagnosis']} | "
            f"{summary['dominant_runtime_bottleneck']} | {summary['steps_with_runtime_adjustment']} | "
            f"{summary['peak_vram_mb']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the systems benchmark and persist a compact summary."""

    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_standard_vs_continuous:
        standard_summary = _run_one(args, use_continuous_batching=False, output_dir=output_dir / "standard")
        continuous_summary = _run_one(args, use_continuous_batching=True, output_dir=output_dir / "continuous")
        comparison = {
            "hardware": _hardware_string(),
            "split": args.split,
            "model_name": args.model,
            "runs": [standard_summary, continuous_summary],
            "comparison_verdict": _comparison_verdict([standard_summary, continuous_summary]),
        }
        (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        print(_render_comparison_table([standard_summary, continuous_summary]))
        print()
        print(comparison["comparison_verdict"])
        return

    summary = _run_one(args, use_continuous_batching=True, output_dir=output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
