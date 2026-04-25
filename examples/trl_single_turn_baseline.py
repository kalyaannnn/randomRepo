"""TRL single-turn MBPP baseline for fair AgentRL comparison demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from examples.byod_task import build_mbpp_comparison_dataset

try:
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
except ImportError:
    GRPOConfig = None
    GRPOTrainer = None
    SFTConfig = None
    SFTTrainer = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TRL on the shared single-turn MBPP comparison task.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="./outputs/trl_single_turn")
    return parser


def build_result_stub() -> dict[str, Any]:
    return {
        "framework": "trl",
        "task_name": "mbpp_single_turn",
        "model_name": "fake/model",
        "seed": 0,
        "sft_epochs": 1,
        "steps": 1,
        "quality_metric": "mean_reward",
        "mean_reward": 0.0,
        "peak_vram_mb": 0.0,
        "wall_time_s": 0.0,
    }


def main(argv: list[str] | None = None, *, return_metrics: bool = False) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    _require_trl()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    dataset = build_mbpp_comparison_dataset(limit=args.limit, seed=args.seed)
    peft_config = _build_peft_config()
    sft_args = SFTConfig(
        output_dir=str(out_dir / "bootstrap"),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.sft_epochs,
        max_steps=-1,
        report_to=[],
        seed=args.seed,
    )
    sft_trainer = SFTTrainer(
        model=args.model,
        args=sft_args,
        train_dataset=dataset["sft"],
        peft_config=peft_config,
    )
    sft_trainer.train()

    grpo_args = GRPOConfig(
        output_dir=str(out_dir / "rl"),
        per_device_train_batch_size=args.batch_size * args.group_size,
        num_generations=args.group_size,
        max_completion_length=args.max_new_tokens,
        max_steps=args.steps,
        report_to=[],
        seed=args.seed,
    )
    grpo_trainer = GRPOTrainer(
        model=sft_trainer.model,
        args=grpo_args,
        train_dataset=dataset["rl"],
        reward_funcs=[dataset["reward_fn"]],
        peft_config=None,
    )
    train_result = grpo_trainer.train()
    wall_time_s = time.perf_counter() - started
    metrics = _summarize_trl_result(
        train_result=train_result,
        model_name=args.model,
        seed=args.seed,
        sft_epochs=args.sft_epochs,
        steps=args.steps,
        wall_time_s=wall_time_s,
    )
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics if return_metrics else None


def _require_trl() -> None:
    if any(item is None for item in (GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer)):
        raise SystemExit("TRL is required for this baseline. Install with `pip install trl`.")


def _build_peft_config() -> Any:
    try:
        from peft import LoraConfig
    except ImportError:
        return None
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def _summarize_trl_result(
    *,
    train_result: Any,
    model_name: str,
    seed: int,
    sft_epochs: int,
    steps: int,
    wall_time_s: float,
) -> dict[str, Any]:
    result_metrics = getattr(train_result, "metrics", {}) or {}
    mean_reward = float(
        result_metrics.get("train_reward", result_metrics.get("reward", result_metrics.get("mean_reward", 0.0)))
    )
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return {
        "framework": "trl",
        "task_name": "mbpp_single_turn",
        "model_name": model_name,
        "seed": seed,
        "sft_epochs": sft_epochs,
        "steps": steps,
        "quality_metric": "mean_reward",
        "mean_reward": mean_reward,
        "peak_vram_mb": peak_vram,
        "wall_time_s": wall_time_s,
    }


if __name__ == "__main__":
    main()
