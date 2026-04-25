"""AgentRL single-turn MBPP baseline for fair TRL comparison demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import time
from typing import Any

import torch

from agentrl import GRPOConfig, GRPOTrainer, SFTBootstrapTrainer
from agentrl.memory.layout import SharedWeightLayout
from examples.byod_task import build_mbpp_comparison_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentRL on the shared single-turn MBPP comparison task.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="./outputs/agentrl_single_turn")
    return parser


def run_bootstrap(*, task: Any, config: GRPOConfig, output_dir: Path) -> str:
    """Run AgentRL adapter-only SFT bootstrap and return the saved adapter path."""

    try:
        from peft import LoraConfig
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("AgentRL bootstrap requires `transformers` and `peft`.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    layout = SharedWeightLayout(
        model_name=config.model_name,
        lora_config=lora_config,
        dtype=config.dtype,
        device=config.device,
        trust_remote_code=config.trust_remote_code,
        sdpa_backend=config.sdpa_backend,
    )
    trainer = SFTBootstrapTrainer(config=config, tokenizer=tokenizer, layout=layout)
    trainer.train(task.supervised_samples(tokenizer=tokenizer), epochs=config.steps)
    adapter_dir = output_dir / "adapter"
    return str(trainer.save_adapter(adapter_dir))


def build_result_stub() -> dict[str, Any]:
    return {
        "framework": "agentrl",
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    task = build_mbpp_comparison_task(limit=args.limit, seed=args.seed)
    bootstrap_config = GRPOConfig(
        model_name=args.model,
        steps=args.sft_epochs,
        batch_size=max(1, args.batch_size),
        max_new_tokens=args.max_new_tokens,
        output_dir=str(out_dir / "bootstrap"),
        use_continuous_batching=False,
        seed=args.seed,
    )
    adapter_path = run_bootstrap(task=task, config=bootstrap_config, output_dir=out_dir / "bootstrap")

    rl_config = GRPOConfig(
        model_name=args.model,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=str(out_dir / "rl"),
        init_adapter_path=adapter_path,
        use_continuous_batching=False,
        seed=args.seed,
    )
    trainer = GRPOTrainer(config=rl_config, environment=task.environment, verifier=task.verifier)
    history = trainer.train() or []
    wall_time_s = time.perf_counter() - started
    metrics = _summarize_history(
        history=history,
        framework="agentrl",
        model_name=args.model,
        seed=args.seed,
        sft_epochs=args.sft_epochs,
        steps=args.steps,
        wall_time_s=wall_time_s,
    )
    metrics["adapter_path"] = adapter_path
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics if return_metrics else None


def _summarize_history(
    *,
    history: list[dict[str, Any]],
    framework: str,
    model_name: str,
    seed: int,
    sft_epochs: int,
    steps: int,
    wall_time_s: float,
) -> dict[str, Any]:
    rewards = [float(row.get("mean_reward", 0.0)) for row in history]
    peak_vram = max([float(row.get("peak_vram_mb", 0.0)) for row in history] or [0.0])
    if torch.cuda.is_available():
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / (1024 * 1024))
    return {
        "framework": framework,
        "task_name": "mbpp_single_turn",
        "model_name": model_name,
        "seed": seed,
        "sft_epochs": sft_epochs,
        "steps": steps,
        "quality_metric": "mean_reward",
        "mean_reward": mean(rewards) if rewards else 0.0,
        "peak_vram_mb": peak_vram,
        "wall_time_s": wall_time_s,
    }


if __name__ == "__main__":
    main()
