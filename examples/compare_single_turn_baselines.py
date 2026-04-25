"""Compare AgentRL and TRL on the same single-turn SFT->GRPO pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.agentrl_single_turn_baseline import main as run_agentrl_main
from examples.trl_single_turn_baseline import main as run_trl_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched AgentRL and TRL single-turn baselines.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="./outputs/single_turn_compare")
    return parser


def run_agentrl(args: list[str]) -> dict[str, Any]:
    return run_agentrl_main(args, return_metrics=True) or {}


def run_trl(args: list[str]) -> dict[str, Any]:
    return run_trl_main(args, return_metrics=True) or {}


def main(argv: list[str] | None = None, *, return_metrics: bool = False) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_args = [
        "--model",
        args.model,
        "--limit",
        str(args.limit),
        "--seed",
        str(args.seed),
        "--sft-epochs",
        str(args.sft_epochs),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--group-size",
        str(args.group_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    agentrl_metrics = run_agentrl(shared_args + ["--output-dir", str(out_dir / "agentrl")])
    trl_metrics = run_trl(shared_args + ["--output-dir", str(out_dir / "trl")])
    comparison = {
        "claim_scope": {
            "fairness_track": "AgentRL standard rollout vs TRL after matched SFT bootstrap.",
            "systems_track": "AgentRL-only runtime modes show the systems moat.",
        },
        "agentrl": agentrl_metrics,
        "trl": trl_metrics,
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return comparison if return_metrics else None


if __name__ == "__main__":
    main()
