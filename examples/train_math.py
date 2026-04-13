"""Minimal end-to-end AgentRL training example."""

from __future__ import annotations

import argparse

from agentrl import GRPOConfig, GRPOTrainer
from examples.math_env import MathEnvironment, MathVerifier


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the math demo."""

    parser = argparse.ArgumentParser(description="Run AgentRL on the bundled math environment.")
    parser.add_argument("--model", required=True, help="Transformers model name or local path.")
    parser.add_argument("--steps", type=int, default=5, help="Number of GRPO update steps.")
    parser.add_argument("--batch-size", type=int, default=2, help="Prompts sampled per training step.")
    parser.add_argument("--group-size", type=int, default=4, help="Responses sampled per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens per turn.")
    parser.add_argument("--output-dir", default="./checkpoints", help="Directory for metrics and replay files.")
    parser.add_argument(
        "--split",
        default="smoke",
        choices=["smoke", "easy", "train", "eval"],
        help="Problem split to sample from.",
    )
    return parser


def main() -> None:
    """Create the trainer and run the math demo."""

    args = build_parser().parse_args()
    config = GRPOConfig(
        model_name=args.model,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        use_continuous_batching=False,
    )
    trainer = GRPOTrainer(
        config=config,
        environment=MathEnvironment(split=args.split),
        verifier=MathVerifier(),
    )
    trainer.train()


if __name__ == "__main__":
    main()
