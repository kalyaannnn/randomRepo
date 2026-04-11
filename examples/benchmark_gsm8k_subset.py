"""Benchmark runner for the bundled GSM8K-style subset."""

from __future__ import annotations

import argparse

from agentrl import GRPOConfig, GRPOTrainer
from examples.gsm8k_subset import GSM8KSubsetEnvironment, GSM8KSubsetVerifier


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the GSM8K-style benchmark runner."""

    parser = argparse.ArgumentParser(
        description="Run AgentRL on the bundled GSM8K-style benchmark subset."
    )
    parser.add_argument("--model", required=True, help="Transformers model name or local path.")
    parser.add_argument("--steps", type=int, default=10, help="Number of GRPO update steps.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts sampled per training step.")
    parser.add_argument("--group-size", type=int, default=4, help="Responses sampled per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens per turn.")
    parser.add_argument(
        "--output-dir",
        default="./checkpoints_gsm8k_subset",
        help="Directory for metrics and replay files.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Benchmark subset split to sample from.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=128,
        help="Number of filtered GSM8K examples to retain in the benchmark subset.",
    )
    parser.add_argument(
        "--max-question-words",
        type=int,
        default=45,
        help="Keep only GSM8K questions up to this word count.",
    )
    parser.add_argument(
        "--curriculum",
        default="easy",
        choices=["easy", "standard"],
        help="Use an easy-ranked GSM8K subset or the first standard filtered slice.",
    )
    parser.add_argument(
        "--replay-every",
        type=int,
        default=1,
        help="Serialize trajectories every N steps for benchmark inspection.",
    )
    return parser


def main() -> None:
    """Create the trainer and run the bundled GSM8K-style benchmark."""

    args = build_parser().parse_args()
    config = GRPOConfig(
        model_name=args.model,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        replay_every=args.replay_every,
    )
    trainer = GRPOTrainer(
        config=config,
        environment=GSM8KSubsetEnvironment(
            split=args.split,
            subset_size=args.subset_size,
            max_question_words=args.max_question_words,
            curriculum=args.curriculum,
        ),
        verifier=GSM8KSubsetVerifier(),
    )
    trainer.train()


if __name__ == "__main__":
    main()
