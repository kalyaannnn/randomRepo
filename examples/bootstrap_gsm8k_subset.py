"""Bootstrap a LoRA adapter on a real GSM8K subset before GRPO."""

from __future__ import annotations

import argparse

from agentrl import GRPOConfig, SFTBootstrapTrainer
from agentrl.memory.layout import SharedWeightLayout
from examples.gsm8k_subset import GSM8KSubsetEnvironment


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for GSM8K bootstrap runs."""

    parser = argparse.ArgumentParser(
        description="Run adapter-only SFT bootstrap on a real GSM8K subset.",
    )
    parser.add_argument("--model", required=True, help="Transformers model name or local path.")
    parser.add_argument("--epochs", type=int, default=1, help="Full SFT passes over the subset.")
    parser.add_argument("--batch-size", type=int, default=4, help="Supervised batch size.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum prompt-plus-target token length retained during SFT.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Adapter learning rate for the supervised warm start.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="./bootstrap_gsm8k_adapter",
        help="Directory to save the trained LoRA adapter.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="GSM8K split to bootstrap on.",
    )
    parser.add_argument("--subset-size", type=int, default=128, help="Retained GSM8K examples.")
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
        help="Subset ranking policy before truncation.",
    )
    return parser


def main() -> None:
    """Run bootstrap SFT and save the resulting adapter."""

    args = build_parser().parse_args()
    config = GRPOConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_seq_length,
        lr=args.lr,
        steps=max(1, args.epochs),
        output_dir=args.adapter_dir,
        use_continuous_batching=False,
        use_speculative_decoding=False,
    )
    environment = GSM8KSubsetEnvironment(
        split=args.split,
        subset_size=args.subset_size,
        max_question_words=args.max_question_words,
        curriculum=args.curriculum,
    )

    try:
        from peft import LoraConfig
        from transformers import AutoTokenizer
        import torch
    except ImportError as exc:
        raise ImportError(
            "bootstrap_gsm8k_subset requires `transformers` and `peft` to be installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=config.trust_remote_code,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = config.device
    if device in {None, "auto"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    layout = SharedWeightLayout(
        model_name=args.model,
        lora_config=LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
        dtype=config.dtype,
        device=device,
        trust_remote_code=config.trust_remote_code,
        sdpa_backend=config.sdpa_backend,
    )
    trainer = SFTBootstrapTrainer(config=config, tokenizer=tokenizer, layout=layout)
    history = trainer.train(environment.supervised_samples(tokenizer=tokenizer), epochs=args.epochs)
    output_dir = trainer.save_adapter(args.adapter_dir)
    final_loss = history[-1]["loss"] if history else float("nan")
    print(f"Saved adapter to {output_dir}")
    print(f"Final bootstrap loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
