"""Strict exact-match evaluation for a saved GSM8K adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentrl import GRPOConfig
from agentrl.memory.layout import SharedWeightLayout
from examples.gsm8k_subset import GSM8KSubsetEnvironment, GSM8KSubsetVerifier


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for GSM8K strict evaluation."""

    parser = argparse.ArgumentParser(
        description="Run strict exact-match evaluation on a real GSM8K subset.",
    )
    parser.add_argument("--model", required=True, help="Transformers model name or local path.")
    parser.add_argument(
        "--init-adapter-path",
        required=True,
        help="LoRA adapter directory produced by bootstrap or GRPO training.",
    )
    parser.add_argument(
        "--output-dir",
        default="./eval_gsm8k_subset",
        help="Directory to write the summary JSON and predictions JSONL.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="GSM8K split to evaluate.",
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generation budget per evaluation example.",
    )
    return parser


def main() -> None:
    """Run greedy strict evaluation and persist a compact report."""

    args = build_parser().parse_args()
    config = GRPOConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        use_continuous_batching=False,
        do_sample=False,
        temperature=0.0,
        init_adapter_path=args.init_adapter_path,
    )
    environment = GSM8KSubsetEnvironment(
        split=args.split,
        subset_size=args.subset_size,
        max_question_words=args.max_question_words,
        curriculum=args.curriculum,
    )
    verifier = GSM8KSubsetVerifier(reward_mode="strict")

    try:
        from peft import LoraConfig
        from transformers import AutoTokenizer
        import torch
    except ImportError as exc:
        raise ImportError(
            "eval_gsm8k_subset requires `transformers`, `peft`, and `torch` to be installed."
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
        adapter_path=args.init_adapter_path,
    )
    model = layout.model
    model.eval()
    model.config.use_cache = True

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    exact_matches = 0.0
    total = 0
    total_response_tokens = 0
    with predictions_path.open("w", encoding="utf-8") as handle:
        for problem in environment.problems():
            prompt = environment.render_prompt(tokenizer, problem.question)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(layout.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(layout.device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            response_ids = generated[:, input_ids.shape[-1] :]
            response = environment.postprocess_response(tokenizer.decode(response_ids[0], skip_special_tokens=True))
            reward = verifier.verify(
                response,
                {
                    "question": problem.question,
                    "answer": problem.answer,
                    "solution": problem.solution,
                    "split": args.split,
                    "curriculum": args.curriculum,
                    "dataset": environment.dataset_name,
                    "dataset_config_name": environment.dataset_config_name,
                },
            )
            record = {
                "question": problem.question,
                "answer": problem.answer,
                "prediction": response,
                "reward": reward,
            }
            handle.write(json.dumps(record) + "\n")
            exact_matches += reward
            total += 1
            total_response_tokens += int(response_ids.shape[-1])

    summary = {
        "model_name": args.model,
        "adapter_path": args.init_adapter_path,
        "split": args.split,
        "subset_size": total,
        "curriculum": args.curriculum,
        "max_question_words": args.max_question_words,
        "max_new_tokens": args.max_new_tokens,
        "exact_match_rate": (exact_matches / total) if total else 0.0,
        "mean_reward": (exact_matches / total) if total else 0.0,
        "mean_response_tokens": (total_response_tokens / total) if total else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
