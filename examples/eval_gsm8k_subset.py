"""Strict exact-match evaluation for a saved GSM8K adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
        default=96,
        help="Maximum generation budget per evaluation example.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of completions to draw per prompt. Values > 1 enable sampling diagnostics.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=None,
        help="K to use for pass@k. Defaults to --num-samples.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature used when --num-samples > 1.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus threshold used when --num-samples > 1.",
    )
    return parser


def _build_env_state(
    environment: GSM8KSubsetEnvironment,
    problem: Any,
    split: str,
    curriculum: str,
) -> dict[str, int | str]:
    """Build verifier state for one GSM8K problem."""

    return {
        "question": problem.question,
        "answer": problem.answer,
        "solution": problem.solution,
        "split": split,
        "curriculum": curriculum,
        "dataset": environment.dataset_name,
        "dataset_config_name": environment.dataset_config_name,
    }


def _response_token_lengths(
    response_ids: Any,
    pad_token_id: int | None,
) -> list[int]:
    """Estimate generated token counts per sampled response."""

    if response_ids.ndim != 2:
        raise ValueError("response_ids must be rank-2 [num_samples, seq_len].")
    if pad_token_id is None:
        return [int(response_ids.shape[-1])] * int(response_ids.shape[0])
    mask = response_ids.ne(pad_token_id)
    return [int(count.item()) for count in mask.sum(dim=-1)]


def main() -> None:
    """Run greedy strict evaluation and persist a compact report."""

    args = build_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0.")
    effective_pass_k = args.num_samples if args.pass_k is None else args.pass_k
    if effective_pass_k <= 0:
        raise ValueError("--pass-k must be > 0.")
    if effective_pass_k > args.num_samples:
        raise ValueError("--pass-k cannot exceed --num-samples.")
    do_sample = args.num_samples > 1
    if do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be > 0 when --num-samples > 1.")

    config = GRPOConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        use_continuous_batching=False,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else 0.0,
        top_p=args.top_p if do_sample else 1.0,
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

    pass_at_1_total = 0.0
    pass_at_k_total = 0.0
    any_correct_total = 0.0
    total = 0
    total_reward = 0.0
    total_response_tokens = 0
    total_sampled_responses = 0
    total_hit_token_cap = 0
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
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "temperature": args.temperature if do_sample else 0.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "num_return_sequences": args.num_samples,
            }
            if do_sample and args.top_p < 1.0:
                generate_kwargs["top_p"] = args.top_p
            with torch.no_grad():
                generated = model.generate(**generate_kwargs)
            response_ids = generated[:, input_ids.shape[-1] :]
            response_lengths = _response_token_lengths(response_ids, tokenizer.pad_token_id)
            env_state = _build_env_state(environment, problem, split=args.split, curriculum=args.curriculum)
            sampled_responses: list[str] = []
            parsed_predictions: list[int | None] = []
            rewards: list[float] = []
            hit_max_new_tokens: list[bool] = []

            for sample_index in range(args.num_samples):
                response = environment.postprocess_response(
                    tokenizer.decode(response_ids[sample_index], skip_special_tokens=True)
                )
                sampled_responses.append(response)
                parsed_predictions.append(verifier.extract_terminal_final_answer(response))
                rewards.append(float(verifier.verify(response, env_state)))
                hit_max_new_tokens.append(response_lengths[sample_index] >= args.max_new_tokens)

            record = {
                "question": problem.question,
                "answer": problem.answer,
                "gold_answer": problem.answer,
                "prediction": sampled_responses[0],
                "parsed_prediction": parsed_predictions[0],
                "reward": rewards[0],
                "sampled_responses": sampled_responses,
                "parsed_predictions": parsed_predictions,
                "rewards": rewards,
                "response_lengths": response_lengths,
                "hit_max_new_tokens": hit_max_new_tokens,
                "any_correct": any(reward > 0.0 for reward in rewards),
            }
            handle.write(json.dumps(record) + "\n")
            pass_at_1_total += rewards[0]
            pass_at_k_total += 1.0 if any(reward > 0.0 for reward in rewards[:effective_pass_k]) else 0.0
            any_correct_total += 1.0 if any(reward > 0.0 for reward in rewards) else 0.0
            total += 1
            total_reward += sum(rewards)
            total_response_tokens += sum(response_lengths)
            total_sampled_responses += len(rewards)
            total_hit_token_cap += sum(hit_max_new_tokens)

    summary = {
        "model_name": args.model,
        "adapter_path": args.init_adapter_path,
        "split": args.split,
        "subset_size": total,
        "curriculum": args.curriculum,
        "max_question_words": args.max_question_words,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": args.num_samples,
        "pass_k": effective_pass_k,
        "do_sample": do_sample,
        "temperature": args.temperature if do_sample else 0.0,
        "top_p": args.top_p if do_sample else 1.0,
        "exact_match_rate": (pass_at_1_total / total) if total else 0.0,
        "pass_at_1": (pass_at_1_total / total) if total else 0.0,
        "pass_at_k": (pass_at_k_total / total) if total else 0.0,
        "fraction_with_any_correct": (any_correct_total / total) if total else 0.0,
        "mean_reward": (total_reward / total_sampled_responses) if total_sampled_responses else 0.0,
        "mean_response_tokens": (total_response_tokens / total_sampled_responses) if total_sampled_responses else 0.0,
        "fraction_hitting_max_new_tokens": (
            total_hit_token_cap / total_sampled_responses
        ) if total_sampled_responses else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
