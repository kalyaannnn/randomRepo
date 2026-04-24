"""Standard rollout orchestration for AgentRL."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import ConfigurationError, GRPOConfig
from agentrl.generation.prefill import ChunkedPrefillMixin


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RolloutBatch:
    """Internal multi-turn training batch produced by rollout collection."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_policy_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    metadata: dict[str, Any]


class RolloutOrchestrator(ChunkedPrefillMixin):
    """Collect grouped rollouts and build multi-turn GRPO training batches."""

    def __init__(
        self,
        config: GRPOConfig,
        environment: BaseEnvironment,
        verifier: BaseVerifier,
        tokenizer: Any,
        layout: Any,
        device: torch.device | None = None,
        rng: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.environment = environment
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.layout = layout
        self.device = device or self._infer_device()
        self.rng = rng
        self._reset_runtime_stats()

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

    def collect(self) -> RolloutBatch:
        """Run one rollout phase and return a batched multi-turn training batch."""

        self._reset_runtime_stats()
        episodes: list[dict[str, Any]] = []
        for _ in range(self.config.batch_size):
            root_env = self._clone_environment(self.environment)
            initial_observation = root_env.reset()
            prompt_group: list[dict[str, Any]] = []
            for _ in range(self.config.group_size):
                env = self._clone_environment(root_env)
                prompt_group.append(self._run_episode(env, initial_observation))
            episodes.extend(prompt_group)

        input_ids, attention_mask, completion_mask = self._pack_sequences(episodes)
        flat_input_ids = input_ids.view(-1, input_ids.shape[-1])
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        flat_completion_mask = completion_mask.view(-1, completion_mask.shape[-1])

        model_config = getattr(self.layout.model, "config", None)
        if model_config is not None:
            model_config.use_cache = True

        with torch.no_grad():
            policy_sequences = self._compute_logprobs(
                self.layout.policy_forward,
                flat_input_ids,
                flat_attention_mask,
                flat_completion_mask,
            )

        if model_config is not None:
            model_config.use_cache = False

        rewards = torch.tensor(
            [[episode["reward"] for episode in episodes[i : i + self.config.group_size]]
             for i in range(0, len(episodes), self.config.group_size)],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = self._compute_advantages(rewards)

        metadata = self._build_metadata(episodes, rewards)
        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            old_policy_logprobs=policy_sequences.view_as(input_ids),
            rewards=rewards,
            advantages=advantages,
            metadata=metadata,
        )

    def _run_episode(self, env: BaseEnvironment, initial_observation: str) -> dict[str, Any]:
        """Generate one complete episode from a forked environment state."""

        observations = [initial_observation]
        actions: list[str] = []
        truncated = False

        for step_index in range(self.config.max_episode_steps):
            prompt_text = self._render_generation_prompt(observations, actions)
            response_text = self._generate_text(prompt_text)
            actions.append(response_text)
            next_observation, done = env.step(response_text)
            if done:
                break
            observations.append(next_observation)
        else:
            truncated = True
            done = False
            LOGGER.warning(
                "Episode hit max_episode_steps=%s before environment termination.",
                self.config.max_episode_steps,
            )

        final_response = actions[-1] if actions else ""
        reward = float(self.verifier.verify(final_response, env.state()))
        transcript_text, assistant_spans = self._render_transcript(observations, actions)
        return {
            "prompt_text": observations[0],
            "final_response": final_response,
            "responses": list(actions),
            "observations": list(observations),
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "transcript_text": transcript_text,
            "assistant_spans": assistant_spans,
        }

    def _clone_environment(self, environment: BaseEnvironment) -> BaseEnvironment:
        """Fork environment state for grouped rollouts without changing the API."""

        try:
            return copy.deepcopy(environment)
        except Exception as exc:
            raise ConfigurationError(
                "Environment must be deepcopy-safe after reset() when group_size > 1."
            ) from exc

    def _render_generation_prompt(self, observations: list[str], actions: list[str]) -> str:
        """Render the canonical plain-text transcript for the next generation step."""

        custom_renderer = getattr(self.environment, "render_generation_prompt", None)
        if callable(custom_renderer):
            return custom_renderer(self.tokenizer, observations, actions)

        parts: list[str] = []
        for index, observation in enumerate(observations):
            parts.append("Observation:\n")
            parts.append(observation)
            parts.append("\n\n")
            if index < len(actions):
                parts.append("Assistant:\n")
                parts.append(actions[index])
                parts.append("\n\n")
        parts.append("Assistant:\n")
        return "".join(parts)

    def _render_transcript(
        self,
        observations: list[str],
        actions: list[str],
    ) -> tuple[str, list[tuple[int, int]]]:
        """Render a full transcript and record character spans of assistant text."""

        custom_renderer = getattr(self.environment, "render_transcript", None)
        if callable(custom_renderer):
            return custom_renderer(self.tokenizer, observations, actions)

        parts: list[str] = []
        assistant_spans: list[tuple[int, int]] = []
        cursor = 0

        for index, observation in enumerate(observations):
            prefix = "Observation:\n"
            parts.extend((prefix, observation, "\n\n"))
            cursor += len(prefix) + len(observation) + 2
            if index < len(actions):
                assistant_prefix = "Assistant:\n"
                parts.append(assistant_prefix)
                cursor += len(assistant_prefix)
                start = cursor
                parts.append(actions[index])
                cursor += len(actions[index])
                end = cursor
                assistant_spans.append((start, end))
                parts.append("\n\n")
                cursor += 2

        return "".join(parts), assistant_spans

    def _generate_text(self, prompt_text: str) -> str:
        """Generate one assistant response from the current policy."""

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        if self.config.max_prompt_tokens is not None and input_ids.shape[-1] > self.config.max_prompt_tokens:
            input_ids = input_ids[:, -self.config.max_prompt_tokens :]
            attention_mask = attention_mask[:, -self.config.max_prompt_tokens :]
        prompt_tokens = int(attention_mask.sum().item())
        self._runtime_stats["prefill_tokens"] += float(prompt_tokens)

        generation_model = self.layout.model
        generation_model.config.use_cache = True
        use_chunked_prefill = input_ids.shape[-1] > self.config.prefill_chunk_size
        if use_chunked_prefill:
            with torch.no_grad():
                response_ids = self._generate_with_chunked_prefill(
                    generation_model,
                    input_ids,
                    attention_mask,
                )
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            return self._postprocess_response(response_text)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if self.config.do_sample and self.config.top_p < 1.0:
            generate_kwargs["top_p"] = self.config.top_p
        start = time.perf_counter()
        with torch.no_grad():
            generated = generation_model.generate(**generate_kwargs)
        self._runtime_stats["decode_time_ms"] += (time.perf_counter() - start) * 1000.0

        prompt_length = input_ids.shape[-1]
        response_ids = generated[:, prompt_length:]
        self._runtime_stats["decode_tokens"] += float(response_ids.numel())
        response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return self._postprocess_response(response_text)

    def _generate_with_chunked_prefill(
        self,
        generation_model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate tokens by chunk-prefilling the prompt, then decoding autoregressively."""

        prompt_tokens = int(attention_mask.sum().item())
        prefill_start = time.perf_counter()
        next_token_logits, past_key_values = self.chunked_prefill_for_generation(
            model=generation_model,
            prompt_ids=input_ids,
            chunk_size=self.config.prefill_chunk_size,
            attention_mask=attention_mask,
        )
        self._runtime_stats["prefill_time_ms"] += (time.perf_counter() - prefill_start) * 1000.0
        generated_tokens: list[torch.Tensor] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        decode_start = time.perf_counter()
        current_context_tokens = prompt_tokens

        for _ in range(self.config.max_new_tokens):
            next_token = self._sample_next_token(next_token_logits)
            generated_tokens.append(next_token)
            self._runtime_stats["decode_tokens"] += float(next_token.numel())
            if eos_token_id is not None and bool((next_token == eos_token_id).all().item()):
                break
            self._runtime_stats["cache_reuse_tokens"] += float(current_context_tokens)
            outputs = generation_model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            current_context_tokens += 1
        self._runtime_stats["decode_time_ms"] += (time.perf_counter() - decode_start) * 1000.0

        if not generated_tokens:
            return torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=self.device)
        return torch.stack(generated_tokens, dim=1)

    def _sample_next_token(self, next_token_logits: torch.Tensor) -> torch.Tensor:
        """Sample or greedily decode the next token from final-step logits."""

        if self.config.do_sample and self.config.temperature > 0:
            filtered_logits = self._apply_top_p(next_token_logits / self.config.temperature)
            probs = torch.softmax(filtered_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1, generator=self.rng).squeeze(-1)
        return torch.argmax(next_token_logits, dim=-1)

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply nucleus filtering to logits when configured."""

        if self.config.top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > self.config.top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        filtered_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        return torch.empty_like(filtered_logits).scatter(-1, sorted_indices, filtered_logits)

    def _postprocess_response(self, response_text: str) -> str:
        """Apply optional environment- or config-driven stop-string cleanup."""

        custom_postprocess = getattr(self.environment, "postprocess_response", None)
        if callable(custom_postprocess):
            return custom_postprocess(response_text)

        truncated = response_text
        for stop_string in self.config.stop_strings:
            stop_index = truncated.find(stop_string)
            if stop_index != -1:
                truncated = truncated[:stop_index]
        return truncated.rstrip()

    def _pack_sequences(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize transcripts and pad them into `[B, G, L]` tensors."""

        tokenized: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_length = 0
        for episode in episodes:
            seq_ids, seq_attention, seq_completion = self._tokenize_transcript(
                episode["transcript_text"],
                episode["assistant_spans"],
            )
            tokenized.append((seq_ids, seq_attention, seq_completion))
            max_length = max(max_length, int(seq_ids.numel()))

        if self.config.pad_to_multiple_of is not None and max_length > 0:
            remainder = max_length % self.config.pad_to_multiple_of
            if remainder != 0:
                max_length += self.config.pad_to_multiple_of - remainder

        padded_ids: list[torch.Tensor] = []
        padded_attention: list[torch.Tensor] = []
        padded_completion: list[torch.Tensor] = []
        pad_token_id = int(self.tokenizer.pad_token_id)

        for seq_ids, seq_attention, seq_completion in tokenized:
            pad_amount = max_length - int(seq_ids.numel())
            padded_ids.append(torch.nn.functional.pad(seq_ids, (0, pad_amount), value=pad_token_id))
            padded_attention.append(torch.nn.functional.pad(seq_attention, (0, pad_amount), value=0))
            padded_completion.append(torch.nn.functional.pad(seq_completion, (0, pad_amount), value=0))

        stacked_ids = torch.stack(padded_ids, dim=0).to(self.device)
        stacked_attention = torch.stack(padded_attention, dim=0).to(self.device)
        stacked_completion = torch.stack(padded_completion, dim=0).to(self.device)
        total_token_capacity = max_length * len(tokenized)
        total_real_tokens = sum(int(seq_ids.numel()) for seq_ids, _seq_attention, _seq_completion in tokenized)
        self._runtime_stats["sequence_padding_waste_tokens"] = float(total_token_capacity - total_real_tokens)
        self._runtime_stats["sequence_padding_total_tokens"] = float(total_token_capacity)
        shape = (self.config.batch_size, self.config.group_size, max_length)
        return (
            stacked_ids.view(shape),
            stacked_attention.view(shape),
            stacked_completion.view(shape),
        )

    def _tokenize_transcript(
        self,
        transcript_text: str,
        assistant_spans: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize one transcript and align assistant-token masks."""

        encoded = self.tokenizer(
            transcript_text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"][0].to(dtype=torch.long)
        attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0].to(dtype=torch.long)
        offsets = encoded.get("offset_mapping")

        if offsets is None:
            return self._fallback_piecewise_tokenize(transcript_text, assistant_spans)

        completion_mask = torch.zeros(input_ids.shape[-1], dtype=torch.bool)
        for token_index, (start, end) in enumerate(offsets[0].tolist()):
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in assistant_spans):
                completion_mask[token_index] = True

        return input_ids, attention_mask, completion_mask

    def _fallback_piecewise_tokenize(
        self,
        transcript_text: str,
        assistant_spans: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fallback tokenizer path when offset mappings are unavailable.

        This path tokenizes plain-text segments independently, which can differ
        slightly from whole-string tokenization for some BPE tokenizers. It is a
        deterministic fallback used only when no offset-aware tokenizer exists.
        """

        boundaries = sorted(
            {0, len(transcript_text), *[item for span in assistant_spans for item in span]}
        )
        token_ids_parts: list[torch.Tensor] = []
        completion_mask_parts: list[torch.Tensor] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            segment = transcript_text[start:end]
            if not segment:
                continue
            encoded = self.tokenizer(segment, return_tensors="pt", add_special_tokens=False)
            segment_ids = encoded["input_ids"][0].to(dtype=torch.long)
            is_completion = any(
                start >= span_start and end <= span_end for span_start, span_end in assistant_spans
            )
            token_ids_parts.append(segment_ids)
            completion_mask_parts.append(
                torch.full((segment_ids.numel(),), is_completion, dtype=torch.bool)
            )

        input_ids = torch.cat(token_ids_parts, dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        completion_mask = torch.cat(completion_mask_parts, dim=0)
        return input_ids, attention_mask, completion_mask

    def _compute_logprobs(
        self,
        forward_fn: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token logprobs aligned to sequence positions."""

        logits = forward_fn(input_ids=input_ids, attention_mask=attention_mask)
        token_logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        next_tokens = input_ids[:, 1:].unsqueeze(-1)
        gathered = token_logprobs.gather(dim=-1, index=next_tokens).squeeze(-1)

        aligned = torch.zeros_like(input_ids, dtype=logits.dtype)
        aligned[:, 1:] = gathered * completion_mask[:, 1:].to(dtype=logits.dtype)
        return aligned

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards within each prompt group and clip advantages."""

        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        zero_std = std.squeeze(-1) == 0
        if bool(zero_std.any()):
            LOGGER.warning(
                "reward_std == 0 for %s prompt groups; advantages for those groups are zero.",
                int(zero_std.sum().item()),
            )

        normalized = torch.where(
            std > 0,
            (rewards - mean) / (std + 1e-8),
            torch.zeros_like(rewards),
        )
        return normalized.clamp(min=-self.config.clip_range, max=self.config.clip_range)

    def _build_metadata(self, episodes: list[dict[str, Any]], rewards: torch.Tensor) -> dict[str, Any]:
        """Build replay/debug metadata and batch-level exploration warnings."""

        grouped_prompts = []
        grouped_responses = []
        grouped_transcripts = []
        unique_ratios = []
        repeated_response = None

        for start in range(0, len(episodes), self.config.group_size):
            group = episodes[start : start + self.config.group_size]
            prompt = group[0]["prompt_text"]
            responses = [episode["final_response"] for episode in group]
            grouped_prompts.append(prompt)
            grouped_responses.append(responses)
            grouped_transcripts.append([episode["transcript_text"] for episode in group])
            unique_ratio = len(set(responses)) / float(self.config.group_size)
            unique_ratios.append(unique_ratio)
            if repeated_response is None and unique_ratio < 1.0:
                counts: dict[str, int] = {}
                for response in responses:
                    counts[response] = counts.get(response, 0) + 1
                repeated_response = max(counts, key=counts.get)

        unique_response_ratio = float(sum(unique_ratios) / len(unique_ratios)) if unique_ratios else 0.0
        if unique_response_ratio < 0.3 and repeated_response is not None:
            LOGGER.warning(
                "unique_response_ratio < 0.3 (%.3f). Repeated response: %r",
                unique_response_ratio,
                repeated_response,
            )

        generation_padding_total = float(self._runtime_stats["generation_padding_total_tokens"])
        generation_padding_waste = float(self._runtime_stats["generation_padding_waste_tokens"])
        sequence_padding_total = float(self._runtime_stats["sequence_padding_total_tokens"])
        sequence_padding_waste = float(self._runtime_stats["sequence_padding_waste_tokens"])
        total_padding_total = generation_padding_total + sequence_padding_total
        total_padding_waste = generation_padding_waste + sequence_padding_waste
        prefill_time_ms = float(self._runtime_stats["prefill_time_ms"])
        decode_time_ms = float(self._runtime_stats["decode_time_ms"])
        prefill_tokens = float(self._runtime_stats["prefill_tokens"])
        decode_tokens = float(self._runtime_stats["decode_tokens"])
        cache_reuse_tokens = float(self._runtime_stats["cache_reuse_tokens"])
        scheduler_prefill_passes = float(self._runtime_stats["scheduler_prefill_passes"])
        scheduler_decode_passes = float(self._runtime_stats["scheduler_decode_passes"])

        return {
            "prompts": grouped_prompts,
            "responses": grouped_responses,
            "transcripts": grouped_transcripts,
            "unique_response_ratio": unique_response_ratio,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "prefill_time_ms": prefill_time_ms,
            "decode_time_ms": decode_time_ms,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_tokens_per_second": (
                prefill_tokens / (prefill_time_ms / 1000.0)
            ) if prefill_time_ms > 0 else 0.0,
            "decode_tokens_per_second": (
                decode_tokens / (decode_time_ms / 1000.0)
            ) if decode_time_ms > 0 else 0.0,
            "cache_reuse_tokens": cache_reuse_tokens,
            "cache_reuse_effectiveness": (
                cache_reuse_tokens / (cache_reuse_tokens + prefill_tokens)
            ) if (cache_reuse_tokens + prefill_tokens) > 0 else 0.0,
            "generation_padding_waste_tokens": generation_padding_waste,
            "generation_padding_ratio": (
                generation_padding_waste / generation_padding_total
            ) if generation_padding_total > 0 else 0.0,
            "sequence_padding_waste_tokens": sequence_padding_waste,
            "sequence_padding_ratio": (
                sequence_padding_waste / sequence_padding_total
            ) if sequence_padding_total > 0 else 0.0,
            "padding_waste_tokens": total_padding_waste,
            "padding_ratio": (
                total_padding_waste / total_padding_total
            ) if total_padding_total > 0 else 0.0,
            "scheduler_prefill_token_budget": float(self._runtime_stats["scheduler_prefill_token_budget"]),
            "scheduler_decode_token_budget": float(self._runtime_stats["scheduler_decode_token_budget"]),
            "scheduler_prefill_passes": scheduler_prefill_passes,
            "scheduler_decode_passes": scheduler_decode_passes,
            "scheduler_prefill_admitted_sequences": float(
                self._runtime_stats["scheduler_prefill_admitted_sequences"]
            ),
            "scheduler_decode_admitted_sequences": float(
                self._runtime_stats["scheduler_decode_admitted_sequences"]
            ),
            "scheduler_prefill_kv_budget_mb": float(self._runtime_stats["scheduler_prefill_kv_budget_mb"]),
            "scheduler_decode_kv_budget_mb": float(self._runtime_stats["scheduler_decode_kv_budget_mb"]),
            "scheduler_prefill_admitted_kv_mb": float(self._runtime_stats["scheduler_prefill_admitted_kv_mb"]),
            "scheduler_decode_admitted_kv_mb": float(self._runtime_stats["scheduler_decode_admitted_kv_mb"]),
            "scheduler_prefill_kv_pressure": (
                float(self._runtime_stats["scheduler_prefill_kv_pressure"]) / scheduler_prefill_passes
            ) if scheduler_prefill_passes > 0 else 0.0,
            "scheduler_decode_kv_pressure": (
                float(self._runtime_stats["scheduler_decode_kv_pressure"]) / scheduler_decode_passes
            ) if scheduler_decode_passes > 0 else 0.0,
            "scheduler_prefill_block_budget": float(self._runtime_stats["scheduler_prefill_block_budget"]),
            "scheduler_decode_block_budget": float(self._runtime_stats["scheduler_decode_block_budget"]),
            "scheduler_prefill_admitted_blocks": float(self._runtime_stats["scheduler_prefill_admitted_blocks"]),
            "scheduler_decode_admitted_blocks": float(self._runtime_stats["scheduler_decode_admitted_blocks"]),
            "scheduler_decode_growth_block_demand": float(
                self._runtime_stats["scheduler_decode_growth_block_demand"]
            ),
            "scheduler_decode_growth_blocks_admitted": float(
                self._runtime_stats["scheduler_decode_growth_blocks_admitted"]
            ),
            "scheduler_length_sort_passes": float(self._runtime_stats["scheduler_length_sort_passes"]),
            "scheduler_length_sorted_sequences": float(self._runtime_stats["scheduler_length_sorted_sequences"]),
            "scheduler_deferred_sequences": float(self._runtime_stats["scheduler_deferred_sequences"]),
            "scheduler_max_concurrent_sequences": float(
                self._runtime_stats["scheduler_max_concurrent_sequences"]
            ),
            "paged_kv_block_size_tokens": float(self._runtime_stats["paged_kv_block_size_tokens"]),
            "paged_kv_free_block_count": float(self._runtime_stats["paged_kv_free_block_count"]),
            "paged_kv_used_block_count": float(self._runtime_stats["paged_kv_used_block_count"]),
            "paged_kv_allocator_occupancy": float(self._runtime_stats["paged_kv_allocator_occupancy"]),
            "paged_kv_block_reuse_count": float(self._runtime_stats["paged_kv_block_reuse_count"]),
            "paged_kv_allocator_pressure": float(self._runtime_stats["paged_kv_allocator_pressure"]),
            "paged_kv_max_blocks_in_use": float(self._runtime_stats["paged_kv_max_blocks_in_use"]),
            "paged_kv_resident_sequences": float(self._runtime_stats["paged_kv_resident_sequences"]),
            "paged_kv_preempted_sequences": float(self._runtime_stats["paged_kv_preempted_sequences"]),
            "paged_kv_max_preempted_sequences": float(self._runtime_stats["paged_kv_max_preempted_sequences"]),
        }

    def _reset_runtime_stats(self) -> None:
        """Reset generation-side runtime counters for one rollout collection."""

        self._runtime_stats = {
            "prefill_time_ms": 0.0,
            "decode_time_ms": 0.0,
            "prefill_tokens": 0.0,
            "decode_tokens": 0.0,
            "cache_reuse_tokens": 0.0,
            "generation_padding_waste_tokens": 0.0,
            "generation_padding_total_tokens": 0.0,
            "sequence_padding_waste_tokens": 0.0,
            "sequence_padding_total_tokens": 0.0,
            "scheduler_prefill_token_budget": 0.0,
            "scheduler_decode_token_budget": 0.0,
            "scheduler_prefill_passes": 0.0,
            "scheduler_decode_passes": 0.0,
            "scheduler_prefill_admitted_sequences": 0.0,
            "scheduler_decode_admitted_sequences": 0.0,
            "scheduler_prefill_kv_budget_mb": 0.0,
            "scheduler_decode_kv_budget_mb": 0.0,
            "scheduler_prefill_admitted_kv_mb": 0.0,
            "scheduler_decode_admitted_kv_mb": 0.0,
            "scheduler_prefill_kv_pressure": 0.0,
            "scheduler_decode_kv_pressure": 0.0,
            "scheduler_prefill_block_budget": 0.0,
            "scheduler_decode_block_budget": 0.0,
            "scheduler_prefill_admitted_blocks": 0.0,
            "scheduler_decode_admitted_blocks": 0.0,
            "scheduler_decode_growth_block_demand": 0.0,
            "scheduler_decode_growth_blocks_admitted": 0.0,
            "scheduler_length_sort_passes": 0.0,
            "scheduler_length_sorted_sequences": 0.0,
            "scheduler_deferred_sequences": 0.0,
            "scheduler_max_concurrent_sequences": 0.0,
            "paged_kv_block_size_tokens": 0.0,
            "paged_kv_free_block_count": 0.0,
            "paged_kv_used_block_count": 0.0,
            "paged_kv_allocator_occupancy": 0.0,
            "paged_kv_block_reuse_count": 0.0,
            "paged_kv_allocator_pressure": 0.0,
            "paged_kv_max_blocks_in_use": 0.0,
            "paged_kv_resident_sequences": 0.0,
            "paged_kv_preempted_sequences": 0.0,
            "paged_kv_max_preempted_sequences": 0.0,
        }

    def _infer_device(self) -> torch.device:
        """Infer the execution device from the attached model layout."""

        parameter = next(self.layout.model.parameters(), None)
        if parameter is None:
            return torch.device("cpu")
        return parameter.device
