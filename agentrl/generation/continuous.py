"""Continuous batching rollout orchestration for AgentRL."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from agentrl.core.base import BaseEnvironment
from agentrl.core.rollout import RolloutBatch, RolloutOrchestrator
from agentrl.generation.paged_kv import PagedKVAllocator, PagedKVCacheStore
from agentrl.generation.scheduler import (
    dtype_bytes,
    estimate_kv_cache_sequence_bytes,
    kv_cache_geometry,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _EpisodeState:
    """Mutable per-episode state used by the continuous scheduler."""

    env: BaseEnvironment
    prompt_text: str
    observations: list[str]
    actions: list[str]
    done: bool = False
    truncated: bool = False
    reward: float = 0.0
    transcript_text: str = ""
    assistant_spans: list[tuple[int, int]] | None = None


@dataclass(slots=True)
class _ScheduledSequence:
    """Tokenized prompt state tracked by the continuous scheduler."""

    original_index: int
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor

    @property
    def prompt_tokens(self) -> int:
        return int(self.prompt_mask.sum().item())


@dataclass(slots=True)
class _ContinuousSchedulerState:
    """Execution-policy-driven admission state for continuous rollout."""

    max_batch_size: int
    prefill_token_budget: int
    decode_token_budget: int
    prefill_cost_budget: int
    decode_cost_budget: int
    kv_bytes_per_token: int | None = None
    block_size_tokens: int = 1
    kv_bytes_per_block: int | None = None


class ContinuousBatchingOrchestrator(RolloutOrchestrator):
    """Rollout orchestrator that drops finished sequences during decoding."""

    PAGED_KV_BLOCK_SIZE_TOKENS = 16

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._high_padding_streak = 0

    def collect(self) -> RolloutBatch:
        """Collect one rollout batch using continuous per-step scheduling."""

        self._reset_runtime_stats()
        states: list[_EpisodeState] = []
        for _ in range(self.config.batch_size):
            root_env = self._clone_environment(self.environment)
            initial_observation = root_env.reset()
            for _ in range(self.config.group_size):
                states.append(
                    _EpisodeState(
                        env=self._clone_environment(root_env),
                        prompt_text=initial_observation,
                        observations=[initial_observation],
                        actions=[],
                    )
                )

        padding_ratios: list[float] = []
        for turn_index in range(self.config.max_episode_steps):
            active_indices = [index for index, state in enumerate(states) if not state.done]
            if not active_indices:
                break

            prompts = [
                self._render_generation_prompt(states[index].observations, states[index].actions)
                for index in active_indices
            ]
            ordered_indices, ordered_prompts = self._order_active_prompts_by_length(active_indices, prompts)
            responses, padding_ratio = self._generate_active_batch(ordered_prompts)
            responses_by_index = {
                state_index: response_text
                for state_index, response_text in zip(ordered_indices, responses, strict=True)
            }
            padding_ratios.append(padding_ratio)
            self._track_padding_ratio(padding_ratio)

            for state_index in active_indices:
                response_text = responses_by_index[state_index]
                state = states[state_index]
                state.actions.append(response_text)
                next_observation, done = state.env.step(response_text)
                if done:
                    state.done = True
                    state.reward = float(self.verifier.verify(response_text, state.env.state()))
                    transcript_text, spans = self._render_transcript(state.observations, state.actions)
                    state.transcript_text = transcript_text
                    state.assistant_spans = spans
                else:
                    state.observations.append(next_observation)

        for state in states:
            if state.done:
                continue
            state.truncated = True
            state.reward = float(self.verifier.verify(state.actions[-1] if state.actions else "", state.env.state()))
            state.transcript_text, state.assistant_spans = self._render_transcript(state.observations, state.actions)
            LOGGER.warning(
                "Episode hit max_episode_steps=%s before environment termination.",
                self.config.max_episode_steps,
            )

        episode_dicts = [
            {
                "prompt_text": state.prompt_text,
                "final_response": state.actions[-1] if state.actions else "",
                "responses": list(state.actions),
                "observations": list(state.observations),
                "reward": state.reward,
                "done": state.done,
                "truncated": state.truncated,
                "transcript_text": state.transcript_text,
                "assistant_spans": state.assistant_spans or [],
            }
            for state in states
        ]

        input_ids, attention_mask, action_mask = self._pack_sequences(episode_dicts)
        flat_input_ids = input_ids.view(-1, input_ids.shape[-1])
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        flat_action_mask = action_mask.view(-1, action_mask.shape[-1])

        model_config = getattr(self.layout.model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

        with torch.no_grad():
            policy_sequences = self._compute_logprobs(
                self.layout.policy_forward,
                flat_input_ids,
                flat_attention_mask,
                flat_action_mask,
            )
            ref_sequences = self._compute_logprobs(
                self.layout.reference_forward,
                flat_input_ids,
                flat_attention_mask,
                flat_action_mask,
            )

        rewards = torch.tensor(
            [[episode["reward"] for episode in episode_dicts[i : i + self.config.group_size]]
             for i in range(0, len(episode_dicts), self.config.group_size)],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = self._compute_advantages(rewards)
        metadata = self._build_metadata(episode_dicts, rewards)

        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            policy_logprobs=policy_sequences.view_as(input_ids),
            ref_logprobs=ref_sequences.view_as(input_ids),
            rewards=rewards,
            advantages=advantages,
            metadata=metadata,
        )

    def _generate_active_batch(self, prompt_texts: list[str]) -> tuple[list[str], float]:
        """Generate responses for active prompts with step-level dynamic batching."""

        encoded_inputs = [
            self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            for prompt_text in prompt_texts
        ]
        prompt_ids = [encoded["input_ids"][0].to(self.device) for encoded in encoded_inputs]
        prompt_masks = [
            encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0].to(self.device)
            for encoded in encoded_inputs
        ]
        if self.config.max_prompt_tokens is not None:
            truncated_ids: list[torch.Tensor] = []
            truncated_masks: list[torch.Tensor] = []
            for prompt, mask in zip(prompt_ids, prompt_masks, strict=True):
                if prompt.numel() > self.config.max_prompt_tokens:
                    truncated_ids.append(prompt[-self.config.max_prompt_tokens :])
                    truncated_masks.append(mask[-self.config.max_prompt_tokens :])
                else:
                    truncated_ids.append(prompt)
                    truncated_masks.append(mask)
            prompt_ids = truncated_ids
            prompt_masks = truncated_masks

        scheduler = self._build_scheduler_state(active_count=len(prompt_ids))
        scheduled_sequences = [
            _ScheduledSequence(
                original_index=index,
                prompt_ids=prompt,
                prompt_mask=mask,
            )
            for index, (prompt, mask) in enumerate(zip(prompt_ids, prompt_masks, strict=True))
        ]
        self._runtime_stats["scheduler_prefill_token_budget"] += float(scheduler.prefill_token_budget)
        self._runtime_stats["scheduler_decode_token_budget"] += float(scheduler.decode_token_budget)
        self._runtime_stats["scheduler_prefill_kv_budget_mb"] += self._bytes_to_mb(scheduler.prefill_cost_budget)
        self._runtime_stats["scheduler_decode_kv_budget_mb"] += self._bytes_to_mb(scheduler.decode_cost_budget)
        if self.config.use_paged_kv_continuous:
            self._runtime_stats["scheduler_prefill_block_budget"] += float(
                max(1, scheduler.prefill_token_budget // scheduler.block_size_tokens)
            )
            self._runtime_stats["scheduler_decode_block_budget"] += float(
                max(1, scheduler.decode_token_budget // scheduler.block_size_tokens)
            )

        generation_model = self.layout.model
        if self._supports_persistent_kv_decode(generation_model):
            if not self.config.use_paged_kv_continuous:
                return self._generate_active_batch_with_legacy_cache(scheduled_sequences, scheduler)
            return self._generate_active_batch_with_cache(scheduled_sequences, scheduler)
        return self._generate_active_batch_without_cache(scheduled_sequences, scheduler)

    def _generate_active_batch_with_legacy_cache(
        self,
        scheduled_sequences: list[_ScheduledSequence],
        scheduler: _ContinuousSchedulerState,
    ) -> tuple[list[str], float]:
        """Legacy persistent-KV continuous batching baseline."""

        generation_model = self.layout.model
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        generated_ids = [
            torch.empty(0, dtype=torch.long, device=self.device)
            for _ in scheduled_sequences
        ]
        finished = [False for _ in scheduled_sequences]
        generated_steps = [0 for _ in scheduled_sequences]
        sequence_lengths = [sequence.prompt_tokens for sequence in scheduled_sequences]
        sequence_caches: list[Any | None] = [None for _ in scheduled_sequences]
        next_logits_by_index: dict[int, torch.Tensor] = {}
        self._runtime_stats["prefill_tokens"] += float(sum(sequence_lengths))

        prefill_start = time.perf_counter()
        for batch in self._iter_scheduled_prefill_batches(scheduled_sequences, scheduler):
            prompt_ids = [sequence.prompt_ids for sequence in batch]
            prompt_masks = [sequence.prompt_mask for sequence in batch]
            batch_caches, batch_logits = self._prefill_prompt_caches(generation_model, prompt_ids, prompt_masks)
            for offset, sequence in enumerate(batch):
                sequence_caches[sequence.original_index] = batch_caches[offset]
                next_logits_by_index[sequence.original_index] = batch_logits[offset : offset + 1]
        self._runtime_stats["prefill_time_ms"] += (time.perf_counter() - prefill_start) * 1000.0

        total_padding_tokens = 0
        total_step_tokens = 0
        decode_start = time.perf_counter()
        while True:
            active_indices = [
                index
                for index, is_finished in enumerate(finished)
                if not is_finished and generated_steps[index] < self.config.max_new_tokens
            ]
            if not active_indices:
                break

            admitted_indices = self._select_admitted_indices(
                candidate_indices=active_indices,
                estimated_tokens={index: sequence_lengths[index] for index in active_indices},
                estimated_costs={
                    index: self._estimate_sequence_kv_cost(sequence_lengths[index], scheduler)
                    for index in active_indices
                },
                cost_budget=scheduler.decode_cost_budget,
                max_batch_size=scheduler.max_batch_size,
                phase="decode",
                scheduler=scheduler,
            )
            active_logits = torch.cat([next_logits_by_index[index] for index in admitted_indices], dim=0)
            next_tokens = self._sample_next_token(active_logits)

            decode_buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
            for batch_offset, episode_index in enumerate(admitted_indices):
                token_tensor = next_tokens[batch_offset : batch_offset + 1].to(dtype=torch.long, device=self.device)
                generated_ids[episode_index] = torch.cat((generated_ids[episode_index], token_tensor), dim=0)
                self._runtime_stats["decode_tokens"] += float(token_tensor.numel())
                generated_steps[episode_index] += 1
                token = int(token_tensor.item())
                if eos_token_id is not None and token == eos_token_id:
                    finished[episode_index] = True
                    continue
                decode_buckets.setdefault(sequence_lengths[episode_index], []).append((episode_index, token_tensor))

            if not decode_buckets:
                break

            deferred_logits = {
                index: next_logits_by_index[index]
                for index in active_indices
                if index not in admitted_indices
            }
            next_logits_by_index = deferred_logits
            for sequence_length, bucket in decode_buckets.items():
                total_step_tokens += sequence_length * len(bucket)
                self._runtime_stats["cache_reuse_tokens"] += float(sequence_length * len(bucket))

                bucket_indices = [episode_index for episode_index, _token_tensor in bucket]
                bucket_tokens = torch.stack([token_tensor for _episode_index, token_tensor in bucket], dim=0)
                bucket_attention = torch.ones(
                    (len(bucket), sequence_length + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                bucket_cache = self._stack_past_key_values([sequence_caches[index] for index in bucket_indices])
                outputs = generation_model(
                    input_ids=bucket_tokens,
                    attention_mask=bucket_attention,
                    past_key_values=bucket_cache,
                    use_cache=True,
                )
                split_cache = self._split_past_key_values(outputs.past_key_values, len(bucket_indices))
                bucket_logits = outputs.logits[:, -1, :]

                for offset, episode_index in enumerate(bucket_indices):
                    sequence_caches[episode_index] = split_cache[offset]
                    sequence_lengths[episode_index] += 1
                    next_logits_by_index[episode_index] = bucket_logits[offset : offset + 1]
        self._runtime_stats["decode_time_ms"] += (time.perf_counter() - decode_start) * 1000.0
        self._runtime_stats["generation_padding_waste_tokens"] += float(total_padding_tokens)
        self._runtime_stats["generation_padding_total_tokens"] += float(total_step_tokens)

        decoded = [
            self._postprocess_response(self.tokenizer.decode(tokens, skip_special_tokens=True))
            for tokens in generated_ids
        ]
        padding_ratio = float(total_padding_tokens / total_step_tokens) if total_step_tokens else 0.0
        return decoded, padding_ratio

    def _generate_active_batch_with_cache(
        self,
        scheduled_sequences: list[_ScheduledSequence],
        scheduler: _ContinuousSchedulerState,
    ) -> tuple[list[str], float]:
        """Generate with persistent per-sequence KV caches across active decoding."""

        generation_model = self.layout.model
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        generated_ids = [
            torch.empty(0, dtype=torch.long, device=self.device)
            for _ in scheduled_sequences
        ]
        finished = [False for _ in scheduled_sequences]
        generated_steps = [0 for _ in scheduled_sequences]
        sequence_lengths = [sequence.prompt_tokens for sequence in scheduled_sequences]
        next_logits_by_index: dict[int, torch.Tensor] = {}
        paged_kv = self._build_paged_kv_allocator(scheduler, sequence_lengths)
        self._runtime_stats["prefill_tokens"] += float(sum(sequence_lengths))
        self._update_paged_kv_runtime_stats(paged_kv)

        prefill_start = time.perf_counter()
        for batch in self._iter_scheduled_prefill_batches(scheduled_sequences, scheduler):
            prompt_ids = [sequence.prompt_ids for sequence in batch]
            prompt_masks = [sequence.prompt_mask for sequence in batch]
            batch_caches, batch_logits = self._prefill_prompt_caches(generation_model, prompt_ids, prompt_masks)
            for offset, sequence in enumerate(batch):
                cache = batch_caches[offset]
                paged_kv.write_sequence_cache(
                    sequence.original_index,
                    self._cache_to_legacy(cache),
                    cache,
                )
                paged_kv.set_resident_cache(
                    sequence_id=sequence.original_index,
                    cache=cache,
                    cache_template=cache,
                )
                next_logits_by_index[sequence.original_index] = batch_logits[offset : offset + 1]
        self._runtime_stats["prefill_time_ms"] += (time.perf_counter() - prefill_start) * 1000.0

        total_padding_tokens = 0
        total_step_tokens = 0
        decode_start = time.perf_counter()
        while True:
            active_indices = [
                index
                for index, is_finished in enumerate(finished)
                if not is_finished and generated_steps[index] < self.config.max_new_tokens
            ]
            if not active_indices:
                break

            admitted_indices = self._select_admitted_indices(
                candidate_indices=active_indices,
                estimated_tokens={index: sequence_lengths[index] for index in active_indices},
                estimated_costs={
                    index: self._estimate_sequence_kv_cost(sequence_lengths[index], scheduler)
                    for index in active_indices
                },
                cost_budget=scheduler.decode_cost_budget,
                max_batch_size=scheduler.max_batch_size,
                phase="decode",
                scheduler=scheduler,
            )
            admitted_indices = self._apply_paged_kv_growth_admission(
                allocator=paged_kv,
                candidate_indices=admitted_indices,
                sequence_lengths=sequence_lengths,
                scheduler=scheduler,
            )
            self._update_paged_kv_preemption_runtime_stats(
                allocator=paged_kv,
                active_indices=active_indices,
                admitted_indices=admitted_indices,
            )
            active_logits = torch.cat([next_logits_by_index[index] for index in admitted_indices], dim=0)
            next_tokens = self._sample_next_token(active_logits)

            decode_buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
            for batch_offset, episode_index in enumerate(admitted_indices):
                token_tensor = next_tokens[batch_offset : batch_offset + 1].to(dtype=torch.long, device=self.device)
                generated_ids[episode_index] = torch.cat((generated_ids[episode_index], token_tensor), dim=0)
                self._runtime_stats["decode_tokens"] += float(token_tensor.numel())
                generated_steps[episode_index] += 1
                token = int(token_tensor.item())
                if eos_token_id is not None and token == eos_token_id:
                    finished[episode_index] = True
                    paged_kv.release(episode_index)
                    continue
                decode_buckets.setdefault(sequence_lengths[episode_index], []).append((episode_index, token_tensor))
            self._update_paged_kv_runtime_stats(paged_kv)

            if not decode_buckets:
                break

            deferred_logits = {
                index: next_logits_by_index[index]
                for index in active_indices
                if index not in admitted_indices
            }
            next_logits_by_index = deferred_logits
            for sequence_length, bucket in decode_buckets.items():
                total_step_tokens += sequence_length * len(bucket)
                self._runtime_stats["cache_reuse_tokens"] += float(sequence_length * len(bucket))

                bucket_indices = [episode_index for episode_index, _token_tensor in bucket]
                bucket_tokens = torch.stack([token_tensor for _episode_index, token_tensor in bucket], dim=0)
                bucket_attention = torch.ones(
                    (len(bucket), sequence_length + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                resident_caches = [paged_kv.resident_cache(index) for index in bucket_indices]
                bucket_cache = self._stack_past_key_values(resident_caches)
                outputs = generation_model(
                    input_ids=bucket_tokens,
                    attention_mask=bucket_attention,
                    past_key_values=bucket_cache,
                    use_cache=True,
                )
                split_caches = self._split_past_key_values(outputs.past_key_values, len(bucket_indices))
                bucket_logits = outputs.logits[:, -1, :]
                for episode_index, token_tensor in bucket:
                    paged_kv.append_tokens(episode_index, int(token_tensor.numel()))

                for offset, episode_index in enumerate(bucket_indices):
                    paged_kv.set_resident_cache(
                        sequence_id=episode_index,
                        cache=split_caches[offset],
                        cache_template=split_caches[offset],
                    )
                    sequence_lengths[episode_index] += 1
                    next_logits_by_index[episode_index] = bucket_logits[offset : offset + 1]
        self._runtime_stats["decode_time_ms"] += (time.perf_counter() - decode_start) * 1000.0
        self._runtime_stats["generation_padding_waste_tokens"] += float(total_padding_tokens)
        self._runtime_stats["generation_padding_total_tokens"] += float(total_step_tokens)
        for episode_index, is_finished in enumerate(finished):
            if not is_finished:
                paged_kv.release(episode_index)
        self._update_paged_kv_runtime_stats(paged_kv)

        decoded = [
            self._postprocess_response(self.tokenizer.decode(tokens, skip_special_tokens=True))
            for tokens in generated_ids
        ]
        padding_ratio = float(total_padding_tokens / total_step_tokens) if total_step_tokens else 0.0
        return decoded, padding_ratio

    def _order_active_prompts_by_length(
        self,
        active_indices: list[int],
        prompts: list[str],
    ) -> tuple[list[int], list[str]]:
        """Order active prompts by estimated length while preserving response mapping."""

        ordered_pairs = sorted(
            zip(active_indices, prompts, strict=True),
            key=lambda item: (len(item[1]), item[0]),
        )
        self._runtime_stats["scheduler_length_sorted_sequences"] += float(len(ordered_pairs))
        if len(ordered_pairs) > 1:
            self._runtime_stats["scheduler_length_sort_passes"] += 1.0
        return (
            [state_index for state_index, _prompt in ordered_pairs],
            [prompt for _state_index, prompt in ordered_pairs],
        )

    def _generate_active_batch_without_cache(
        self,
        scheduled_sequences: list[_ScheduledSequence],
        scheduler: _ContinuousSchedulerState,
    ) -> tuple[list[str], float]:
        """Fallback generation path for models without a cache-aware forward."""

        prompt_ids = [sequence.prompt_ids for sequence in scheduled_sequences]
        prompt_masks = [sequence.prompt_mask for sequence in scheduled_sequences]
        current_ids = [prompt.clone() for prompt in prompt_ids]
        generated_ids = [
            torch.empty(0, dtype=torch.long, device=self.device)
            for _ in scheduled_sequences
        ]
        finished = [False for _ in scheduled_sequences]
        generated_steps = [0 for _ in scheduled_sequences]
        sequence_lengths = [int(mask.sum().item()) for mask in prompt_masks]
        paged_kv = (
            self._build_paged_kv_allocator(scheduler, sequence_lengths)
            if self.config.use_paged_kv_continuous
            else None
        )
        self._runtime_stats["prefill_tokens"] += float(sum(int(mask.sum().item()) for mask in prompt_masks))
        if paged_kv is not None:
            self._update_paged_kv_runtime_stats(paged_kv)

        total_padding_tokens = 0
        total_step_tokens = 0

        if any(prompt.numel() > self.config.prefill_chunk_size for prompt in prompt_ids):
            prefill_start = time.perf_counter()
            generated_ids, current_ids, finished = self._prime_with_chunked_prefill(
                prompt_ids=prompt_ids,
                prompt_masks=prompt_masks,
                generated_ids=generated_ids,
                current_ids=current_ids,
                finished=finished,
            )
            self._runtime_stats["prefill_time_ms"] += (time.perf_counter() - prefill_start) * 1000.0
            initial_decode_tokens = sum(int(tokens.numel()) for tokens in generated_ids)
            self._runtime_stats["decode_tokens"] += float(initial_decode_tokens)
            for index, tokens in enumerate(generated_ids):
                generated_steps[index] = int(tokens.numel())
                if paged_kv is not None and generated_steps[index] > 0:
                    paged_kv.append_tokens(index, generated_steps[index])
                sequence_lengths[index] += generated_steps[index]
                if finished[index]:
                    if paged_kv is not None:
                        paged_kv.release(index)
                    continue
            if paged_kv is not None:
                self._update_paged_kv_runtime_stats(paged_kv)

        decode_start = time.perf_counter()
        while True:
            active_indices = [
                index
                for index, is_finished in enumerate(finished)
                if not is_finished and generated_steps[index] < self.config.max_new_tokens
            ]
            if not active_indices:
                break

            admitted_indices = self._select_admitted_indices(
                candidate_indices=active_indices,
                estimated_tokens={index: int(current_ids[index].numel()) for index in active_indices},
                estimated_costs={
                    index: self._estimate_sequence_kv_cost(int(current_ids[index].numel()), scheduler)
                    for index in active_indices
                },
                cost_budget=scheduler.decode_cost_budget,
                max_batch_size=scheduler.max_batch_size,
                phase="decode",
                scheduler=scheduler,
            )
            if paged_kv is not None:
                self._update_paged_kv_preemption_runtime_stats(
                    allocator=paged_kv,
                    active_indices=active_indices,
                    admitted_indices=admitted_indices,
                )
            active_sequences = [current_ids[index] for index in admitted_indices]
            max_length = max(sequence.numel() for sequence in active_sequences)
            total_step_tokens += max_length * len(active_sequences)
            total_padding_tokens += sum(max_length - int(sequence.numel()) for sequence in active_sequences)

            next_tokens = self._sample_next_tokens(active_sequences, admitted_indices)
            for batch_offset, episode_index in enumerate(admitted_indices):
                token = int(next_tokens[batch_offset].item())
                token_tensor = torch.tensor([token], dtype=torch.long, device=self.device)
                generated_ids[episode_index] = torch.cat((generated_ids[episode_index], token_tensor), dim=0)
                current_ids[episode_index] = torch.cat((current_ids[episode_index], token_tensor), dim=0)
                generated_steps[episode_index] += 1
                sequence_lengths[episode_index] += 1
                if paged_kv is not None:
                    paged_kv.append_tokens(episode_index, 1)
                self._runtime_stats["decode_tokens"] += 1.0
                if token == getattr(self.tokenizer, "eos_token_id", None):
                    finished[episode_index] = True
                    if paged_kv is not None:
                        paged_kv.release(episode_index)
            if paged_kv is not None:
                self._update_paged_kv_runtime_stats(paged_kv)
        self._runtime_stats["decode_time_ms"] += (time.perf_counter() - decode_start) * 1000.0
        self._runtime_stats["generation_padding_waste_tokens"] += float(total_padding_tokens)
        self._runtime_stats["generation_padding_total_tokens"] += float(total_step_tokens)
        if paged_kv is not None:
            for episode_index, is_finished in enumerate(finished):
                if not is_finished:
                    paged_kv.release(episode_index)
            self._update_paged_kv_runtime_stats(paged_kv)

        decoded = [self._postprocess_response(self.tokenizer.decode(tokens, skip_special_tokens=True)) for tokens in generated_ids]
        padding_ratio = float(total_padding_tokens / total_step_tokens) if total_step_tokens else 0.0
        return decoded, padding_ratio

    def _build_scheduler_state(self, active_count: int) -> _ContinuousSchedulerState:
        """Build an execution-policy-aware scheduler budget for one active turn."""

        max_batch_size = max(1, min(int(self.config.chunk_size or self.config.group_size), active_count))
        policy_factor = {
            "safe": 0.5,
            "balanced": 1.0,
            "max_throughput": 1.5,
        }[self.config.execution_policy]
        prompt_window = int(self.config.max_prompt_tokens or self.config.prefill_chunk_size)
        prefill_budget = max(
            self.config.prefill_chunk_size,
            int(max_batch_size * self.config.prefill_chunk_size * policy_factor),
        )
        decode_budget = max(
            prompt_window,
            int(max_batch_size * prompt_window * policy_factor),
        )
        model_config = getattr(self.layout.model, "config", None)
        kv_bytes = None
        block_size = self.PAGED_KV_BLOCK_SIZE_TOKENS
        if model_config is not None:
            try:
                num_layers, num_heads, head_dim = kv_cache_geometry(model_config)
                kv_bytes = estimate_kv_cache_sequence_bytes(
                    sequence_tokens=1,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype_bytes=dtype_bytes(self.config.dtype),
                )
            except AttributeError:
                kv_bytes = None
        return _ContinuousSchedulerState(
            max_batch_size=max_batch_size,
            prefill_token_budget=prefill_budget,
            decode_token_budget=decode_budget,
            prefill_cost_budget=(
                prefill_budget * kv_bytes if kv_bytes is not None else prefill_budget
            ),
            decode_cost_budget=(
                decode_budget * kv_bytes if kv_bytes is not None else decode_budget
            ),
            kv_bytes_per_token=kv_bytes,
            block_size_tokens=block_size,
            kv_bytes_per_block=(kv_bytes * block_size) if kv_bytes is not None else None,
        )

    def _iter_scheduled_prefill_batches(
        self,
        scheduled_sequences: list[_ScheduledSequence],
        scheduler: _ContinuousSchedulerState,
    ) -> list[list[_ScheduledSequence]]:
        """Split prompt prefill into token-budgeted admission waves."""

        ordered = sorted(
            scheduled_sequences,
            key=lambda sequence: (sequence.prompt_tokens, sequence.original_index),
        )
        batches: list[list[_ScheduledSequence]] = []
        start = 0
        while start < len(ordered):
            remaining = ordered[start:]
            admitted_positions = self._select_admitted_indices(
                candidate_indices=list(range(len(remaining))),
                estimated_tokens={index: remaining[index].prompt_tokens for index in range(len(remaining))},
                estimated_costs={
                    index: self._estimate_sequence_kv_cost(remaining[index].prompt_tokens, scheduler)
                    for index in range(len(remaining))
                },
                cost_budget=scheduler.prefill_cost_budget,
                max_batch_size=scheduler.max_batch_size,
                phase="prefill",
                scheduler=scheduler,
            )
            batch = [remaining[index] for index in admitted_positions]
            batches.append(batch)
            start += len(batch)
        return batches

    def _select_admitted_indices(
        self,
        candidate_indices: list[int],
        estimated_tokens: dict[int, int],
        estimated_costs: dict[int, int],
        cost_budget: int,
        max_batch_size: int,
        phase: str,
        scheduler: _ContinuousSchedulerState,
    ) -> list[int]:
        """Greedily admit a fair cost-budgeted subset and record scheduler stats."""

        if not candidate_indices:
            return []

        if phase == "decode":
            ordered_candidates = sorted(
                candidate_indices,
                key=lambda index: (estimated_tokens[index], index),
            )
        else:
            ordered_candidates = list(candidate_indices)

        admitted: list[int] = []
        used_cost = 0
        for index in ordered_candidates:
            if len(admitted) >= max_batch_size:
                break
            estimate = max(1, int(estimated_costs[index]))
            if admitted and used_cost + estimate > cost_budget:
                continue
            admitted.append(index)
            used_cost += estimate

        if not admitted:
            admitted = [ordered_candidates[0]]
        elif (
            phase == "decode"
            and self.config.execution_policy != "safe"
            and len(candidate_indices) <= max_batch_size
        ):
            admitted = list(ordered_candidates[:max_batch_size])

        deferred = max(0, len(candidate_indices) - len(admitted))
        self._runtime_stats[f"scheduler_{phase}_passes"] += 1.0
        self._runtime_stats[f"scheduler_{phase}_admitted_sequences"] += float(len(admitted))
        self._runtime_stats["scheduler_deferred_sequences"] += float(deferred)
        self._runtime_stats["scheduler_max_concurrent_sequences"] = max(
            float(self._runtime_stats["scheduler_max_concurrent_sequences"]),
            float(len(admitted)),
        )
        if self.config.use_paged_kv_continuous:
            admitted_blocks = float(
                sum(self._estimate_sequence_block_count(estimated_tokens[index], scheduler) for index in admitted)
            )
            if phase == "prefill":
                self._runtime_stats["scheduler_prefill_admitted_blocks"] += admitted_blocks
            else:
                self._runtime_stats["scheduler_decode_admitted_blocks"] += admitted_blocks
        admitted_cost_mb = self._bytes_to_mb(used_cost) if scheduler.kv_bytes_per_token is not None else 0.0
        budget_cost_mb = self._bytes_to_mb(cost_budget) if scheduler.kv_bytes_per_token is not None else 0.0
        self._runtime_stats[f"scheduler_{phase}_admitted_kv_mb"] += admitted_cost_mb
        self._runtime_stats[f"scheduler_{phase}_kv_pressure"] += (
            admitted_cost_mb / budget_cost_mb
        ) if budget_cost_mb > 0 else 0.0
        return admitted

    def _estimate_sequence_kv_cost(
        self,
        sequence_tokens: int,
        scheduler: _ContinuousSchedulerState,
    ) -> int:
        """Estimate scheduler admission cost for one sequence."""

        token_count = max(1, int(sequence_tokens))
        if not self.config.use_paged_kv_continuous:
            if scheduler.kv_bytes_per_token is None:
                return token_count
            return token_count * scheduler.kv_bytes_per_token

        block_count = self._estimate_sequence_block_count(token_count, scheduler)
        if scheduler.kv_bytes_per_token is None:
            return max(1, block_count * scheduler.block_size_tokens)
        return block_count * max(1, int(scheduler.kv_bytes_per_block or 0))

    def _estimate_sequence_block_count(
        self,
        sequence_tokens: int,
        scheduler: _ContinuousSchedulerState,
    ) -> int:
        tokens = max(1, int(sequence_tokens))
        return (tokens + scheduler.block_size_tokens - 1) // scheduler.block_size_tokens

    def _build_paged_kv_allocator(
        self,
        scheduler: _ContinuousSchedulerState,
        sequence_lengths: list[int],
    ) -> PagedKVCacheStore:
        """Build a fixed-block allocator for runtime accounting.

        The current branch uses the allocator to track paged-KV occupancy and
        reuse semantics before the full block-backed decode path replaces the
        existing cache-object implementation.
        """

        block_size_tokens = 16
        prompt_block_total = sum(
            (sequence_tokens + block_size_tokens - 1) // block_size_tokens
            for sequence_tokens in sequence_lengths
        )
        decode_block_headroom = len(sequence_lengths) * (
            (self.config.max_new_tokens + block_size_tokens - 1) // block_size_tokens
        )
        max_tokens_budget = max(
            scheduler.prefill_token_budget,
            scheduler.decode_token_budget,
            prompt_block_total * block_size_tokens + decode_block_headroom * block_size_tokens,
        )
        total_blocks = max(
            1,
            int((max_tokens_budget + block_size_tokens - 1) // block_size_tokens),
        )
        allocator = PagedKVAllocator(
            total_blocks=total_blocks,
            block_size_tokens=block_size_tokens,
        )
        store = PagedKVCacheStore(allocator=allocator)
        for episode_index, prompt_tokens in enumerate(sequence_lengths):
            store.reserve(episode_index, prompt_tokens)
        return store

    def _update_paged_kv_runtime_stats(self, allocator: PagedKVCacheStore) -> None:
        """Mirror allocator counters into rollout runtime stats."""

        metrics = allocator.metrics()
        self._runtime_stats["paged_kv_block_size_tokens"] = float(allocator.allocator.block_size_tokens)
        self._runtime_stats["paged_kv_free_block_count"] = metrics["paged_kv_free_block_count"]
        self._runtime_stats["paged_kv_used_block_count"] = metrics["paged_kv_used_block_count"]
        self._runtime_stats["paged_kv_allocator_occupancy"] = metrics["paged_kv_allocator_occupancy"]
        self._runtime_stats["paged_kv_block_reuse_count"] = metrics["paged_kv_block_reuse_count"]
        self._runtime_stats["paged_kv_allocator_pressure"] = metrics["paged_kv_allocator_pressure"]
        self._runtime_stats["paged_kv_max_blocks_in_use"] = max(
            self._runtime_stats["paged_kv_max_blocks_in_use"],
            metrics["paged_kv_max_blocks_in_use"],
        )
        self._runtime_stats["paged_kv_resident_sequences"] = metrics["paged_kv_resident_sequences"]

    def _update_paged_kv_preemption_runtime_stats(
        self,
        allocator: PagedKVCacheStore,
        active_indices: list[int],
        admitted_indices: list[int],
    ) -> None:
        """Record scheduler-level soft preemption for resident sequences.

        A sequence is considered preempted in this v1 design when it remains
        live with resident KV blocks but is not admitted on the current decode
        step.
        """

        admitted_set = set(admitted_indices)
        resident_preempted = sum(
            1
            for index in active_indices
            if index not in admitted_set and allocator.has_sequence(index)
        )
        self._runtime_stats["paged_kv_preempted_sequences"] += float(resident_preempted)
        self._runtime_stats["paged_kv_max_preempted_sequences"] = max(
            self._runtime_stats["paged_kv_max_preempted_sequences"],
            float(resident_preempted),
        )

    def _apply_paged_kv_growth_admission(
        self,
        allocator: PagedKVCacheStore,
        candidate_indices: list[int],
        sequence_lengths: list[int],
        scheduler: _ContinuousSchedulerState,
    ) -> list[int]:
        """Trim decode admission when next-token growth would exceed free blocks.

        Any sequence whose next token would cross a block boundary requires one
        extra free block. We keep the scheduler's fairness order, but cap the
        admitted set so block growth remains feasible without eviction.
        """

        free_blocks = allocator.allocator.free_block_count
        admitted: list[int] = []
        growth_blocks_used = 0
        for index in candidate_indices:
            needs_growth_block = (
                max(1, int(sequence_lengths[index])) % scheduler.block_size_tokens == 0
            )
            if needs_growth_block and growth_blocks_used >= free_blocks:
                continue
            admitted.append(index)
            if needs_growth_block:
                growth_blocks_used += 1

        if not admitted and candidate_indices:
            admitted = [candidate_indices[0]]
        self._runtime_stats["scheduler_decode_growth_block_demand"] += float(
            sum(
                1
                for index in candidate_indices
                if max(1, int(sequence_lengths[index])) % scheduler.block_size_tokens == 0
            )
        )
        self._runtime_stats["scheduler_decode_growth_blocks_admitted"] += float(growth_blocks_used)
        return admitted

    def _bytes_to_mb(self, value: int) -> float:
        """Convert a byte estimate to megabytes for reporting."""

        return float(value) / (1024.0 * 1024.0)

    def _prefill_prompt_caches(
        self,
        generation_model: Any,
        prompt_ids: list[torch.Tensor],
        prompt_masks: list[torch.Tensor],
    ) -> tuple[list[Any], torch.Tensor]:
        """Prefill each prompt once and return per-sequence cache state plus next-token logits."""

        sequence_caches: list[Any] = []
        next_logits: list[torch.Tensor] = []

        for prompt, prompt_mask in zip(prompt_ids, prompt_masks, strict=True):
            prompt_batch = prompt.unsqueeze(0)
            prompt_mask_batch = prompt_mask.unsqueeze(0)
            if prompt.numel() > self.config.prefill_chunk_size:
                sequence_logits, sequence_cache = self.chunked_prefill_for_generation(
                    model=generation_model,
                    prompt_ids=prompt_batch,
                    chunk_size=self.config.prefill_chunk_size,
                    attention_mask=prompt_mask_batch,
                )
            else:
                outputs = generation_model(
                    input_ids=prompt_batch,
                    attention_mask=prompt_mask_batch,
                    use_cache=True,
                )
                sequence_logits = outputs.logits[:, -1, :]
                sequence_cache = outputs.past_key_values
            next_logits.append(sequence_logits)
            sequence_caches.append(sequence_cache)

        return sequence_caches, torch.cat(next_logits, dim=0)

    def _prime_with_chunked_prefill(
        self,
        prompt_ids: list[torch.Tensor],
        prompt_masks: list[torch.Tensor],
        generated_ids: list[torch.Tensor],
        current_ids: list[torch.Tensor],
        finished: list[bool],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[bool]]:
        """Generate the first token via chunked prefill for long prompts.

        This reduces prompt-side prefill spikes in the live continuous batching
        path. The remaining decode steps still use the existing active-sequence
        scheduler, so the integration improves the hot path without requiring a
        full cached continuous-decoding engine.
        """

        generation_model = self.layout.model
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        for index, (prompt, prompt_mask) in enumerate(zip(prompt_ids, prompt_masks, strict=True)):
            if prompt.numel() <= self.config.prefill_chunk_size:
                continue
            next_token_logits, _past_key_values = self.chunked_prefill_for_generation(
                model=generation_model,
                prompt_ids=prompt.unsqueeze(0),
                chunk_size=self.config.prefill_chunk_size,
                attention_mask=prompt_mask.unsqueeze(0),
            )
            next_token = self._sample_next_token(next_token_logits)
            token_tensor = next_token.to(dtype=torch.long, device=self.device)
            generated_ids[index] = token_tensor
            current_ids[index] = torch.cat((current_ids[index], token_tensor), dim=0)
            if eos_token_id is not None and bool((token_tensor == eos_token_id).all().item()):
                finished[index] = True
        return generated_ids, current_ids, finished

    def _sample_next_tokens(
        self,
        active_sequences: list[torch.Tensor],
        active_indices: list[int],
    ) -> torch.Tensor:
        """Sample one token for each currently active sequence."""

        generation_model = self.layout.model
        if hasattr(generation_model, "generate_step"):
            sampled = generation_model.generate_step(active_sequences=active_sequences, active_indices=active_indices)
            if isinstance(sampled, torch.Tensor):
                return sampled.to(self.device)
            return torch.tensor(sampled, dtype=torch.long, device=self.device)

        chunk_size = self.config.chunk_size or len(active_sequences)
        chunk_outputs: list[torch.Tensor] = []
        for start in range(0, len(active_sequences), chunk_size):
            chunk_sequences = active_sequences[start : start + chunk_size]
            padded = torch.nn.utils.rnn.pad_sequence(
                chunk_sequences,
                batch_first=True,
                padding_value=int(self.tokenizer.pad_token_id),
            )
            attention_mask = (padded != int(self.tokenizer.pad_token_id)).long()
            logits = generation_model(input_ids=padded, attention_mask=attention_mask).logits
            next_token_logits = logits[:, -1, :]

            sampled = self._sample_next_token(next_token_logits)
            chunk_outputs.append(sampled)
        return torch.cat(chunk_outputs, dim=0)

    def _supports_persistent_kv_decode(self, generation_model: Any) -> bool:
        """Return whether the model exposes a forward suitable for cached decode."""

        forward = getattr(type(generation_model), "forward", None)
        return forward is not None and forward is not nn.Module.forward

    def _stack_past_key_values(self, caches: list[Any]) -> Any:
        """Batch multiple single-sequence cache objects into one cache."""

        if not caches:
            raise ValueError("Cannot stack an empty cache list.")
        if len(caches) == 1:
            return caches[0]

        sample_cache = caches[0]
        if isinstance(sample_cache, tuple):
            return self._stack_legacy_cache(caches)
        legacy_caches = [self._cache_to_legacy(cache) for cache in caches]
        stacked = self._stack_legacy_cache(legacy_caches)
        return self._cache_from_legacy(sample_cache, stacked)

    def _split_past_key_values(self, cache: Any, batch_size: int) -> list[Any]:
        """Split a batched cache back into one cache object per sequence."""

        if batch_size <= 0:
            return []
        if batch_size == 1:
            return [cache]

        if isinstance(cache, tuple):
            return self._split_legacy_cache(cache)
        legacy_cache = self._cache_to_legacy(cache)
        return [
            self._cache_from_legacy(cache, sequence_cache)
            for sequence_cache in self._split_legacy_cache(legacy_cache)
        ]

    def _cache_to_legacy(
        self,
        cache: Any,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Normalize supported cache objects into the legacy tuple format."""

        if isinstance(cache, tuple):
            return cache

        to_legacy_cache = getattr(cache, "to_legacy_cache", None)
        if callable(to_legacy_cache):
            return to_legacy_cache()

        layers = getattr(cache, "layers", None)
        if layers is not None:
            legacy_layers = []
            for layer in layers:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None:
                    keys = getattr(layer, "key_cache", None)
                    values = getattr(layer, "value_cache", None)
                if keys is None or values is None:
                    break
                legacy_layers.append((keys, values))
            else:
                return tuple(legacy_layers)

        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            return tuple(
                (keys, values)
                for keys, values in zip(key_cache, value_cache, strict=True)
            )

        raise TypeError(f"Unsupported cache type for conversion: {type(cache)!r}")

    def _cache_from_legacy(
        self,
        cache_like: Any,
        legacy_cache: tuple[tuple[torch.Tensor, ...], ...],
    ) -> Any:
        """Rebuild a cache object from the legacy tuple format when supported."""

        if isinstance(cache_like, tuple):
            return legacy_cache

        cache_type = type(cache_like)
        from_legacy_cache = getattr(cache_type, "from_legacy_cache", None)
        if not callable(from_legacy_cache):
            from_legacy_cache = getattr(cache_like, "from_legacy_cache", None)
        if callable(from_legacy_cache):
            try:
                return from_legacy_cache(legacy_cache)
            except TypeError:
                pass

        # Newer cache implementations can often be reconstructed directly from
        # legacy `(key, value)` tuples via their constructor.
        try:
            return cache_type(ddp_cache_data=legacy_cache)
        except TypeError:
            pass

        config = getattr(self.layout.model, "config", None)
        if config is not None:
            try:
                return cache_type(ddp_cache_data=legacy_cache, config=config)
            except TypeError:
                pass

        raise TypeError(f"Unsupported cache type for reconstruction: {cache_type!r}")

    def _stack_legacy_cache(
        self,
        caches: list[tuple[tuple[torch.Tensor, ...], ...]],
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Concatenate legacy tuple caches along the batch dimension."""

        stacked_layers = []
        for layer_caches in zip(*caches, strict=True):
            stacked_layers.append(
                tuple(
                    torch.cat(items, dim=0)
                    for items in zip(*layer_caches, strict=True)
                )
            )
        return tuple(stacked_layers)

    def _split_legacy_cache(
        self,
        cache: tuple[tuple[torch.Tensor, ...], ...],
    ) -> list[tuple[tuple[torch.Tensor, ...], ...]]:
        """Split a batched legacy cache into one cache tuple per sequence."""

        batch_size = cache[0][0].shape[0]
        split_caches = []
        for batch_index in range(batch_size):
            split_caches.append(
                tuple(
                    tuple(state[batch_index : batch_index + 1] for state in layer_cache)
                    for layer_cache in cache
                )
            )
        return split_caches

    def _track_padding_ratio(self, padding_ratio: float) -> None:
        """Track sustained padding inefficiency and emit guidance."""

        if padding_ratio > 0.4:
            self._high_padding_streak += 1
        else:
            self._high_padding_streak = 0

        LOGGER.info("padding_ratio=%.4f", padding_ratio)
        if self._high_padding_streak >= 10:
            LOGGER.warning(
                "padding_ratio > 0.4 for 10+ consecutive generation steps. "
                "Consider sorting prompts by estimated length in the next epoch."
            )
