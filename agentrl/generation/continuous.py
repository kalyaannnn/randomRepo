"""Continuous batching rollout orchestration for AgentRL."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from agentrl.core.base import BaseEnvironment
from agentrl.core.rollout import RolloutBatch, RolloutOrchestrator


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


class ContinuousBatchingOrchestrator(RolloutOrchestrator):
    """Rollout orchestrator that drops finished sequences during decoding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._high_padding_streak = 0

    def collect(self) -> RolloutBatch:
        """Collect one rollout batch using continuous per-step scheduling."""

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
            responses, padding_ratio = self._generate_active_batch(prompts)
            padding_ratios.append(padding_ratio)
            self._track_padding_ratio(padding_ratio)

            for state_index, response_text in zip(active_indices, responses, strict=True):
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
        metadata["padding_ratio"] = float(sum(padding_ratios) / len(padding_ratios)) if padding_ratios else 0.0

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

        generation_model = self.layout.model
        if self._supports_persistent_kv_decode(generation_model):
            return self._generate_active_batch_with_cache(prompt_ids, prompt_masks)
        return self._generate_active_batch_without_cache(prompt_ids, prompt_masks)

    def _generate_active_batch_with_cache(
        self,
        prompt_ids: list[torch.Tensor],
        prompt_masks: list[torch.Tensor],
    ) -> tuple[list[str], float]:
        """Generate with persistent per-sequence KV caches across active decoding."""

        generation_model = self.layout.model
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        generated_ids = [torch.empty(0, dtype=torch.long, device=self.device) for _ in prompt_ids]
        finished = [False for _ in prompt_ids]
        sequence_lengths = [int(mask.sum().item()) for mask in prompt_masks]
        sequence_caches, next_logits = self._prefill_prompt_caches(generation_model, prompt_ids, prompt_masks)
        next_logits_by_index = {
            index: next_logits[index : index + 1]
            for index in range(len(prompt_ids))
        }

        total_padding_tokens = 0
        total_step_tokens = 0

        for _ in range(self.config.max_new_tokens):
            active_indices = [index for index, is_finished in enumerate(finished) if not is_finished]
            if not active_indices:
                break

            active_logits = torch.cat([next_logits_by_index[index] for index in active_indices], dim=0)
            next_tokens = self._sample_next_token(active_logits)

            decode_buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
            for batch_offset, episode_index in enumerate(active_indices):
                token_tensor = next_tokens[batch_offset : batch_offset + 1].to(dtype=torch.long, device=self.device)
                generated_ids[episode_index] = torch.cat((generated_ids[episode_index], token_tensor), dim=0)
                token = int(token_tensor.item())
                if eos_token_id is not None and token == eos_token_id:
                    finished[episode_index] = True
                    continue
                decode_buckets.setdefault(sequence_lengths[episode_index], []).append((episode_index, token_tensor))

            if not decode_buckets:
                break

            next_logits_by_index = {}
            for sequence_length, bucket in decode_buckets.items():
                total_step_tokens += sequence_length * len(bucket)

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

        decoded = [self._postprocess_response(self.tokenizer.decode(tokens, skip_special_tokens=True)) for tokens in generated_ids]
        padding_ratio = float(total_padding_tokens / total_step_tokens) if total_step_tokens else 0.0
        return decoded, padding_ratio

    def _generate_active_batch_without_cache(
        self,
        prompt_ids: list[torch.Tensor],
        prompt_masks: list[torch.Tensor],
    ) -> tuple[list[str], float]:
        """Fallback generation path for models without a cache-aware forward."""

        current_ids = [prompt.clone() for prompt in prompt_ids]
        generated_ids = [torch.empty(0, dtype=torch.long, device=self.device) for _ in prompt_ids]
        finished = [False for _ in prompt_ids]

        total_padding_tokens = 0
        total_step_tokens = 0

        if any(prompt.numel() > self.config.prefill_chunk_size for prompt in prompt_ids):
            generated_ids, current_ids, finished = self._prime_with_chunked_prefill(
                prompt_ids=prompt_ids,
                prompt_masks=prompt_masks,
                generated_ids=generated_ids,
                current_ids=current_ids,
                finished=finished,
            )

        for _ in range(self.config.max_new_tokens):
            active_indices = [index for index, is_finished in enumerate(finished) if not is_finished]
            if not active_indices:
                break

            active_sequences = [current_ids[index] for index in active_indices]
            max_length = max(sequence.numel() for sequence in active_sequences)
            total_step_tokens += max_length * len(active_sequences)
            total_padding_tokens += sum(max_length - int(sequence.numel()) for sequence in active_sequences)

            next_tokens = self._sample_next_tokens(active_sequences, active_indices)
            for batch_offset, episode_index in enumerate(active_indices):
                token = int(next_tokens[batch_offset].item())
                token_tensor = torch.tensor([token], dtype=torch.long, device=self.device)
                generated_ids[episode_index] = torch.cat((generated_ids[episode_index], token_tensor), dim=0)
                current_ids[episode_index] = torch.cat((current_ids[episode_index], token_tensor), dim=0)
                if token == getattr(self.tokenizer, "eos_token_id", None):
                    finished[episode_index] = True

        decoded = [self._postprocess_response(self.tokenizer.decode(tokens, skip_special_tokens=True)) for tokens in generated_ids]
        padding_ratio = float(total_padding_tokens / total_step_tokens) if total_step_tokens else 0.0
        return decoded, padding_ratio

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
        if hasattr(sample_cache, "to_legacy_cache") and hasattr(type(sample_cache), "from_legacy_cache"):
            legacy_caches = [cache.to_legacy_cache() for cache in caches]
            stacked = self._stack_legacy_cache(legacy_caches)
            return type(sample_cache).from_legacy_cache(stacked)
        if isinstance(sample_cache, tuple):
            return self._stack_legacy_cache(caches)
        raise TypeError(f"Unsupported cache type for batching: {type(sample_cache)!r}")

    def _split_past_key_values(self, cache: Any, batch_size: int) -> list[Any]:
        """Split a batched cache back into one cache object per sequence."""

        if batch_size <= 0:
            return []
        if batch_size == 1:
            return [cache]

        if hasattr(cache, "to_legacy_cache") and hasattr(type(cache), "from_legacy_cache"):
            legacy_cache = cache.to_legacy_cache()
            return [
                type(cache).from_legacy_cache(sequence_cache)
                for sequence_cache in self._split_legacy_cache(legacy_cache)
            ]
        if isinstance(cache, tuple):
            return self._split_legacy_cache(cache)
        raise TypeError(f"Unsupported cache type for splitting: {type(cache)!r}")

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
