"""Standard rollout orchestration for AgentRL."""

from __future__ import annotations

import copy
import logging
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
    action_mask: torch.Tensor
    policy_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
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

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

    def collect(self) -> RolloutBatch:
        """Run one rollout phase and return a batched multi-turn training batch."""

        episodes: list[dict[str, Any]] = []
        for _ in range(self.config.batch_size):
            root_env = self._clone_environment(self.environment)
            initial_observation = root_env.reset()
            prompt_group: list[dict[str, Any]] = []
            for _ in range(self.config.group_size):
                env = self._clone_environment(root_env)
                prompt_group.append(self._run_episode(env, initial_observation))
            episodes.extend(prompt_group)

        input_ids, attention_mask, action_mask = self._pack_sequences(episodes)
        flat_input_ids = input_ids.view(-1, input_ids.shape[-1])
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        flat_action_mask = action_mask.view(-1, action_mask.shape[-1])

        model_config = getattr(self.layout.model, "config", None)
        if model_config is not None:
            model_config.use_cache = True

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
            action_mask=action_mask,
            policy_logprobs=policy_sequences.view_as(input_ids),
            ref_logprobs=ref_sequences.view_as(input_ids),
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

        generation_model = self.layout.model
        generation_model.config.use_cache = True
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        with torch.no_grad():
            generated = generation_model.generate(**generate_kwargs)

        prompt_length = input_ids.shape[-1]
        response_ids = generated[:, prompt_length:]
        return self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

    def _pack_sequences(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize transcripts and pad them into `[B, G, L]` tensors."""

        tokenized: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_length = 0
        for episode in episodes:
            seq_ids, seq_attention, seq_action = self._tokenize_transcript(
                episode["transcript_text"],
                episode["assistant_spans"],
            )
            tokenized.append((seq_ids, seq_attention, seq_action))
            max_length = max(max_length, int(seq_ids.numel()))

        if self.config.pad_to_multiple_of is not None and max_length > 0:
            remainder = max_length % self.config.pad_to_multiple_of
            if remainder != 0:
                max_length += self.config.pad_to_multiple_of - remainder

        padded_ids: list[torch.Tensor] = []
        padded_attention: list[torch.Tensor] = []
        padded_action: list[torch.Tensor] = []
        pad_token_id = int(self.tokenizer.pad_token_id)

        for seq_ids, seq_attention, seq_action in tokenized:
            pad_amount = max_length - int(seq_ids.numel())
            padded_ids.append(torch.nn.functional.pad(seq_ids, (0, pad_amount), value=pad_token_id))
            padded_attention.append(torch.nn.functional.pad(seq_attention, (0, pad_amount), value=0))
            padded_action.append(torch.nn.functional.pad(seq_action, (0, pad_amount), value=0))

        stacked_ids = torch.stack(padded_ids, dim=0).to(self.device)
        stacked_attention = torch.stack(padded_attention, dim=0).to(self.device)
        stacked_action = torch.stack(padded_action, dim=0).to(self.device)
        shape = (self.config.batch_size, self.config.group_size, max_length)
        return (
            stacked_ids.view(shape),
            stacked_attention.view(shape),
            stacked_action.view(shape),
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

        action_mask = torch.zeros(input_ids.shape[-1], dtype=torch.bool)
        for token_index, (start, end) in enumerate(offsets[0].tolist()):
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in assistant_spans):
                action_mask[token_index] = True

        return input_ids, attention_mask, action_mask

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
        action_mask_parts: list[torch.Tensor] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            segment = transcript_text[start:end]
            if not segment:
                continue
            encoded = self.tokenizer(segment, return_tensors="pt", add_special_tokens=False)
            segment_ids = encoded["input_ids"][0].to(dtype=torch.long)
            is_action = any(start >= span_start and end <= span_end for span_start, span_end in assistant_spans)
            token_ids_parts.append(segment_ids)
            action_mask_parts.append(torch.full((segment_ids.numel(),), is_action, dtype=torch.bool))

        input_ids = torch.cat(token_ids_parts, dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        action_mask = torch.cat(action_mask_parts, dim=0)
        return input_ids, attention_mask, action_mask

    def _compute_logprobs(
        self,
        forward_fn: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token logprobs aligned to sequence positions."""

        logits = forward_fn(input_ids=input_ids, attention_mask=attention_mask)
        token_logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        next_tokens = input_ids[:, 1:].unsqueeze(-1)
        gathered = token_logprobs.gather(dim=-1, index=next_tokens).squeeze(-1)

        aligned = torch.zeros_like(input_ids, dtype=logits.dtype)
        aligned[:, 1:] = gathered * action_mask[:, 1:].to(dtype=logits.dtype)
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

        return {
            "prompts": grouped_prompts,
            "responses": grouped_responses,
            "transcripts": grouped_transcripts,
            "unique_response_ratio": unique_response_ratio,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
        }

    def _infer_device(self) -> torch.device:
        """Infer the execution device from the attached model layout."""

        parameter = next(self.layout.model.parameters(), None)
        if parameter is None:
            return torch.device("cpu")
        return parameter.device
