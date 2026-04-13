"""Speculative decoding rollout orchestration for AgentRL."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from agentrl.core.base import BaseEnvironment
from agentrl.core.config import GRPOConfig
from agentrl.core.rollout import RolloutBatch, RolloutOrchestrator


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _SpeculativeStep:
    """Accepted rollout token plus its policy log probability."""

    token_id: int
    policy_logprob: float


class SpeculativeRolloutOrchestrator(RolloutOrchestrator):
    """Rollout orchestrator that uses a frozen draft model for speculative decoding."""

    def __init__(
        self,
        config: GRPOConfig,
        environment: BaseEnvironment,
        verifier: Any,
        tokenizer: Any,
        layout: Any,
        device: torch.device | None = None,
        rng: torch.Generator | None = None,
        draft_model: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            environment=environment,
            verifier=verifier,
            tokenizer=tokenizer,
            layout=layout,
            device=device,
            rng=rng,
        )
        self.draft_model = draft_model or self._load_draft_model()

    @staticmethod
    def break_even_calculator(
        draft_model_size_B: float,
        policy_model_size_B: float,
        K: int,
    ) -> float:
        """Estimate speculative-decoding speedup from the project prompt formula."""

        if draft_model_size_B <= 0 or policy_model_size_B <= 0:
            raise ValueError("Model sizes must be positive.")
        if K <= 0:
            raise ValueError("K must be positive.")

        draft_to_policy = draft_model_size_B / policy_model_size_B
        return K / (1.0 + draft_to_policy)

    def collect(self) -> RolloutBatch:
        """Collect a rollout batch using speculative decoding."""

        episodes: list[dict[str, Any]] = []
        for _ in range(self.config.batch_size):
            root_env = self._clone_environment(self.environment)
            initial_observation = root_env.reset()
            prompt_group: list[dict[str, Any]] = []
            for _ in range(self.config.group_size):
                env = self._clone_environment(root_env)
                prompt_group.append(self._run_episode(env, initial_observation))
            episodes.extend(prompt_group)

        input_ids, attention_mask, action_mask, policy_logprobs = self._pack_speculative_sequences(episodes)
        flat_input_ids = input_ids.view(-1, input_ids.shape[-1])
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        flat_action_mask = action_mask.view(-1, action_mask.shape[-1])

        with torch.no_grad():
            ref_logprobs = self._compute_logprobs(
                self.layout.reference_forward,
                flat_input_ids,
                flat_attention_mask,
                flat_action_mask,
            )

        rewards = torch.tensor(
            [[episode["reward"] for episode in episodes[i : i + self.config.group_size]]
             for i in range(0, len(episodes), self.config.group_size)],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = self._compute_advantages(rewards)
        metadata = self._build_metadata(episodes, rewards)
        metadata["speculative_k"] = self.config.speculative_k

        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            policy_logprobs=policy_logprobs,
            ref_logprobs=ref_logprobs.view_as(input_ids),
            rewards=rewards,
            advantages=advantages,
            metadata=metadata,
        )

    def _run_episode(self, env: BaseEnvironment, initial_observation: str) -> dict[str, Any]:
        """Generate one episode using speculative decoding per assistant turn."""

        observations = [initial_observation]
        actions: list[str] = []
        turn_token_ids: list[torch.Tensor] = []
        turn_policy_logprobs: list[torch.Tensor] = []
        truncated = False

        for _ in range(self.config.max_episode_steps):
            prompt_text = self._render_generation_prompt(observations, actions)
            generated_ids, generated_logprobs = self._generate_speculative_tokens(prompt_text)
            response_text = self._postprocess_response(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
            actions.append(response_text)
            turn_token_ids.append(generated_ids.clone())
            turn_policy_logprobs.append(generated_logprobs.clone())
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
            "turn_token_ids": turn_token_ids,
            "turn_policy_logprobs": turn_policy_logprobs,
        }

    def _generate_speculative_tokens(self, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one response with draft proposals and policy verification."""

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(prompt_ids)).to(self.device)

        if self.config.max_prompt_tokens is not None and prompt_ids.shape[-1] > self.config.max_prompt_tokens:
            prompt_ids = prompt_ids[:, -self.config.max_prompt_tokens :]
            attention_mask = attention_mask[:, -self.config.max_prompt_tokens :]

        generated: list[int] = []
        policy_logprobs: list[float] = []

        while len(generated) < self.config.max_new_tokens:
            prefix_ids = torch.tensor([prompt_ids[0].tolist() + generated], dtype=torch.long, device=self.device)
            draft_tokens, draft_logprobs, draft_probs = self._draft_propose(prefix_ids)
            accepted = self._verify_draft(prefix_ids, draft_tokens, draft_logprobs, draft_probs)
            if not accepted:
                accepted = [self._direct_policy_step(prefix_ids)]

            for step in accepted:
                generated.append(step.token_id)
                policy_logprobs.append(step.policy_logprob)
                if step.token_id == getattr(self.tokenizer, "eos_token_id", None):
                    return (
                        torch.tensor(generated, dtype=torch.long, device=self.device),
                        torch.tensor(policy_logprobs, dtype=torch.float32, device=self.device),
                    )
                if len(generated) >= self.config.max_new_tokens:
                    break

        return (
            torch.tensor(generated, dtype=torch.long, device=self.device),
            torch.tensor(policy_logprobs, dtype=torch.float32, device=self.device),
        )

    def _draft_propose(
        self,
        prefix_ids: torch.Tensor,
    ) -> tuple[list[int], list[float], list[torch.Tensor]]:
        """Have the draft model autoregressively propose up to `K` tokens."""

        proposed_tokens: list[int] = []
        proposed_logprobs: list[float] = []
        proposed_probs: list[torch.Tensor] = []
        current = prefix_ids

        for _ in range(min(self.config.speculative_k, self.config.max_new_tokens)):
            attention_mask = torch.ones_like(current)
            outputs = self.draft_model(input_ids=current, attention_mask=attention_mask)
            next_logits = outputs.logits[:, -1, :]
            next_probs = torch.softmax(next_logits / self._effective_temperature(), dim=-1)
            next_token = self._sample_from_probs(next_probs[0])
            proposed_tokens.append(next_token)
            proposed_logprobs.append(float(torch.log(next_probs[0, next_token] + 1e-12).item()))
            proposed_probs.append(next_probs[0].detach())

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            current = torch.cat((current, next_token_tensor), dim=-1)
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

        return proposed_tokens, proposed_logprobs, proposed_probs

    def _verify_draft(
        self,
        prefix_ids: torch.Tensor,
        draft_tokens: list[int],
        draft_logprobs: list[float],
        draft_probs: list[torch.Tensor],
    ) -> list[_SpeculativeStep]:
        """Verify draft proposals against the policy and return accepted output tokens."""

        if not draft_tokens:
            return []

        verify_input = torch.tensor(
            [prefix_ids[0].tolist() + draft_tokens],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(verify_input)
        logits = self.layout.policy_forward(input_ids=verify_input, attention_mask=attention_mask)
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        accepted_steps: list[_SpeculativeStep] = []
        prefix_length = prefix_ids.shape[-1]
        for index, (token_id, draft_logprob, draft_prob_vector) in enumerate(
            zip(draft_tokens, draft_logprobs, draft_probs, strict=True)
        ):
            policy_position = prefix_length - 1 + index
            policy_logprob = float(logprobs[0, policy_position, token_id].item())
            acceptance = min(1.0, float(torch.exp(torch.tensor(policy_logprob - draft_logprob)).item()))
            if torch.rand(1, generator=self.rng).item() <= acceptance:
                accepted_steps.append(_SpeculativeStep(token_id=token_id, policy_logprob=policy_logprob))
                if token_id == getattr(self.tokenizer, "eos_token_id", None):
                    return accepted_steps
                continue

            correction_probs = (probs[0, policy_position, :] - draft_prob_vector.to(self.device)).clamp(min=0)
            if float(correction_probs.sum().item()) <= 0:
                correction_probs = probs[0, policy_position, :]
            else:
                correction_probs = correction_probs / correction_probs.sum()

            corrected_token = self._sample_from_probs(correction_probs)
            corrected_logprob = float(logprobs[0, policy_position, corrected_token].item())
            accepted_steps.append(_SpeculativeStep(token_id=corrected_token, policy_logprob=corrected_logprob))
            return accepted_steps

        bonus_position = verify_input.shape[-1] - 1
        bonus_probs = probs[0, bonus_position, :]
        bonus_logprobs = logprobs[0, bonus_position, :]
        bonus_token = self._sample_from_probs(bonus_probs)
        accepted_steps.append(
            _SpeculativeStep(
                token_id=bonus_token,
                policy_logprob=float(bonus_logprobs[bonus_token].item()),
            )
        )
        return accepted_steps

    def _direct_policy_step(self, prefix_ids: torch.Tensor) -> _SpeculativeStep:
        """Fallback one-token policy sample when draft proposals are unavailable."""

        attention_mask = torch.ones_like(prefix_ids)
        logits = self.layout.policy_forward(input_ids=prefix_ids, attention_mask=attention_mask)
        last_logprobs = torch.log_softmax(logits[:, -1, :], dim=-1)
        token_id = self._sample_from_probs(last_logprobs.exp()[0])
        return _SpeculativeStep(token_id=token_id, policy_logprob=float(last_logprobs[0, token_id].item()))

    def _pack_speculative_sequences(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble piecewise tokenized transcripts with cached policy logprobs."""

        sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_length = 0
        for episode in episodes:
            assembled = self._assemble_episode_sequence(
                observations=episode["observations"],
                turn_token_ids=episode["turn_token_ids"],
                turn_policy_logprobs=episode["turn_policy_logprobs"],
            )
            sequences.append(assembled)
            max_length = max(max_length, int(assembled[0].numel()))

        padded_ids: list[torch.Tensor] = []
        padded_attention: list[torch.Tensor] = []
        padded_action: list[torch.Tensor] = []
        padded_policy_lps: list[torch.Tensor] = []
        pad_token_id = int(self.tokenizer.pad_token_id)

        for seq_ids, seq_attention, seq_action, seq_policy_lps in sequences:
            pad_amount = max_length - int(seq_ids.numel())
            padded_ids.append(torch.nn.functional.pad(seq_ids, (0, pad_amount), value=pad_token_id))
            padded_attention.append(torch.nn.functional.pad(seq_attention, (0, pad_amount), value=0))
            padded_action.append(torch.nn.functional.pad(seq_action, (0, pad_amount), value=0))
            padded_policy_lps.append(torch.nn.functional.pad(seq_policy_lps, (0, pad_amount), value=0.0))

        shape = (self.config.batch_size, self.config.group_size, max_length)
        return (
            torch.stack(padded_ids, dim=0).to(self.device).view(shape),
            torch.stack(padded_attention, dim=0).to(self.device).view(shape),
            torch.stack(padded_action, dim=0).to(self.device).view(shape),
            torch.stack(padded_policy_lps, dim=0).to(self.device).view(shape),
        )

    def _assemble_episode_sequence(
        self,
        observations: list[str],
        turn_token_ids: list[torch.Tensor],
        turn_policy_logprobs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one full transcript sequence with aligned action-token logprobs."""

        ids_parts: list[torch.Tensor] = []
        action_parts: list[torch.Tensor] = []
        policy_parts: list[torch.Tensor] = []

        for index, observation in enumerate(observations):
            for text in ("Observation:\n", observation, "\n\n"):
                token_ids = self._tokenize_text_ids(text)
                ids_parts.append(token_ids)
                action_parts.append(torch.zeros(token_ids.shape[-1], dtype=torch.bool))
                policy_parts.append(torch.zeros(token_ids.shape[-1], dtype=torch.float32))

            if index < len(turn_token_ids):
                assistant_prefix = self._tokenize_text_ids("Assistant:\n")
                ids_parts.append(assistant_prefix)
                action_parts.append(torch.zeros(assistant_prefix.shape[-1], dtype=torch.bool))
                policy_parts.append(torch.zeros(assistant_prefix.shape[-1], dtype=torch.float32))

                action_ids = turn_token_ids[index].to(dtype=torch.long, device="cpu")
                action_logprobs = turn_policy_logprobs[index].to(dtype=torch.float32, device="cpu")
                ids_parts.append(action_ids)
                action_parts.append(torch.ones(action_ids.shape[-1], dtype=torch.bool))
                policy_parts.append(action_logprobs)

                separator = self._tokenize_text_ids("\n\n")
                ids_parts.append(separator)
                action_parts.append(torch.zeros(separator.shape[-1], dtype=torch.bool))
                policy_parts.append(torch.zeros(separator.shape[-1], dtype=torch.float32))

        input_ids = torch.cat(ids_parts, dim=0)
        attention_mask = torch.ones(input_ids.shape[-1], dtype=torch.long)
        action_mask = torch.cat(action_parts, dim=0)
        policy_logprobs = torch.cat(policy_parts, dim=0)
        return input_ids, attention_mask, action_mask, policy_logprobs

    def _tokenize_text_ids(self, text: str) -> torch.Tensor:
        """Tokenize a plain-text segment into one-dimensional token ids."""

        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"][0].to(dtype=torch.long, device="cpu")

    def _sample_from_probs(self, probs: torch.Tensor) -> int:
        """Sample or greedily select one token id from a probability vector."""

        if self.config.do_sample and self.config.temperature > 0:
            return int(torch.multinomial(probs, num_samples=1, generator=self.rng).item())
        return int(torch.argmax(probs).item())

    def _effective_temperature(self) -> float:
        """Clamp the temperature to a positive value for speculative sampling math."""

        return self.config.temperature if self.config.temperature > 0 else 1.0

    def _load_draft_model(self) -> Any:
        """Load the frozen draft model used for speculative proposals."""

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Speculative decoding requires `transformers` to load the draft model.") from exc

        dtype = getattr(torch, self.config.dtype)
        draft_model = AutoModelForCausalLM.from_pretrained(
            self.config.draft_model_name,
            torch_dtype=dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        draft_model.to(self.device)
        draft_model.eval()
        for parameter in draft_model.parameters():
            parameter.requires_grad = False
        return draft_model
