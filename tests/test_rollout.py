from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.core.rollout import RolloutOrchestrator


class CharTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 3

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        input_ids = torch.tensor([[ord(character) for character in text]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        encoded: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = torch.tensor(
                [[(index, index + 1) for index, _ in enumerate(text)]],
                dtype=torch.long,
            )
        return encoded

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        values = token_ids.tolist()
        return "".join(chr(value) for value in values if value != self.pad_token_id)


class TwoTurnEnvironment(BaseEnvironment):
    def __init__(self) -> None:
        self.turn = 0

    def reset(self) -> str:
        self.turn = 0
        return "start"

    def step(self, action: str) -> tuple[str, bool]:
        if self.turn == 0:
            self.turn += 1
            return ("finish", False)
        self.turn += 1
        return ("done", True)

    def state(self) -> dict[str, str]:
        return {"expected": "done"}


class FinalAnswerVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, str]) -> float:
        return 1.0 if response == env_state["expected"] else 0.0


class FakeGenerationModel(torch.nn.Module):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self.outputs = list(outputs)
        self.config = SimpleNamespace(use_cache=False)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        del attention_mask, max_new_tokens, temperature, do_sample, pad_token_id, eos_token_id
        text = self.outputs.pop(0)
        response = torch.tensor([[ord(character) for character in text]], dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, response), dim=-1)


class FakeLayout:
    def __init__(self, outputs: list[str]) -> None:
        self.model = FakeGenerationModel(outputs)

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 256
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float32)
        logits[:, :, 65] = 1.0
        return logits


class ChunkedPrefillModel(torch.nn.Module):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.output = [ord(character) for character in output]
        self.config = SimpleNamespace(use_cache=False)
        self.forward_calls: list[tuple[int, int | None]] = []
        self.generate_called = False
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: int | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        marker = past_key_values
        self.forward_calls.append((input_ids.shape[-1], marker))
        vocab = 256
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab), -1e9, dtype=torch.float32)
        token = self.output[0] if input_ids.shape[-1] > 1 else self.output[1]
        logits[:, -1, token] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=len(self.forward_calls))

    def generate(self, **kwargs) -> torch.Tensor:
        del kwargs
        self.generate_called = True
        raise AssertionError("generate() should not be used when chunked prefill is active.")


class ChunkedPrefillLayout:
    def __init__(self, output: str) -> None:
        self.model = ChunkedPrefillModel(output)

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)


def test_rollout_collects_multi_turn_grouped_batch() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=8,
        max_episode_steps=3,
    )
    layout = FakeLayout(outputs=["plan", "done", "plan", "bad"])
    orchestrator = RolloutOrchestrator(
        config=config,
        environment=TwoTurnEnvironment(),
        verifier=FinalAnswerVerifier(),
        tokenizer=CharTokenizer(),
        layout=layout,
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.input_ids.shape[0:2] == (1, 2)
    assert batch.input_ids.shape[-1] % config.pad_to_multiple_of == 0
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.completion_mask.shape == batch.input_ids.shape
    assert batch.old_policy_logprobs.shape == batch.input_ids.shape
    assert batch.rewards.tolist() == [[1.0, 0.0]]
    assert batch.advantages[0, 0].item() == pytest.approx(1.0, rel=1e-5)
    assert batch.advantages[0, 1].item() == pytest.approx(-1.0, rel=1e-5)
    assert batch.completion_mask.sum().item() > 0
    assert batch.metadata["responses"] == [["done", "bad"]]
    assert batch.metadata["prefill_tokens"] > 0.0
    assert batch.metadata["decode_tokens"] > 0.0
    assert "prefill_tokens_per_second" in batch.metadata
    assert "decode_tokens_per_second" in batch.metadata
    assert "padding_waste_tokens" in batch.metadata


def test_rollout_completion_mask_and_old_policy_logprobs_follow_completion_tokens() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=8,
        max_episode_steps=3,
    )
    layout = FakeLayout(outputs=["plan", "done", "plan", "bad"])
    orchestrator = RolloutOrchestrator(
        config=config,
        environment=TwoTurnEnvironment(),
        verifier=FinalAnswerVerifier(),
        tokenizer=CharTokenizer(),
        layout=layout,
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    transcript = batch.metadata["transcripts"][0][0]
    expected_mask = torch.zeros_like(batch.completion_mask[0, 0], dtype=torch.bool)
    for completion_text in ("plan", "done"):
        start = transcript.index(completion_text)
        end = start + len(completion_text)
        expected_mask[start:end] = True

    assert transcript.startswith("Observation:\nstart\n\nAssistant:\nplan")
    assert torch.equal(batch.completion_mask[0, 0], expected_mask)
    assert batch.completion_mask[0, 0, 0].item() is False
    assert torch.all(batch.old_policy_logprobs.masked_select(~batch.completion_mask) == 0)
    assert torch.any(batch.old_policy_logprobs.masked_select(batch.completion_mask) != 0)


def test_rollout_warns_when_episode_hits_turn_cap(caplog: pytest.LogCaptureFixture) -> None:
    class NeverDoneEnvironment(BaseEnvironment):
        def __init__(self) -> None:
            self.counter = 0

        def reset(self) -> str:
            self.counter = 0
            return "start"

        def step(self, action: str) -> tuple[str, bool]:
            del action
            self.counter += 1
            return (f"obs-{self.counter}", False)

        def state(self) -> dict[str, str]:
            return {"expected": "never"}

    caplog.set_level("WARNING")
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_episode_steps=1,
    )
    layout = FakeLayout(outputs=["x", "y"])
    orchestrator = RolloutOrchestrator(
        config=config,
        environment=NeverDoneEnvironment(),
        verifier=FinalAnswerVerifier(),
        tokenizer=CharTokenizer(),
        layout=layout,
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert "max_episode_steps=1" in caplog.text
    assert batch.metadata["responses"] == [["x", "y"]]


def test_rollout_uses_chunked_prefill_for_long_prompts() -> None:
    class LongPromptEnvironment(BaseEnvironment):
        def reset(self) -> str:
            return "x" * 12

        def step(self, action: str) -> tuple[str, bool]:
            del action
            return ("done", True)

        def state(self) -> dict[str, str]:
            return {"expected": "ok"}

    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=2,
        prefill_chunk_size=4,
        do_sample=False,
    )
    layout = ChunkedPrefillLayout(output="ok")
    orchestrator = RolloutOrchestrator(
        config=config,
        environment=LongPromptEnvironment(),
        verifier=FinalAnswerVerifier(),
        tokenizer=CharTokenizer(),
        layout=layout,
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.metadata["responses"] == [["ok", "ok"]]
    assert batch.metadata["prefill_time_ms"] >= 0.0
    assert batch.metadata["decode_time_ms"] >= 0.0
    assert batch.metadata["cache_reuse_tokens"] > 0.0
    assert batch.metadata["cache_reuse_effectiveness"] > 0.0
    assert layout.model.generate_called is False
    assert layout.model.forward_calls[:3] == [(4, None), (4, 1), (4, 2)]
