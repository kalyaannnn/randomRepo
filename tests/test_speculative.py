from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.core.trainer import GRPOTrainer
from agentrl.generation.speculative import SpeculativeRolloutOrchestrator


class CharTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 3
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens, return_offsets_mapping
        token_ids = torch.tensor([[ord(character) for character in text]], dtype=torch.long)
        return {
            "input_ids": token_ids,
            "attention_mask": torch.ones_like(token_ids),
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(value)) for value in token_ids.tolist() if int(value) not in {0, 3})


class SingleTurnEnvironment(BaseEnvironment):
    def reset(self) -> str:
        return "task"

    def step(self, action: str) -> tuple[str, bool]:
        del action
        return ("done", True)

    def state(self) -> dict[str, str]:
        return {"expected": "ab"}


class PrefixVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, str]) -> float:
        return 1.0 if response == env_state["expected"] else 0.0


def _next_token_for_text(text: str) -> int:
    generated = text.split("Assistant:\n")[-1]
    if generated == "":
        return ord("a")
    if generated == "a":
        return ord("b")
    return 3


class PredictiveModel(torch.nn.Module):
    def __init__(self, strong: bool = True) -> None:
        super().__init__()
        self.strong = strong
        self.config = SimpleNamespace(use_cache=False)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, 256), -8.0 if self.strong else -3.0, dtype=torch.float32)
        for batch_index in range(batch):
            sequence = input_ids[batch_index].tolist()
            for position in range(seq):
                prefix = "".join(chr(int(value)) for value in sequence[: position + 1] if int(value) != 0)
                next_token = _next_token_for_text(prefix)
                logits[batch_index, position, next_token] = 8.0 if self.strong else 3.0
        return SimpleNamespace(logits=logits)

    __call__ = forward


class Layout:
    def __init__(self) -> None:
        self.model = PredictiveModel(strong=True)

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)

    def trainable_parameters(self):
        return [self.model.anchor]


def test_break_even_calculator_matches_prompt_formula() -> None:
    speedup = SpeculativeRolloutOrchestrator.break_even_calculator(
        draft_model_size_B=0.5,
        policy_model_size_B=1.5,
        K=4,
    )

    assert speedup == pytest.approx(4 / (1 + (0.5 / 1.5)))


def test_speculative_orchestrator_collects_cached_policy_logprobs() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
        do_sample=False,
        use_speculative_decoding=True,
        draft_model_name="fake/draft",
    )
    orchestrator = SpeculativeRolloutOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        draft_model=PredictiveModel(strong=False),
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.rewards.tolist() == [[1.0, 1.0]]
    assert batch.metadata["responses"] == [["ab", "ab"]]
    assert batch.metadata["speculative_k"] == 4
    assert batch.old_policy_logprobs.abs().sum().item() > 0.0
    assert batch.completion_mask.dtype == torch.bool
    assert batch.completion_mask.any()
    assert torch.all(batch.old_policy_logprobs[~batch.completion_mask] == 0.0)
    assert batch.old_policy_logprobs[batch.completion_mask].abs().sum().item() > 0.0


def test_trainer_selects_speculative_orchestrator_from_config() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        steps=1,
        max_new_tokens=4,
        do_sample=False,
        use_continuous_batching=False,
        use_speculative_decoding=True,
        draft_model_name="fake/draft",
    )
    trainer = GRPOTrainer(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        draft_model=PredictiveModel(strong=False),
    )

    assert isinstance(trainer.rollout, SpeculativeRolloutOrchestrator)


def test_speculative_verify_draft_samples_on_prefix_device(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
        do_sample=False,
        use_speculative_decoding=True,
        draft_model_name="fake/draft",
    )
    orchestrator = SpeculativeRolloutOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        draft_model=PredictiveModel(strong=False),
        device=torch.device("cpu"),
    )

    seen: dict[str, object] = {}
    original_rand = torch.rand

    def recording_rand(*args, **kwargs):
        seen["device"] = kwargs.get("device")
        return original_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", recording_rand)

    prefix_ids = torch.tensor([[ord("t"), ord("a")]], dtype=torch.long, device=orchestrator.device)
    accepted = orchestrator._verify_draft(
        prefix_ids=prefix_ids,
        draft_tokens=[ord("a")],
        draft_logprobs=[0.0],
        draft_probs=[torch.full((256,), 1 / 256, dtype=torch.float32, device=orchestrator.device)],
    )

    assert accepted
    assert seen["device"] == prefix_ids.device
