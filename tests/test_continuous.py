from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.generation.continuous import ContinuousBatchingOrchestrator

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover - transformers is a required dependency in normal test runs
    DynamicCache = None


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
        return "".join(chr(int(value)) for value in token_ids.tolist() if value not in {0, 3})


class SingleTurnEnvironment(BaseEnvironment):
    def __init__(self, label: str = "task") -> None:
        self.label = label

    def reset(self) -> str:
        return self.label

    def step(self, action: str) -> tuple[str, bool]:
        del action
        return ("done", True)

    def state(self) -> dict[str, str]:
        return {"expected": "ab"}


class PrefixVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, str]) -> float:
        return 1.0 if response.startswith(env_state["expected"]) else 0.0


class StepModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=False,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=8,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate_step(self, active_sequences: list[torch.Tensor], active_indices: list[int]) -> list[int]:
        del active_indices
        outputs = []
        for sequence in active_sequences:
            text = "".join(chr(int(value)) for value in sequence.tolist() if value != 0)
            generated = text.split("Assistant:\n")[-1]
            if generated == "":
                outputs.append(ord("a"))
            elif generated == "a":
                outputs.append(ord("b"))
            else:
                outputs.append(3)
        return outputs


class Layout:
    def __init__(self) -> None:
        self.model = StepModel()

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)


class ConstantStepModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=False,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=8,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate_step(self, active_sequences: list[torch.Tensor], active_indices: list[int]) -> list[int]:
        del active_sequences
        return [ord("a")] * len(active_indices)


class ConstantLayout:
    def __init__(self) -> None:
        self.model = ConstantStepModel()

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)


class ChunkedStepModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=False,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=8,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.prefill_calls: list[tuple[int, int | None]] = []
        self.decode_calls: list[tuple[int, int | None, int]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        past_length = None if past_key_values is None else int(past_key_values[0][0].shape[2])
        if input_ids.shape[-1] > 1:
            self.prefill_calls.append((input_ids.shape[-1], past_length))
        else:
            self.decode_calls.append((input_ids.shape[0], input_ids.shape[-1], past_length))
        vocab = 256
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab), -1e9, dtype=torch.float32)
        if input_ids.shape[-1] > 1:
            logits[:, -1, ord("a")] = 0.0
        else:
            next_tokens = torch.where(input_ids[:, -1] == ord("a"), ord("b"), 3)
            logits[:, -1, :] = -1e9
            logits[torch.arange(input_ids.shape[0]), -1, next_tokens] = 0.0

        previous_length = 0 if past_length is None else past_length
        new_length = previous_length + input_ids.shape[-1]
        key = torch.zeros((input_ids.shape[0], 1, new_length, 1), dtype=torch.float32)
        value = torch.zeros((input_ids.shape[0], 1, new_length, 1), dtype=torch.float32)
        return SimpleNamespace(logits=logits, past_key_values=((key, value),))


class ChunkedLayout:
    def __init__(self) -> None:
        self.model = ChunkedStepModel()

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)


class DynamicCacheStepModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=False,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=8,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        past_length = 0 if past_key_values is None else int(past_key_values.get_seq_length())

        vocab = 256
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab), -1e9, dtype=torch.float32)
        if past_length == 0:
            logits[:, -1, ord("a")] = 0.0
        else:
            next_tokens = torch.where(input_ids[:, -1] == ord("a"), ord("b"), 3)
            logits[torch.arange(input_ids.shape[0]), -1, next_tokens] = 0.0

        new_length = past_length + input_ids.shape[-1]
        key = torch.zeros((input_ids.shape[0], 1, new_length, 1), dtype=torch.float32)
        value = torch.zeros((input_ids.shape[0], 1, new_length, 1), dtype=torch.float32)
        return SimpleNamespace(
            logits=logits,
            past_key_values=DynamicCache.from_legacy_cache(((key, value),)),
        )


class DynamicCacheLayout:
    def __init__(self) -> None:
        self.model = DynamicCacheStepModel()

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        batch, seq = input_ids.shape
        return torch.zeros((batch, seq, 256), dtype=torch.float32)


class ConstructorCache:
    def __init__(self, ddp_cache_data=None, config=None) -> None:
        self._legacy = tuple(ddp_cache_data or ())
        self.config = config

    def to_legacy_cache(self):
        return self._legacy


def test_continuous_batching_collects_and_reports_padding_ratio() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.rewards.tolist() == [[1.0, 1.0]]
    assert "padding_ratio" in batch.metadata
    assert 0.0 <= batch.metadata["padding_ratio"] <= 1.0
    assert "prefill_time_ms" in batch.metadata
    assert "decode_time_ms" in batch.metadata
    assert "decode_tokens_per_second" in batch.metadata
    assert "cache_reuse_effectiveness" in batch.metadata
    assert "scheduler_prefill_token_budget" in batch.metadata
    assert "scheduler_decode_token_budget" in batch.metadata
    assert "scheduler_max_concurrent_sequences" in batch.metadata
    assert "scheduler_prefill_kv_budget_mb" in batch.metadata
    assert "scheduler_decode_kv_budget_mb" in batch.metadata
    assert "scheduler_decode_admitted_kv_mb" in batch.metadata
    assert batch.metadata["scheduler_prefill_kv_budget_mb"] > 0.0
    assert batch.metadata["scheduler_decode_kv_budget_mb"] > 0.0


def test_continuous_batching_uses_chunked_prefill_for_long_prompts() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
        prefill_chunk_size=4,
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(label="x" * 12),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=ChunkedLayout(),
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.metadata["responses"] == [["ab", "ab"]]
    assert batch.metadata["prefill_tokens"] > 0.0
    assert batch.metadata["decode_tokens"] > 0.0
    assert batch.metadata["cache_reuse_tokens"] > 0.0
    assert orchestrator.layout.model.prefill_calls[:3] == [(4, None), (4, 4), (4, 8)]
    first_decode, second_decode = orchestrator.layout.model.decode_calls[:2]
    assert first_decode[:2] == (2, 1)
    assert second_decode[:2] == (2, 1)
    assert second_decode[2] == first_decode[2] + 1


def test_continuous_scheduler_defers_sequences_under_safe_policy() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=4,
        chunk_size=2,
        max_new_tokens=1,
        prefill_chunk_size=32,
        execution_policy="safe",
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=ConstantLayout(),
        device=torch.device("cpu"),
    )

    prompts = ["x" * length for length in (10, 11, 12, 13)]
    responses, padding_ratio = orchestrator._generate_active_batch(prompts)

    assert responses == ["a", "a", "a", "a"]
    assert 0.0 <= padding_ratio <= 1.0
    assert orchestrator._runtime_stats["scheduler_decode_passes"] >= 2.0
    assert orchestrator._runtime_stats["scheduler_deferred_sequences"] > 0.0
    assert orchestrator._runtime_stats["scheduler_max_concurrent_sequences"] < 4.0


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_continuous_batching_stacks_and_splits_dynamic_cache() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=DynamicCacheLayout(),
        device=torch.device("cpu"),
    )

    first = DynamicCache.from_legacy_cache(
        ((torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1), torch.zeros((1, 1, 3, 1))),)
    )
    second = DynamicCache.from_legacy_cache(
        ((torch.arange(10, 13, dtype=torch.float32).view(1, 1, 3, 1), torch.ones((1, 1, 3, 1))),)
    )

    stacked = orchestrator._stack_past_key_values([first, second])
    assert isinstance(stacked, DynamicCache)
    assert tuple(stacked.to_legacy_cache()[0][0].shape) == (2, 1, 3, 1)

    split = orchestrator._split_past_key_values(stacked, 2)
    assert len(split) == 2
    assert all(isinstance(item, DynamicCache) for item in split)
    assert torch.equal(split[0].to_legacy_cache()[0][0], first.to_legacy_cache()[0][0])
    assert torch.equal(split[1].to_legacy_cache()[0][0], second.to_legacy_cache()[0][0])


def test_continuous_batching_reconstructs_constructor_based_cache() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        device=torch.device("cpu"),
    )
    first = ConstructorCache(ddp_cache_data=((torch.zeros((1, 1, 3, 1)), torch.zeros((1, 1, 3, 1))),))
    second = ConstructorCache(ddp_cache_data=((torch.ones((1, 1, 3, 1)), torch.ones((1, 1, 3, 1))),))

    stacked = orchestrator._stack_past_key_values([first, second])
    split = orchestrator._split_past_key_values(stacked, batch_size=2)

    assert isinstance(stacked, ConstructorCache)
    assert tuple(stacked.to_legacy_cache()[0][0].shape) == (2, 1, 3, 1)
    assert all(isinstance(item, ConstructorCache) for item in split)
    assert torch.equal(split[0].to_legacy_cache()[0][0], first.to_legacy_cache()[0][0])
    assert torch.equal(split[1].to_legacy_cache()[0][0], second.to_legacy_cache()[0][0])


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_continuous_batching_collects_with_dynamic_cache_model() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=DynamicCacheLayout(),
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.rewards.tolist() == [[1.0, 1.0]]
    assert batch.metadata["responses"] == [["ab", "ab"]]
