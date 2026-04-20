from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from agentrl.core.base import BaseEnvironment, BaseVerifier
from agentrl.core.config import GRPOConfig
from agentrl.generation.continuous import ContinuousBatchingOrchestrator, _ScheduledSequence
from agentrl.generation.paged_kv import PagedKVCacheStore

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


class ConfigRequiredConstructorCache:
    def __init__(self, ddp_cache_data=None, config=None) -> None:
        if config is None:
            raise TypeError("config is required")
        self._legacy = tuple(ddp_cache_data or ())

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
    assert "scheduler_prefill_block_budget" in batch.metadata
    assert "scheduler_decode_block_budget" in batch.metadata
    assert "scheduler_prefill_admitted_blocks" in batch.metadata
    assert "scheduler_decode_admitted_blocks" in batch.metadata
    assert "scheduler_length_sort_passes" in batch.metadata
    assert "scheduler_length_sorted_sequences" in batch.metadata
    assert "paged_kv_block_size_tokens" in batch.metadata
    assert "paged_kv_free_block_count" in batch.metadata
    assert "paged_kv_used_block_count" in batch.metadata
    assert "paged_kv_allocator_occupancy" in batch.metadata
    assert "paged_kv_block_reuse_count" in batch.metadata
    assert "paged_kv_allocator_pressure" in batch.metadata
    assert "paged_kv_max_blocks_in_use" in batch.metadata
    assert "paged_kv_resident_sequences" in batch.metadata
    assert "paged_kv_preempted_sequences" in batch.metadata
    assert "paged_kv_max_preempted_sequences" in batch.metadata
    assert batch.metadata["scheduler_prefill_kv_budget_mb"] > 0.0
    assert batch.metadata["scheduler_decode_kv_budget_mb"] > 0.0
    assert batch.metadata["scheduler_prefill_block_budget"] == 0.0
    assert batch.metadata["scheduler_decode_block_budget"] == 0.0
    assert batch.metadata["scheduler_prefill_admitted_blocks"] == 0.0
    assert batch.metadata["scheduler_decode_admitted_blocks"] == 0.0
    assert batch.metadata["paged_kv_block_size_tokens"] == 0.0
    assert batch.metadata["paged_kv_free_block_count"] == 0.0
    assert batch.metadata["paged_kv_max_blocks_in_use"] == 0.0


def test_paged_kv_continuous_collects_block_metrics() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
        use_paged_kv_continuous=True,
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
    assert batch.metadata["scheduler_prefill_block_budget"] > 0.0
    assert batch.metadata["scheduler_decode_block_budget"] > 0.0
    assert batch.metadata["paged_kv_block_size_tokens"] == 16.0
    assert batch.metadata["paged_kv_free_block_count"] >= 0.0
    assert batch.metadata["paged_kv_max_blocks_in_use"] >= 1.0


def test_paged_kv_prefill_seeds_resident_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=1,
        use_paged_kv_continuous=True,
        do_sample=False,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=ChunkedLayout(),
        device=torch.device("cpu"),
    )
    scheduler = orchestrator._build_scheduler_state(active_count=2)
    prompts = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([1, 2], dtype=torch.long)]
    masks = [torch.ones_like(prompt) for prompt in prompts]
    sequences = [
        _ScheduledSequence(original_index=index, prompt_ids=prompt, prompt_mask=mask)
        for index, (prompt, mask) in enumerate(zip(prompts, masks, strict=True))
    ]

    resident_sequence_ids: list[int] = []
    original_build = orchestrator._build_paged_kv_allocator

    def capture_store(*args, **kwargs):
        store = original_build(*args, **kwargs)
        original_set_resident_cache = store.set_resident_cache

        def capture_set_resident_cache(*, sequence_id: int, cache: object, cache_template: object | None = None) -> None:
            original_set_resident_cache(
                sequence_id=sequence_id,
                cache=cache,
                cache_template=cache_template,
            )
            resident_sequence_ids.append(sequence_id)
            assert store.has_resident_cache(sequence_id) is True

        monkeypatch.setattr(store, "set_resident_cache", capture_set_resident_cache)
        return store

    monkeypatch.setattr(orchestrator, "_build_paged_kv_allocator", capture_store)

    orchestrator._generate_active_batch_with_cache(sequences, scheduler)

    assert resident_sequence_ids[:2] == [0, 1]
    assert set(resident_sequence_ids) == {0, 1}


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
        use_paged_kv_continuous=True,
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
    assert orchestrator._runtime_stats["paged_kv_preempted_sequences"] > 0.0
    assert orchestrator._runtime_stats["paged_kv_max_preempted_sequences"] > 0.0


def test_scheduler_costing_is_mode_aware() -> None:
    legacy_config = GRPOConfig(model_name="fake/model")
    paged_config = GRPOConfig(model_name="fake/model", use_paged_kv_continuous=True)
    legacy = ContinuousBatchingOrchestrator(
        config=legacy_config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        device=torch.device("cpu"),
    )
    paged = ContinuousBatchingOrchestrator(
        config=paged_config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        device=torch.device("cpu"),
    )

    scheduler = legacy._build_scheduler_state(active_count=2)

    assert legacy._estimate_sequence_kv_cost(17, scheduler) == 17 * int(scheduler.kv_bytes_per_token or 1)
    assert paged._estimate_sequence_kv_cost(17, scheduler) == 2 * int(scheduler.kv_bytes_per_block or 1)


def test_continuous_scheduler_orders_active_prompts_by_length() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=1,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=Layout(),
        device=torch.device("cpu"),
    )

    ordered_indices, ordered_prompts = orchestrator._order_active_prompts_by_length(
        active_indices=[4, 1, 3, 2],
        prompts=["xxxx", "y", "zzz", "qq"],
    )

    assert ordered_indices == [1, 2, 3, 4]
    assert ordered_prompts == ["y", "qq", "zzz", "xxxx"]


def test_continuous_scheduler_reports_length_sort_metadata() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=1,
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

    ordered_indices, ordered_prompts = orchestrator._order_active_prompts_by_length(
        active_indices=[0, 1, 2],
        prompts=["xxxx", "y", "zzz"],
    )

    responses, _padding_ratio = orchestrator._generate_active_batch(ordered_prompts)

    assert ordered_indices == [1, 2, 0]
    assert responses == ["a", "a", "a"]
    assert orchestrator._runtime_stats["scheduler_length_sort_passes"] == 1.0
    assert orchestrator._runtime_stats["scheduler_length_sorted_sequences"] == 3.0


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


def test_continuous_batching_reconstructs_constructor_based_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def explode_on_bridge(*args, **kwargs):
        del args, kwargs
        raise AssertionError("constructor-cache path should not use the generic legacy bridge")

    monkeypatch.setattr(orchestrator, "_cache_to_legacy", explode_on_bridge)
    monkeypatch.setattr(orchestrator, "_cache_from_legacy", explode_on_bridge)

    stacked = orchestrator._stack_past_key_values([first, second])
    split = orchestrator._split_past_key_values(stacked, batch_size=2)

    assert isinstance(stacked, ConstructorCache)
    assert tuple(stacked.to_legacy_cache()[0][0].shape) == (2, 1, 3, 1)
    assert all(isinstance(item, ConstructorCache) for item in split)
    assert torch.equal(split[0].to_legacy_cache()[0][0], first.to_legacy_cache()[0][0])
    assert torch.equal(split[1].to_legacy_cache()[0][0], second.to_legacy_cache()[0][0])


def test_continuous_batching_reconstructs_constructor_cache_requiring_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    model_config = orchestrator.layout.model.config
    first = ConfigRequiredConstructorCache(
        ddp_cache_data=((torch.zeros((1, 1, 3, 1)), torch.zeros((1, 1, 3, 1))),),
        config=model_config,
    )
    second = ConfigRequiredConstructorCache(
        ddp_cache_data=((torch.ones((1, 1, 3, 1)), torch.ones((1, 1, 3, 1))),),
        config=model_config,
    )

    def explode_on_bridge(*args, **kwargs):
        del args, kwargs
        raise AssertionError("constructor-cache path should not use the generic legacy bridge")

    monkeypatch.setattr(orchestrator, "_cache_to_legacy", explode_on_bridge)
    monkeypatch.setattr(orchestrator, "_cache_from_legacy", explode_on_bridge)

    stacked = orchestrator._stack_past_key_values([first, second])
    split = orchestrator._split_past_key_values(stacked, batch_size=2)

    assert isinstance(stacked, ConfigRequiredConstructorCache)
    assert tuple(stacked.to_legacy_cache()[0][0].shape) == (2, 1, 3, 1)
    assert all(isinstance(item, ConfigRequiredConstructorCache) for item in split)
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


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_continuous_collects_with_dynamic_cache_model() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=4,
        do_sample=False,
        use_paged_kv_continuous=True,
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
    assert batch.metadata["paged_kv_block_size_tokens"] == 16.0
    assert batch.metadata["paged_kv_max_blocks_in_use"] >= 1.0


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_continuous_decode_uses_resident_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=2,
        use_paged_kv_continuous=True,
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

    def explode_on_decode_read(self, sequence_ids: list[int]):
        raise AssertionError(
            "decode path should not rebuild from legacy cache: "
            f"read_batched_legacy_cache({sequence_ids})"
        )

    def explode_on_decode_write(self, sequence_ids: list[int], legacy_cache, cache_template):
        del legacy_cache, cache_template
        raise AssertionError(
            "decode path should not mirror to legacy cache: "
            f"write_batched_legacy_cache({sequence_ids})"
        )

    original_to_legacy = orchestrator._cache_to_legacy
    original_from_legacy = orchestrator._cache_from_legacy
    conversion_counts = {"to_legacy": 0, "from_legacy": 0}

    def guarded_cache_to_legacy(cache):
        conversion_counts["to_legacy"] += 1
        if conversion_counts["to_legacy"] > 2:
            raise AssertionError("decode path should not convert DynamicCache to legacy tuples")
        return original_to_legacy(cache)

    def guarded_cache_from_legacy(cache_like, legacy_cache):
        del legacy_cache
        conversion_counts["from_legacy"] += 1
        raise AssertionError("decode path should not rebuild DynamicCache from legacy tuples")

    monkeypatch.setattr(PagedKVCacheStore, "read_batched_legacy_cache", explode_on_decode_read)
    monkeypatch.setattr(PagedKVCacheStore, "write_batched_legacy_cache", explode_on_decode_write)
    monkeypatch.setattr(orchestrator, "_cache_to_legacy", guarded_cache_to_legacy)
    monkeypatch.setattr(orchestrator, "_cache_from_legacy", guarded_cache_from_legacy)

    batch = orchestrator.collect()

    assert batch.metadata["responses"] == [["ab", "ab"]]
    assert conversion_counts == {"to_legacy": 2, "from_legacy": 0}


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_continuous_decode_keeps_legacy_materialization_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=2,
        use_paged_kv_continuous=True,
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
    scheduler = orchestrator._build_scheduler_state(active_count=2)
    prompts = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([1, 2], dtype=torch.long)]
    masks = [torch.ones_like(prompt) for prompt in prompts]
    sequences = [
        _ScheduledSequence(original_index=index, prompt_ids=prompt, prompt_mask=mask)
        for index, (prompt, mask) in enumerate(zip(prompts, masks, strict=True))
    ]

    captured_store: list[PagedKVCacheStore] = []
    original_build = orchestrator._build_paged_kv_allocator

    def capture_store(*args, **kwargs):
        store = original_build(*args, **kwargs)
        monkeypatch.setattr(store, "release", lambda sequence_id: None)
        captured_store.append(store)
        return store

    def explode_on_decode_write(self, sequence_ids: list[int], legacy_cache, cache_template):
        del legacy_cache, cache_template
        raise AssertionError(
            "decode path should not mirror to legacy cache: "
            f"write_batched_legacy_cache({sequence_ids})"
        )

    monkeypatch.setattr(orchestrator, "_build_paged_kv_allocator", capture_store)
    monkeypatch.setattr(PagedKVCacheStore, "write_batched_legacy_cache", explode_on_decode_write)

    responses, _padding_ratio = orchestrator._generate_active_batch_with_cache(sequences, scheduler)

    store = captured_store[0]
    resident_cache = store.resident_cache(0).to_legacy_cache()
    store.clear_resident_cache(0)
    legacy_cache = store.read_sequence_legacy_cache(0)

    assert responses == ["ab", "ab"]
    assert store.has_resident_cache(0) is False
    assert tuple(legacy_cache[0][0].shape) == (1, 1, 4, 1)
    assert torch.equal(legacy_cache[0][0], resident_cache[0][0])
    assert torch.equal(legacy_cache[0][1], resident_cache[0][1])


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_continuous_dynamic_cache_handles_block_growth() -> None:
    config = GRPOConfig(
        model_name="fake/model",
        batch_size=1,
        group_size=2,
        max_new_tokens=2,
        do_sample=False,
        use_paged_kv_continuous=True,
    )
    orchestrator = ContinuousBatchingOrchestrator(
        config=config,
        environment=SingleTurnEnvironment(label="x" * 16),
        verifier=PrefixVerifier(),
        tokenizer=CharTokenizer(),
        layout=DynamicCacheLayout(),
        device=torch.device("cpu"),
    )

    batch = orchestrator.collect()

    assert batch.rewards.tolist() == [[1.0, 1.0]]
    assert batch.metadata["responses"] == [["ab", "ab"]]
    assert batch.metadata["paged_kv_max_blocks_in_use"] >= 2.0
