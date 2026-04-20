"""Paged-KV allocator primitives for single-GPU rollout engines."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import torch

@dataclass(frozen=True, slots=True)
class PagedKVSequenceView:
    """Read-only sequence mapping inside the paged-KV allocator."""

    sequence_id: int
    token_count: int
    block_size_tokens: int
    physical_blocks: tuple[int, ...]

    @property
    def logical_block_count(self) -> int:
        return len(self.physical_blocks)


class PagedKVAllocator:
    """Deterministic fixed-block allocator for paged-KV experiments.

    The allocator is intentionally simple: it manages a fixed pool of equally
    sized KV blocks and a per-sequence block table. It does not own tensor
    storage yet; its job is to define allocation semantics, reuse, and metrics
    for the runtime-engine branch.
    """

    def __init__(self, total_blocks: int, block_size_tokens: int) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be > 0.")
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be > 0.")

        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self._free_blocks = list(range(total_blocks))
        heapq.heapify(self._free_blocks)
        self._block_tables: dict[int, list[int]] = {}
        self._token_counts: dict[int, int] = {}
        self._touched_blocks: set[int] = set()
        self._block_reuse_count = 0
        self._max_blocks_in_use = 0

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def used_block_count(self) -> int:
        return self.total_blocks - self.free_block_count

    @property
    def block_reuse_count(self) -> int:
        return self._block_reuse_count

    @property
    def max_blocks_in_use(self) -> int:
        return self._max_blocks_in_use

    @property
    def resident_sequence_count(self) -> int:
        return len(self._block_tables)

    @property
    def allocator_occupancy(self) -> float:
        return self.used_block_count / self.total_blocks

    @property
    def allocator_pressure(self) -> float:
        return self.allocator_occupancy

    def reserve(self, sequence_id: int, token_count: int) -> PagedKVSequenceView:
        """Reserve enough blocks for a fresh sequence."""

        if sequence_id in self._block_tables:
            raise ValueError(f"sequence_id {sequence_id} is already allocated.")
        if token_count < 0:
            raise ValueError("token_count must be >= 0.")

        required_blocks = self._required_blocks(token_count)
        blocks = [self._allocate_block() for _ in range(required_blocks)]
        self._block_tables[sequence_id] = blocks
        self._token_counts[sequence_id] = token_count
        self._update_high_watermark()
        return self.view(sequence_id)

    def append_tokens(self, sequence_id: int, token_count: int) -> PagedKVSequenceView:
        """Grow a sequence by the provided number of tokens."""

        if token_count < 0:
            raise ValueError("token_count must be >= 0.")
        if sequence_id not in self._block_tables:
            raise KeyError(f"Unknown sequence_id {sequence_id}.")
        if token_count == 0:
            return self.view(sequence_id)

        new_token_total = self._token_counts[sequence_id] + token_count
        required_blocks = self._required_blocks(new_token_total)
        current_blocks = len(self._block_tables[sequence_id])
        for _ in range(required_blocks - current_blocks):
            self._block_tables[sequence_id].append(self._allocate_block())
        self._token_counts[sequence_id] = new_token_total
        self._update_high_watermark()
        return self.view(sequence_id)

    def release(self, sequence_id: int) -> None:
        """Release all blocks owned by one sequence."""

        blocks = self._block_tables.pop(sequence_id, None)
        self._token_counts.pop(sequence_id, None)
        if blocks is None:
            raise KeyError(f"Unknown sequence_id {sequence_id}.")
        for block_id in blocks:
            heapq.heappush(self._free_blocks, block_id)

    def logical_to_physical(self, sequence_id: int, logical_block_index: int) -> int:
        """Resolve one logical block index to its physical block id."""

        if logical_block_index < 0:
            raise ValueError("logical_block_index must be >= 0.")
        return self._block_tables[sequence_id][logical_block_index]

    def view(self, sequence_id: int) -> PagedKVSequenceView:
        """Return a read-only snapshot for one sequence."""

        if sequence_id not in self._block_tables:
            raise KeyError(f"Unknown sequence_id {sequence_id}.")
        return PagedKVSequenceView(
            sequence_id=sequence_id,
            token_count=self._token_counts[sequence_id],
            block_size_tokens=self.block_size_tokens,
            physical_blocks=tuple(self._block_tables[sequence_id]),
        )

    def metrics(self) -> dict[str, float]:
        """Return allocator metrics suitable for runtime reporting."""

        return {
            "paged_kv_free_block_count": float(self.free_block_count),
            "paged_kv_used_block_count": float(self.used_block_count),
            "paged_kv_allocator_occupancy": float(self.allocator_occupancy),
            "paged_kv_block_reuse_count": float(self.block_reuse_count),
            "paged_kv_allocator_pressure": float(self.allocator_pressure),
            "paged_kv_max_blocks_in_use": float(self.max_blocks_in_use),
            "paged_kv_resident_sequences": float(self.resident_sequence_count),
        }

    def has_sequence(self, sequence_id: int) -> bool:
        """Return whether the allocator still tracks the sequence."""

        return sequence_id in self._block_tables

    def _allocate_block(self) -> int:
        if not self._free_blocks:
            raise RuntimeError("PagedKVAllocator ran out of free blocks.")
        block_id = heapq.heappop(self._free_blocks)
        if block_id in self._touched_blocks:
            self._block_reuse_count += 1
        else:
            self._touched_blocks.add(block_id)
        return block_id

    def _required_blocks(self, token_count: int) -> int:
        if token_count == 0:
            return 0
        return (token_count + self.block_size_tokens - 1) // self.block_size_tokens

    def _update_high_watermark(self) -> None:
        self._max_blocks_in_use = max(self._max_blocks_in_use, self.used_block_count)


class PagedKVCacheStore:
    """Block-backed store for legacy KV tensors.

    The store owns physical block allocation and keeps KV tensors keyed by
    physical block id. Sequence-level caches are reconstructed only when the
    runtime needs to materialize a decode batch.
    """

    def __init__(self, allocator: PagedKVAllocator) -> None:
        self.allocator = allocator
        self._storage: dict[tuple[int, int, int], torch.Tensor] = {}
        self._cache_templates: dict[int, Any] = {}

    @property
    def resident_sequence_count(self) -> int:
        return self.allocator.resident_sequence_count

    def reserve(self, sequence_id: int, token_count: int) -> PagedKVSequenceView:
        return self.allocator.reserve(sequence_id, token_count)

    def append_tokens(self, sequence_id: int, token_count: int) -> PagedKVSequenceView:
        return self.allocator.append_tokens(sequence_id, token_count)

    def has_sequence(self, sequence_id: int) -> bool:
        return self.allocator.has_sequence(sequence_id)

    def release(self, sequence_id: int) -> None:
        view = self.allocator.view(sequence_id)
        for block_id in view.physical_blocks:
            keys_to_delete = [key for key in self._storage if key[0] == block_id]
            for key in keys_to_delete:
                del self._storage[key]
        self._cache_templates.pop(sequence_id, None)
        self.allocator.release(sequence_id)

    def metrics(self) -> dict[str, float]:
        return self.allocator.metrics()

    def cache_template(self, sequence_id: int) -> Any:
        return self._cache_templates[sequence_id]

    def write_sequence_cache(
        self,
        sequence_id: int,
        legacy_cache: tuple[tuple[torch.Tensor, ...], ...],
        cache_template: Any,
    ) -> None:
        view = self.allocator.view(sequence_id)
        token_count = view.token_count
        block_size = view.block_size_tokens
        self._cache_templates[sequence_id] = cache_template

        if token_count == 0:
            return

        for layer_index, layer_cache in enumerate(legacy_cache):
            for state_index, tensor in enumerate(layer_cache):
                trimmed = tensor[:, :, :token_count, ...]
                chunks = list(trimmed.split(block_size, dim=2))
                if len(chunks) != view.logical_block_count:
                    raise ValueError(
                        "Cache chunk count does not match allocated block count "
                        f"for sequence {sequence_id}: {len(chunks)} vs {view.logical_block_count}."
                    )
                for block_id, chunk in zip(view.physical_blocks, chunks, strict=True):
                    self._storage[(block_id, layer_index, state_index)] = chunk.clone()

    def write_batched_legacy_cache(
        self,
        sequence_ids: list[int],
        legacy_cache: tuple[tuple[torch.Tensor, ...], ...],
        cache_template: Any,
    ) -> None:
        split_caches = self._split_legacy_cache(legacy_cache)
        if len(split_caches) != len(sequence_ids):
            raise ValueError("Batched legacy cache does not match the number of sequence ids.")
        for sequence_id, sequence_cache in zip(sequence_ids, split_caches, strict=True):
            self.write_sequence_cache(sequence_id, sequence_cache, cache_template)

    def read_sequence_legacy_cache(self, sequence_id: int) -> tuple[tuple[torch.Tensor, ...], ...]:
        view = self.allocator.view(sequence_id)
        token_count = view.token_count
        layers: list[tuple[torch.Tensor, ...]] = []

        keys = [key for key in self._storage if key[0] in view.physical_blocks]
        if not keys:
            template = self._cache_templates[sequence_id]
            legacy_template = template if isinstance(template, tuple) else None
            if legacy_template is None and hasattr(template, "to_legacy_cache"):
                legacy_template = template.to_legacy_cache()
            if legacy_template is None:
                raise TypeError(f"Unsupported cache template for sequence {sequence_id}: {type(template)!r}")
            return tuple(
                tuple(tensor[:, :, :token_count, ...].clone() for tensor in layer_template)
                for layer_template in legacy_template
            )

        layer_indices = sorted({layer_index for _block_id, layer_index, _state_index in keys})
        for layer_index in layer_indices:
            state_indices = sorted(
                {
                    state_index
                    for _block_id, key_layer_index, state_index in keys
                    if key_layer_index == layer_index
                }
            )
            states: list[torch.Tensor] = []
            for state_index in state_indices:
                chunks = [
                    self._storage[(block_id, layer_index, state_index)]
                    for block_id in view.physical_blocks
                ]
                reconstructed = torch.cat(chunks, dim=2)[:, :, :token_count, ...]
                states.append(reconstructed)
            layers.append(tuple(states))
        return tuple(layers)

    def read_batched_legacy_cache(self, sequence_ids: list[int]) -> tuple[tuple[torch.Tensor, ...], ...]:
        sequence_caches = [self.read_sequence_legacy_cache(sequence_id) for sequence_id in sequence_ids]
        return self._stack_legacy_cache(sequence_caches)

    @staticmethod
    def _stack_legacy_cache(
        caches: list[tuple[tuple[torch.Tensor, ...], ...]],
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        stacked_layers = []
        for layer_caches in zip(*caches, strict=True):
            stacked_layers.append(
                tuple(
                    torch.cat(items, dim=0)
                    for items in zip(*layer_caches, strict=True)
                )
            )
        return tuple(stacked_layers)

    @staticmethod
    def _split_legacy_cache(
        cache: tuple[tuple[torch.Tensor, ...], ...],
    ) -> list[tuple[tuple[torch.Tensor, ...], ...]]:
        batch_size = cache[0][0].shape[0]
        return [
            tuple(
                tuple(state[batch_index : batch_index + 1] for state in layer_cache)
                for layer_cache in cache
            )
            for batch_index in range(batch_size)
        ]
