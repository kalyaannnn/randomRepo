from __future__ import annotations

import pytest

import torch

from agentrl.generation.paged_kv import PagedKVAllocator, PagedKVCacheStore


def test_paged_kv_allocator_reserves_and_releases_blocks() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)

    first = allocator.reserve(sequence_id=7, token_count=3)

    assert first.physical_blocks == (0, 1)
    assert allocator.used_block_count == 2
    assert allocator.free_block_count == 2

    allocator.release(7)

    assert allocator.used_block_count == 0
    assert allocator.free_block_count == 4


def test_paged_kv_allocator_reuses_freed_blocks_deterministically() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)

    allocator.reserve(sequence_id=1, token_count=2)
    allocator.reserve(sequence_id=2, token_count=2)
    allocator.release(1)

    reused = allocator.reserve(sequence_id=3, token_count=2)

    assert reused.physical_blocks == (0,)
    assert allocator.block_reuse_count == 1


def test_paged_kv_allocator_grows_across_block_boundaries() -> None:
    allocator = PagedKVAllocator(total_blocks=5, block_size_tokens=2)

    allocator.reserve(sequence_id=4, token_count=2)
    grown = allocator.append_tokens(sequence_id=4, token_count=3)

    assert grown.token_count == 5
    assert grown.physical_blocks == (0, 1, 2)
    assert allocator.logical_to_physical(4, 2) == 2


def test_paged_kv_allocator_tracks_high_watermark_and_pressure() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)

    allocator.reserve(sequence_id=1, token_count=4)
    allocator.reserve(sequence_id=2, token_count=1)
    allocator.release(2)

    metrics = allocator.metrics()

    assert metrics["paged_kv_used_block_count"] == 2.0
    assert metrics["paged_kv_free_block_count"] == 2.0
    assert metrics["paged_kv_allocator_occupancy"] == 0.5
    assert metrics["paged_kv_max_blocks_in_use"] == 3.0


def test_paged_kv_allocator_raises_when_out_of_blocks() -> None:
    allocator = PagedKVAllocator(total_blocks=1, block_size_tokens=2)
    allocator.reserve(sequence_id=1, token_count=2)

    with pytest.raises(RuntimeError, match="out of free blocks"):
        allocator.append_tokens(sequence_id=1, token_count=1)


def test_paged_kv_cache_store_roundtrips_legacy_cache() -> None:
    allocator = PagedKVAllocator(total_blocks=8, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=3)

    legacy = ((
        torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1),
        torch.arange(10, 13, dtype=torch.float32).view(1, 1, 3, 1),
    ),)
    store.write_sequence_cache(sequence_id=1, legacy_cache=legacy, cache_template=legacy)

    restored = store.read_sequence_legacy_cache(1)

    assert torch.equal(restored[0][0], legacy[0][0])
    assert torch.equal(restored[0][1], legacy[0][1])


def test_paged_kv_cache_store_roundtrips_batched_legacy_cache() -> None:
    allocator = PagedKVAllocator(total_blocks=8, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=3)
    store.reserve(sequence_id=2, token_count=3)

    batched = ((
        torch.tensor([[[[1.0], [2.0], [3.0]]], [[[4.0], [5.0], [6.0]]]]),
        torch.tensor([[[[7.0], [8.0], [9.0]]], [[[10.0], [11.0], [12.0]]]]),
    ),)
    store.write_batched_legacy_cache(
        [1, 2],
        legacy_cache=batched,
        cache_template=batched,
    )

    restored = store.read_batched_legacy_cache([1, 2])

    assert torch.equal(restored[0][0], batched[0][0])
    assert torch.equal(restored[0][1], batched[0][1])
