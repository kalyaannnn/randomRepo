from __future__ import annotations

import pytest

import torch

from agentrl.generation.paged_kv import PagedKVAllocator, PagedKVCacheStore

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover - depends on transformers version
    DynamicCache = None


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


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_cache_store_materializes_from_resident_cache_when_storage_is_stale() -> None:
    allocator = PagedKVAllocator(total_blocks=8, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=3)

    initial_legacy = ((
        torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1),
        torch.arange(10, 13, dtype=torch.float32).view(1, 1, 3, 1),
    ),)
    store.write_sequence_cache(sequence_id=1, legacy_cache=initial_legacy, cache_template=initial_legacy)

    store.append_tokens(sequence_id=1, token_count=1)
    resident_legacy = ((
        torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1),
        torch.arange(10, 14, dtype=torch.float32).view(1, 1, 4, 1),
    ),)
    store.set_resident_cache(
        sequence_id=1,
        cache=DynamicCache.from_legacy_cache(resident_legacy),
        cache_template=DynamicCache.from_legacy_cache(resident_legacy),
    )

    restored = store.read_sequence_legacy_cache(1)

    assert tuple(restored[0][0].shape) == (1, 1, 4, 1)
    assert torch.equal(restored[0][0], resident_legacy[0][0])
    assert torch.equal(restored[0][1], resident_legacy[0][1])

    store.clear_resident_cache(1)
    restored_after_clear = store.read_sequence_legacy_cache(1)

    assert store.has_resident_cache(1) is False
    assert tuple(restored_after_clear[0][0].shape) == (1, 1, 4, 1)
    assert torch.equal(restored_after_clear[0][0], resident_legacy[0][0])
    assert torch.equal(restored_after_clear[0][1], resident_legacy[0][1])


def test_paged_kv_cache_store_tracks_resident_tuple_cache() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=2)
    legacy = ((torch.ones((1, 1, 2, 1)), torch.zeros((1, 1, 2, 1))),)

    store.set_resident_cache(sequence_id=1, cache=legacy, cache_template=legacy)

    assert store.has_resident_cache(1) is True
    assert store.resident_cache(1) == legacy
    assert store.cache_template(1) == legacy


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_cache_store_tracks_resident_dynamic_cache() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=2)
    legacy = ((torch.ones((1, 1, 2, 1)), torch.zeros((1, 1, 2, 1))),)
    cache = DynamicCache.from_legacy_cache(legacy)

    store.set_resident_cache(sequence_id=1, cache=cache, cache_template=cache)

    assert store.has_resident_cache(1) is True
    assert isinstance(store.resident_cache(1), DynamicCache)


def test_paged_kv_cache_store_clears_resident_cache_on_release() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=2)
    legacy = ((torch.ones((1, 1, 2, 1)), torch.zeros((1, 1, 2, 1))),)
    store.set_resident_cache(sequence_id=1, cache=legacy, cache_template=legacy)

    store.release(1)

    assert store.has_sequence(1) is False
    assert store.has_resident_cache(1) is False


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


@pytest.mark.skipif(DynamicCache is None, reason="transformers DynamicCache is unavailable")
def test_paged_kv_cache_store_roundtrips_dynamic_cache_template() -> None:
    allocator = PagedKVAllocator(total_blocks=8, block_size_tokens=2)
    store = PagedKVCacheStore(allocator=allocator)
    store.reserve(sequence_id=1, token_count=3)
    store.reserve(sequence_id=2, token_count=3)

    batched_legacy = ((
        torch.tensor([[[[1.0], [2.0], [3.0]]], [[[4.0], [5.0], [6.0]]]]),
        torch.tensor([[[[7.0], [8.0], [9.0]]], [[[10.0], [11.0], [12.0]]]]),
    ),)
    template = DynamicCache.from_legacy_cache(batched_legacy)

    store.write_batched_legacy_cache(
        [1, 2],
        legacy_cache=batched_legacy,
        cache_template=template,
    )

    restored_legacy = store.read_batched_legacy_cache([1, 2])
    reconstructed = DynamicCache.from_legacy_cache(restored_legacy)

    assert isinstance(reconstructed, DynamicCache)
    assert torch.equal(restored_legacy[0][0], batched_legacy[0][0])
    assert torch.equal(restored_legacy[0][1], batched_legacy[0][1])
