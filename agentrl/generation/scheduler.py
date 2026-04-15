"""Memory-aware generation scheduling helpers."""

from __future__ import annotations

from typing import Any

import torch


def dtype_bytes(dtype: str) -> int:
    """Return bytes per scalar for supported runtime dtypes."""

    return {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype, 2)


def kv_cache_geometry(model_config: Any) -> tuple[int, int, int]:
    """Return `(layers, heads, head_dim)` needed for KV-cache estimates."""

    num_layers = int(_require_attr(model_config, "num_hidden_layers"))
    num_heads = int(
        getattr(model_config, "num_key_value_heads", None)
        or _require_attr(model_config, "num_attention_heads")
    )
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        hidden_size = int(_require_attr(model_config, "hidden_size"))
        attention_heads = int(_require_attr(model_config, "num_attention_heads"))
        head_dim = hidden_size // attention_heads
    return num_layers, num_heads, int(head_dim)


def estimate_kv_cache_token_bytes(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate KV bytes consumed by one token for one active sequence."""

    return num_layers * num_heads * head_dim * 2 * dtype_bytes


def estimate_kv_cache_sequence_bytes(
    sequence_tokens: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate KV bytes for one sequence with a given cached token length."""

    return max(0, int(sequence_tokens)) * estimate_kv_cache_token_bytes(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )


def estimate_kv_cache_bytes(
    batch_size: int,
    group_size: int,
    max_new_tokens: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate bytes required for an autoregressive KV cache.

    Args:
        batch_size: Number of prompts in the rollout batch.
        group_size: Number of sampled responses per prompt.
        max_new_tokens: Maximum generated length per response.
        num_layers: Transformer layer count.
        num_heads: Attention head count used for cached K/V tensors.
        head_dim: Per-head hidden dimension.
        dtype_bytes: Bytes per scalar value, `2` for fp16 by default.

    Returns:
        Estimated KV cache size in bytes.
    """

    return (
        batch_size
        * group_size
        * estimate_kv_cache_sequence_bytes(
            sequence_tokens=max_new_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
        )
    )


def available_vram_bytes(safety_factor: float = 0.85) -> int:
    """Return conservatively usable VRAM on the current CUDA device.

    Args:
        safety_factor: Fraction of currently free VRAM considered usable.

    Returns:
        Estimated safe usable VRAM in bytes. Returns `0` when CUDA is
        unavailable so callers can degrade gracefully in CPU-only tests.
    """

    if not 0.0 < safety_factor <= 1.0:
        raise ValueError("safety_factor must satisfy 0.0 < safety_factor <= 1.0.")
    if not torch.cuda.is_available():
        return 0

    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    free = max(total - allocated, 0)
    return int(free * safety_factor)


def compute_safe_chunk_size(config: Any, model_config: Any) -> int:
    """Compute the largest response-group chunk that fits available VRAM.

    The chunk size is computed over the per-prompt `group_size` dimension while
    keeping `batch_size` fixed. This matches the intended rollout behavior where
    a large response group is split into smaller generation sub-batches.

    Args:
        config: Runtime config exposing `batch_size`, `group_size`, and
            `max_new_tokens`.
        model_config: Model config exposing `num_hidden_layers`,
            `num_attention_heads`, and either `head_dim` or `hidden_size`.

    Returns:
        The largest chunk size in `[1, group_size]` that fits the current VRAM
        budget. Returns `1` when no larger chunk fits.
    """

    batch_size = int(config.batch_size)
    group_size = int(config.group_size)
    max_new_tokens = int(config.max_new_tokens)

    num_layers, num_heads, head_dim = kv_cache_geometry(model_config)

    budget = available_vram_bytes()
    if budget <= 0:
        return 1

    for chunk_size in range(group_size, 0, -1):
        estimate = estimate_kv_cache_bytes(
            batch_size=batch_size,
            group_size=chunk_size,
            max_new_tokens=max_new_tokens,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            head_dim=int(head_dim),
        )
        if estimate <= budget:
            return chunk_size

    return 1


def _require_attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None:
        raise AttributeError(f"model_config must define `{name}`.")
    return value
