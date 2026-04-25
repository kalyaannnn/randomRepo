from __future__ import annotations

import math

import pytest
import torch

from agentrl.core.trainer import (
    _compute_clipped_grpo_objective,
    _compute_logprob_ratio,
    _compute_sampled_token_kl,
)


def test_compute_logprob_ratio_matches_exponentiated_difference() -> None:
    current_logprobs = torch.tensor([[0.4, -0.2]], dtype=torch.float32)
    old_logprobs = torch.tensor([[0.1, -0.5]], dtype=torch.float32)

    ratio = _compute_logprob_ratio(current_logprobs, old_logprobs, clip_range=100.0)

    expected = torch.exp(torch.tensor([[0.3, 0.3]], dtype=torch.float32))
    assert torch.allclose(ratio, expected)


def test_clipped_grpo_objective_applies_ppo_bounds() -> None:
    current_logprobs = torch.log(torch.tensor([[1.5], [0.5]], dtype=torch.float32))
    old_logprobs = torch.zeros_like(current_logprobs)
    sampled_token_mask = torch.ones_like(current_logprobs, dtype=torch.bool)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

    stats = _compute_clipped_grpo_objective(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        sampled_token_mask=sampled_token_mask,
        epsilon=0.2,
        beta=0.0,
        ref_logprobs=None,
        clip_range=100.0,
    )

    assert stats.policy_loss == pytest.approx(-0.2, abs=1e-6)
    assert stats.total_loss == pytest.approx(-0.2, abs=1e-6)
    assert stats.clip_ratio_region_mean == pytest.approx(1.0, abs=1e-6)
    assert stats.clip_ratio_low_mean == pytest.approx(0.5, abs=1e-6)
    assert stats.clip_ratio_high_mean == pytest.approx(0.5, abs=1e-6)
    assert stats.mean_ratio == pytest.approx(1.0, abs=1e-6)


def test_compute_sampled_token_kl_matches_trl_approximator() -> None:
    current_logprobs = torch.tensor([[-0.2, 0.3]], dtype=torch.float32)
    ref_logprobs = torch.tensor([[-0.1, -0.5]], dtype=torch.float32)

    token_kl = _compute_sampled_token_kl(current_logprobs, ref_logprobs)

    delta = ref_logprobs - current_logprobs
    expected = torch.exp(delta) - delta - 1.0
    assert torch.allclose(token_kl, expected)


def test_clipped_grpo_objective_averages_only_active_completion_tokens() -> None:
    current_logprobs = torch.tensor(
        [[math.log(1.1), math.log(7.0), math.log(0.7)]],
        dtype=torch.float32,
    )
    old_logprobs = torch.zeros_like(current_logprobs)
    ref_logprobs = torch.tensor(
        [[math.log(1.1), math.log(200.0), math.log(0.7)]],
        dtype=torch.float32,
    )
    sampled_token_mask = torch.tensor([[True, False, True]])
    advantages = torch.tensor([2.0], dtype=torch.float32)

    stats = _compute_clipped_grpo_objective(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        sampled_token_mask=sampled_token_mask,
        epsilon=0.2,
        beta=0.5,
        ref_logprobs=ref_logprobs,
        clip_range=100.0,
    )

    assert stats.policy_loss == pytest.approx(-1.8, abs=1e-6)
    assert stats.kl_loss == pytest.approx(0.0, abs=1e-6)
    assert stats.total_loss == pytest.approx(-1.8, abs=1e-6)
    assert stats.mean_ratio == pytest.approx(0.9, abs=1e-6)
    assert stats.mean_token_kl == pytest.approx(0.0, abs=1e-6)
    assert stats.clip_ratio_region_mean == pytest.approx(0.5, abs=1e-6)
    assert stats.clip_ratio_low_mean == pytest.approx(0.5, abs=1e-6)
    assert stats.clip_ratio_high_mean == pytest.approx(0.0, abs=1e-6)
