from __future__ import annotations

from examples.tool_use_env import ToolUseEnvironment, ToolUseTask, ToolUseVerifier


def test_tool_use_environment_progresses_through_short_horizon_episode() -> None:
    env = ToolUseEnvironment(
        split="smoke",
        tasks=[
            ToolUseTask(
                task_id="test-alpha",
                goal="Use the lookup tool to find alpha, then submit it.",
                optimal_actions=("TOOL: lookup[alpha]",),
                final_answer="4",
            )
        ],
        seed=0,
    )
    verifier = ToolUseVerifier()

    prompt = env.reset()
    observation, done = env.step("TOOL: lookup[alpha]")
    final_observation, final_done = env.step("FINAL: 4")
    state = env.state()

    assert "Allowed actions" in prompt
    assert "Tool result" in observation
    assert done is False
    assert final_observation == "episode complete"
    assert final_done is True
    assert state["success"] is True
    assert state["completed_tool_steps"] >= 1
    assert verifier.verify("FINAL: 4", state) == 1.0


def test_tool_use_environment_tracks_invalid_actions_and_shaped_reward() -> None:
    env = ToolUseEnvironment(
        split="easy",
        seed=1,
    )
    verifier = ToolUseVerifier()

    env.reset()
    observation, done = env.step("TOOL: unknown[alpha]")
    _, final_done = env.step("FINAL: wrong")
    state = env.state()
    reward = verifier.verify("FINAL: wrong", state)

    assert "Invalid action" in observation
    assert done is False
    assert final_done is True
    assert state["invalid_action_count"] == 1
    assert state["success"] is False
    assert 0.0 <= reward < 1.0
