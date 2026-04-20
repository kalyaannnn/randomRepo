"""Deterministic short-horizon tool-use task stub for runtime benchmarks."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from agentrl import BaseEnvironment, BaseVerifier


_TOOL_ACTION_RE = re.compile(r"^\s*TOOL:\s*([a-z_]+)\[(.*)\]\s*$", re.IGNORECASE)
_FINAL_ACTION_RE = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ToolUseTask:
    """One deterministic task instance for the tool-use benchmark."""

    task_id: str
    goal: str
    optimal_actions: tuple[str, ...]
    final_answer: str


class ToolUseEnvironment(BaseEnvironment):
    """Small deterministic multi-turn environment for runtime benchmarking.

    The task is intentionally narrow: the model emits one action per turn using
    a fixed text grammar, the environment returns concise tool results, and the
    verifier scores success from the final episode state. This keeps the task
    task-backed and multi-turn without baking any agent schema into the core
    library.
    """

    LOOKUP_TABLE = {
        "alpha": "4",
        "beta": "9",
        "gamma": "12",
        "capital_france": "Paris",
        "planet_red": "Mars",
        "metal_au": "gold",
        "word_prime": "prime",
    }

    def __init__(
        self,
        split: str = "train",
        tasks: list[ToolUseTask] | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self._rng = random.Random(seed)
        self._tasks = tasks or self._default_tasks(split)
        self._current_task: ToolUseTask | None = None
        self._tool_trace: list[dict[str, str]] = []
        self._invalid_action_count = 0
        self._completed_tool_steps = 0
        self._final_submitted = False
        self._success = False
        self._last_action = ""

    def reset(self) -> str:
        self._current_task = self._rng.choice(self._tasks)
        self._tool_trace = []
        self._invalid_action_count = 0
        self._completed_tool_steps = 0
        self._final_submitted = False
        self._success = False
        self._last_action = ""
        return self._render_initial_observation()

    def step(self, action: str) -> tuple[str, bool]:
        if self._current_task is None:
            raise RuntimeError("reset() must be called before step().")

        action = action.strip()
        self._last_action = action

        final_match = _FINAL_ACTION_RE.fullmatch(action)
        if final_match is not None:
            submitted_answer = final_match.group(1).strip()
            self._final_submitted = True
            self._success = submitted_answer == self._current_task.final_answer
            return ("episode complete", True)

        tool_match = _TOOL_ACTION_RE.fullmatch(action)
        if tool_match is None:
            self._invalid_action_count += 1
            return (
                "Invalid action. Use exactly one action of the form "
                "`TOOL: name[arg]` or `FINAL: answer`.",
                False,
            )

        tool_name = tool_match.group(1).lower()
        raw_argument = tool_match.group(2).strip()
        try:
            result = self._execute_tool(tool_name, raw_argument)
        except ValueError as exc:
            self._invalid_action_count += 1
            return (f"Invalid action. {exc}", False)
        self._tool_trace.append(
            {
                "tool": tool_name,
                "argument": raw_argument,
                "result": result,
            }
        )

        expected_index = min(self._completed_tool_steps, len(self._current_task.optimal_actions) - 1)
        if expected_index >= 0 and action == self._current_task.optimal_actions[expected_index]:
            self._completed_tool_steps += 1

        return (f"Tool result: {result}", False)

    def state(self) -> dict[str, object]:
        if self._current_task is None:
            raise RuntimeError("reset() must be called before state().")
        return {
            "task_id": self._current_task.task_id,
            "goal": self._current_task.goal,
            "split": self.split,
            "optimal_actions": list(self._current_task.optimal_actions),
            "expected_final_answer": self._current_task.final_answer,
            "tool_trace": list(self._tool_trace),
            "invalid_action_count": self._invalid_action_count,
            "completed_tool_steps": self._completed_tool_steps,
            "total_tool_steps": len(self._current_task.optimal_actions),
            "final_submitted": self._final_submitted,
            "success": self._success,
            "last_action": self._last_action,
        }

    def render_generation_prompt(
        self,
        tokenizer: object,
        observations: list[str],
        actions: list[str],
    ) -> str:
        del tokenizer
        parts = [
            "You are a tool-using agent.\n"
            "Return exactly one action per turn.\n"
            "Allowed actions:\n"
            "- TOOL: lookup[key]\n"
            "- TOOL: add[a,b]\n"
            "- TOOL: concat[a,b]\n"
            "- FINAL: answer\n\n",
        ]
        for index, observation in enumerate(observations):
            parts.append(f"Observation:\n{observation}\n\n")
            if index < len(actions):
                parts.append(f"Assistant:\n{actions[index]}\n\n")
        parts.append("Assistant:\n")
        return "".join(parts)

    def render_transcript(
        self,
        tokenizer: object,
        observations: list[str],
        actions: list[str],
    ) -> tuple[str, list[tuple[int, int]]]:
        del tokenizer
        parts: list[str] = []
        assistant_spans: list[tuple[int, int]] = []
        cursor = 0

        for index, observation in enumerate(observations):
            prefix = "Observation:\n"
            parts.extend((prefix, observation, "\n\n"))
            cursor += len(prefix) + len(observation) + 2
            if index < len(actions):
                assistant_prefix = "Assistant:\n"
                parts.append(assistant_prefix)
                cursor += len(assistant_prefix)
                start = cursor
                parts.append(actions[index])
                cursor += len(actions[index])
                assistant_spans.append((start, cursor))
                parts.append("\n\n")
                cursor += 2

        return "".join(parts), assistant_spans

    def _render_initial_observation(self) -> str:
        assert self._current_task is not None
        return (
            f"Goal: {self._current_task.goal}\n"
            "Allowed actions:\n"
            "- TOOL: lookup[key]\n"
            "- TOOL: add[a,b]\n"
            "- TOOL: concat[a,b]\n"
            "- FINAL: answer\n"
            "Return exactly one action."
        )

    def _execute_tool(self, tool_name: str, raw_argument: str) -> str:
        if tool_name == "lookup":
            return self.LOOKUP_TABLE.get(raw_argument, f"unknown:{raw_argument}")
        if tool_name == "add":
            left, right = self._split_two_args(raw_argument)
            return str(int(left.strip()) + int(right.strip()))
        if tool_name == "concat":
            left, right = self._split_two_args(raw_argument)
            return f"{left.strip()}{right.strip()}"
        raise ValueError(f"Unknown tool {tool_name!r}.")

    def _split_two_args(self, raw_argument: str) -> tuple[str, str]:
        if "," not in raw_argument:
            raise ValueError(f"Expected two comma-separated arguments, got: {raw_argument!r}")
        left, right = raw_argument.split(",", 1)
        return left, right

    def _default_tasks(self, split: str) -> list[ToolUseTask]:
        smoke = [
            ToolUseTask(
                task_id="smoke-alpha",
                goal="Use the lookup tool to find the value for alpha, then submit it.",
                optimal_actions=("TOOL: lookup[alpha]",),
                final_answer="4",
            ),
            ToolUseTask(
                task_id="smoke-france",
                goal="Use the lookup tool to find the capital of France, then submit it.",
                optimal_actions=("TOOL: lookup[capital_france]",),
                final_answer="Paris",
            ),
        ]
        easy = [
            ToolUseTask(
                task_id="easy-add",
                goal="Look up alpha, add 3 to it, then submit the result.",
                optimal_actions=("TOOL: lookup[alpha]", "TOOL: add[4,3]"),
                final_answer="7",
            ),
            ToolUseTask(
                task_id="easy-concat",
                goal="Look up word_prime, concatenate it with metal_au, then submit the result.",
                optimal_actions=(
                    "TOOL: lookup[word_prime]",
                    "TOOL: lookup[metal_au]",
                    "TOOL: concat[prime,gold]",
                ),
                final_answer="primegold",
            ),
        ]
        train = [
            ToolUseTask(
                task_id="train-sum-two-lookups",
                goal="Look up alpha and beta, add them, then submit the result.",
                optimal_actions=(
                    "TOOL: lookup[alpha]",
                    "TOOL: lookup[beta]",
                    "TOOL: add[4,9]",
                ),
                final_answer="13",
            ),
            ToolUseTask(
                task_id="train-planet-gold",
                goal="Look up planet_red and metal_au, concatenate them, then submit the result.",
                optimal_actions=(
                    "TOOL: lookup[planet_red]",
                    "TOOL: lookup[metal_au]",
                    "TOOL: concat[Mars,gold]",
                ),
                final_answer="Marsgold",
            ),
        ]
        eval_set = [
            ToolUseTask(
                task_id="eval-gamma",
                goal="Look up gamma, add 5 to it, then submit the result.",
                optimal_actions=("TOOL: lookup[gamma]", "TOOL: add[12,5]"),
                final_answer="17",
            ),
        ]
        if split == "smoke":
            return smoke
        if split == "easy":
            return easy
        return train if split == "train" else eval_set


class ToolUseVerifier(BaseVerifier):
    """Episode-level verifier for the tool-use task stub."""

    def verify(self, response: str, env_state: dict[str, object]) -> float:
        del response
        if bool(env_state["success"]):
            return 1.0

        total_tool_steps = max(1, int(env_state["total_tool_steps"]))
        completed_fraction = min(1.0, float(env_state["completed_tool_steps"]) / total_tool_steps)
        invalid_penalty = 0.2 * float(env_state["invalid_action_count"])
        final_bonus = 0.1 if bool(env_state["final_submitted"]) else 0.0
        reward = (0.45 * completed_fraction) + final_bonus - invalid_penalty
        return max(0.0, min(0.95, reward))
