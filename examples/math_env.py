"""Reference math environment and verifier for AgentRL demos."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from agentrl import BaseEnvironment, BaseVerifier


_FINAL_ANSWER_RE = re.compile(r"final answer:\s*(-?\d+)", re.IGNORECASE)
_STRICT_FINAL_ANSWER_RE = re.compile(r"^\s*Final answer:\s*(-?\d+)\s*$", re.IGNORECASE)
_INTEGER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True, slots=True)
class MathProblem:
    """One arithmetic training example."""

    prompt: str
    answer: int


class MathEnvironment(BaseEnvironment):
    """Single-turn arithmetic environment for smoke tests and demos.

    The environment samples from a small built-in dataset by default so the demo
    runs without external dependencies.
    """

    def __init__(
        self,
        split: str = "train",
        problems: list[MathProblem] | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self._rng = random.Random(seed)
        self._problems = problems or self._default_problems(split)
        self._current_problem: MathProblem | None = None

    def reset(self) -> str:
        """Return the next arithmetic prompt."""

        self._current_problem = self._rng.choice(self._problems)
        if self.split == "smoke":
            return (
                "Solve the arithmetic problem.\n"
                "Reply with exactly one line and nothing else:\n"
                "Final answer: <integer>\n\n"
                f"Problem: {self._current_problem.prompt}"
            )
        return (
            "Solve the arithmetic problem and reply with exactly one line:\n"
            "Final answer: <integer>\n\n"
            f"Problem: {self._current_problem.prompt}"
        )

    def step(self, action: str) -> tuple[str, bool]:
        """Mark the single-turn episode as complete."""

        del action
        return ("done", True)

    def state(self) -> dict[str, int | str]:
        """Expose the ground-truth answer to the verifier."""

        if self._current_problem is None:
            raise RuntimeError("reset() must be called before state().")
        return {
            "prompt": self._current_problem.prompt,
            "answer": self._current_problem.answer,
            "split": self.split,
        }

    def _default_problems(self, split: str) -> list[MathProblem]:
        smoke = [
            MathProblem("1 + 1", 2),
            MathProblem("2 + 1", 3),
            MathProblem("2 + 2", 4),
            MathProblem("3 + 1", 4),
            MathProblem("3 + 2", 5),
            MathProblem("4 + 1", 5),
            MathProblem("4 + 2", 6),
            MathProblem("5 + 1", 6),
        ]
        train = [
            MathProblem("7 + 5", 12),
            MathProblem("18 - 9", 9),
            MathProblem("6 * 4", 24),
            MathProblem("27 / 3", 9),
            MathProblem("3 * 8 + 2", 26),
            MathProblem("14 + 11 - 3", 22),
        ]
        eval_set = [
            MathProblem("9 + 9", 18),
            MathProblem("15 - 7", 8),
            MathProblem("5 * 5", 25),
        ]
        if split == "smoke":
            return smoke
        return train if split == "train" else eval_set


class MathVerifier(BaseVerifier):
    """Deterministic verifier that checks integer final answers."""

    def verify(self, response: str, env_state: dict[str, int | str]) -> float:
        """Return `1.0` on exact integer match and `0.0` otherwise."""

        answer = int(env_state["answer"])
        split = str(env_state.get("split", "train"))
        extracted = self._extract_answer(response, strict=(split == "smoke"))
        return 1.0 if extracted == answer else 0.0

    def _extract_answer(self, response: str, strict: bool = False) -> int | None:
        if strict:
            strict_match = _STRICT_FINAL_ANSWER_RE.fullmatch(response)
            if strict_match is None:
                return None
            return int(strict_match.group(1))

        match = _FINAL_ANSWER_RE.search(response)
        if match is not None:
            return int(match.group(1))

        integers = _INTEGER_RE.findall(response)
        if not integers:
            return None
        return int(integers[-1])
