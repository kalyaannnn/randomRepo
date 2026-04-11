"""Real GSM8K subset environment and verifier for AgentRL benchmarks."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from agentrl import BaseEnvironment, BaseVerifier


_STRICT_FINAL_ANSWER_RE = re.compile(r"^\s*Final answer:\s*(-?\d+)\s*$", re.IGNORECASE)
_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")
_INTEGER_RE = re.compile(r"-?\d+")
_EASY_KEYWORDS = (
    "total",
    "left",
    "remain",
    "together",
    "altogether",
    "each",
    "equally",
    "more",
    "less",
    "gave",
    "bought",
    "spent",
)


@dataclass(frozen=True, slots=True)
class GSM8KProblem:
    """One GSM8K example normalized to prompt plus integer answer."""

    question: str
    answer: int
    solution: str


class GSM8KSubsetEnvironment(BaseEnvironment):
    """Single-turn environment backed by a filtered real GSM8K subset.

    The environment loads the HuggingFace GSM8K dataset lazily, normalizes
    integer answers, filters toward shorter examples, and samples from the first
    `subset_size` retained problems. This gives AgentRL a reproducible benchmark
    rung that is more realistic than the synthetic arithmetic demos without
    jumping straight to the entire dataset.
    """

    def __init__(
        self,
        split: str = "train",
        subset_size: int = 128,
        max_question_words: int = 45,
        curriculum: str = "easy",
        dataset_name: str = "gsm8k",
        dataset_config_name: str = "main",
        seed: int = 0,
        problems: list[GSM8KProblem] | None = None,
    ) -> None:
        if subset_size <= 0:
            raise ValueError("subset_size must be > 0.")
        if max_question_words <= 0:
            raise ValueError("max_question_words must be > 0.")
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")
        if curriculum not in {"easy", "standard"}:
            raise ValueError("curriculum must be either 'easy' or 'standard'.")

        self.split = split
        self.subset_size = subset_size
        self.max_question_words = max_question_words
        self.curriculum = curriculum
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self._rng = random.Random(seed)
        self._problems = problems or self._load_problems()
        self._current_problem: GSM8KProblem | None = None

    def reset(self) -> str:
        """Return the next GSM8K prompt formatted for strict answer checking."""

        self._current_problem = self._rng.choice(self._problems)
        return (
            "Solve the following GSM8K math word problem.\n"
            "Reply with exactly one line and nothing else:\n"
            "Final answer: <integer>\n\n"
            f"Problem: {self._current_problem.question}"
        )

    def step(self, action: str) -> tuple[str, bool]:
        """Mark the single-turn episode as complete."""

        del action
        return ("done", True)

    def state(self) -> dict[str, int | str]:
        """Expose the answer key and original solution for replay/debugging."""

        if self._current_problem is None:
            raise RuntimeError("reset() must be called before state().")
        return {
            "question": self._current_problem.question,
            "answer": self._current_problem.answer,
            "solution": self._current_problem.solution,
            "split": self.split,
            "curriculum": self.curriculum,
            "dataset": self.dataset_name,
            "dataset_config_name": self.dataset_config_name,
        }

    def _load_problems(self) -> list[GSM8KProblem]:
        dataset = self._load_dataset_split()
        filtered: list[GSM8KProblem] = []
        for row in dataset:
            question = str(row["question"]).strip()
            solution = str(row["answer"]).strip()
            if len(question.split()) > self.max_question_words:
                continue

            answer = self._extract_gsm8k_answer(solution)
            if answer is None:
                continue

            filtered.append(
                GSM8KProblem(
                    question=question,
                    answer=answer,
                    solution=solution,
                )
            )

        if self.curriculum == "easy":
            filtered.sort(key=self._difficulty_key)

        if len(filtered) < self.subset_size:
            raise RuntimeError(
                "Filtered GSM8K subset is smaller than requested. "
                f"Requested {self.subset_size}, found {len(filtered)} with "
                f"max_question_words={self.max_question_words} and curriculum='{self.curriculum}'."
            )
        return filtered[: self.subset_size]

    def _load_dataset_split(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - covered via monkeypatch-based tests
            raise ImportError(
                "The GSM8K benchmark requires the optional 'datasets' package. "
                "Install it with `pip install datasets` before using "
                "`examples.gsm8k_subset`."
            ) from exc

        dataset = load_dataset(self.dataset_name, self.dataset_config_name, split=self.split)
        return list(dataset)

    @staticmethod
    def _extract_gsm8k_answer(solution: str) -> int | None:
        match = _GSM8K_ANSWER_RE.search(solution)
        if match is None:
            return None
        return int(match.group(1).replace(",", ""))

    @staticmethod
    def _difficulty_key(problem: GSM8KProblem) -> tuple[int, int, int, int, int]:
        """Rank problems so lower tuples represent easier GSM8K examples.

        The heuristic prefers:
        - shorter questions
        - shorter reference solutions
        - fewer integers mentioned in the question
        - smaller absolute final answers
        - questions with common single-step arithmetic keywords
        """

        question_lower = problem.question.lower()
        keyword_bonus = 0 if any(token in question_lower for token in _EASY_KEYWORDS) else 1
        question_words = len(problem.question.split())
        solution_words = len(problem.solution.split())
        integer_count = len(_INTEGER_RE.findall(problem.question))
        abs_answer = abs(problem.answer)
        return (
            keyword_bonus,
            integer_count,
            question_words,
            solution_words,
            abs_answer,
        )


class GSM8KSubsetVerifier(BaseVerifier):
    """Strict exact-match verifier for GSM8K subset runs."""

    def verify(self, response: str, env_state: dict[str, int | str]) -> float:
        """Return 1.0 for an exact `Final answer: <integer>` match."""

        answer = int(env_state["answer"])
        match = _STRICT_FINAL_ANSWER_RE.fullmatch(response)
        if match is None:
            return 0.0
        return 1.0 if int(match.group(1)) == answer else 0.0
