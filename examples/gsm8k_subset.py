"""Real GSM8K subset environment and verifier for AgentRL benchmarks."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from agentrl import BaseEnvironment, BaseVerifier


_FINAL_LINE_RE = re.compile(r"^Final answer:\s*(-?\d+)\s*$", re.IGNORECASE)
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

    SYSTEM_PROMPT = (
        "You are solving grade-school math word problems.\n"
        "You may reason step by step.\n"
        "The last non-empty line of your response must be exactly:\n"
        "Final answer: <integer>"
    )
    STOP_STRINGS = (
        "\nObservation:",
        "\nUser:",
        "\nHuman:",
        "\nProblem:",
    )

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
        """Return the next GSM8K question."""

        self._current_problem = self._rng.choice(self._problems)
        return self._current_problem.question

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

    def supervised_samples(self, tokenizer: Any | None = None) -> list[tuple[str, str]]:
        """Return rationale-based SFT prompt/target pairs aligned with runtime prompting."""

        return [
            (
                self.render_prompt(tokenizer, problem.question),
                self._build_supervised_target(problem.solution, problem.answer),
            )
            for problem in self._problems
        ]

    def problems(self) -> list[GSM8KProblem]:
        """Return the retained benchmark problems in their filtered order."""

        return list(self._problems)

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

    def render_prompt(self, tokenizer: Any | None, question: str) -> str:
        """Render the GSM8K question through the model's chat template."""

        return self.render_generation_prompt(tokenizer, [question], [])

    def render_generation_prompt(
        self,
        tokenizer: Any | None,
        observations: list[str],
        actions: list[str],
    ) -> str:
        """Render the current conversation with a generation prompt."""

        messages = self._build_messages(observations, actions)
        return self._apply_chat_template(tokenizer, messages, add_generation_prompt=True)

    def render_transcript(
        self,
        tokenizer: Any | None,
        observations: list[str],
        actions: list[str],
    ) -> tuple[str, list[tuple[int, int]]]:
        """Render a full chat transcript and locate assistant spans."""

        messages = self._build_messages(observations, actions)
        transcript_text = self._apply_chat_template(tokenizer, messages, add_generation_prompt=False)
        assistant_spans: list[tuple[int, int]] = []
        search_start = 0
        for action in actions:
            start = transcript_text.find(action, search_start)
            if start == -1:
                start = transcript_text.rfind(action)
            if start == -1:
                raise RuntimeError("Could not align assistant content inside GSM8K transcript.")
            end = start + len(action)
            assistant_spans.append((start, end))
            search_start = end
        return transcript_text, assistant_spans

    def postprocess_response(self, response: str) -> str:
        """Trim known prompt-like continuation strings after generation."""

        truncated = response
        for stop_string in self.STOP_STRINGS:
            stop_index = truncated.find(stop_string)
            if stop_index != -1:
                truncated = truncated[:stop_index]
        return truncated.rstrip()

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

    @classmethod
    def _build_supervised_target(cls, solution: str, answer: int) -> str:
        """Convert GSM8K worked solutions into rationale-plus-final-answer targets."""

        rationale = re.sub(r"\s*####\s*-?\d[\d,]*\s*$", "", solution).strip()
        if rationale:
            return f"{rationale}\n\nFinal answer: {answer}"
        return f"Final answer: {answer}"

    @classmethod
    def _build_messages(
        cls,
        observations: list[str],
        actions: list[str],
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": cls.SYSTEM_PROMPT}]
        for index, observation in enumerate(observations):
            messages.append({"role": "user", "content": observation})
            if index < len(actions):
                messages.append({"role": "assistant", "content": actions[index]})
        return messages

    @classmethod
    def _apply_chat_template(
        cls,
        tokenizer: Any | None,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        parts = [f"System:\n{cls.SYSTEM_PROMPT}\n\n"]
        for message in messages[1:]:
            role = message["role"].capitalize()
            parts.append(f"{role}:\n{message['content']}\n\n")
        if add_generation_prompt:
            parts.append("Assistant:\n")
        return "".join(parts)

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
    """Verifier for GSM8K subset runs.

    Only an exact terminal `Final answer: <integer>` line is accepted.
    """

    def __init__(self, reward_mode: str = "strict") -> None:
        if reward_mode not in {"strict", "binary", "shaped"}:
            raise ValueError("reward_mode must be one of: strict, binary, shaped.")
        self.reward_mode = "strict"

    def verify(self, response: str, env_state: dict[str, int | str]) -> float:
        """Return a strict binary reward based on the terminal final-answer line."""

        answer = int(env_state["answer"])
        predicted = self.extract_terminal_final_answer(response)
        if predicted is None:
            return 0.0
        return 1.0 if predicted == answer else 0.0

    @staticmethod
    def extract_terminal_final_answer(response: str) -> int | None:
        """Parse only the last non-empty line as a final answer."""

        if not response:
            return None
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        if not lines:
            return None
        match = _FINAL_LINE_RE.fullmatch(lines[-1])
        if match is None:
            return None
        return int(match.group(1))
    SYSTEM_PROMPT = (
        "You are solving grade-school math word problems.\n"
        "You may reason step by step.\n"
        "The last non-empty line of your response must be exactly:\n"
        "Final answer: <integer>"
    )
    STOP_STRINGS = (
        "\nObservation:",
        "\nUser:",
        "\nHuman:",
        "\nProblem:",
    )
