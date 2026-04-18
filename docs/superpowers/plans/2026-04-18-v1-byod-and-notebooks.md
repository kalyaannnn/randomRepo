# V1 BYOD And Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the v1 BYOD onboarding story by adding a small official high-level API, updating docs, and shipping GSM8K plus canonical BYOD notebooks.

**Architecture:** Add a thin public BYOD adapter layer that composes the existing `BaseEnvironment`, `BaseVerifier`, `SFTBootstrapTrainer`, and `GRPOTrainer` contracts instead of replacing them. Keep the library-owned high-level API narrow, move the current example types behind that API, and teach the workflow through notebooks and docs rather than adding a baked-in benchmark feature.

**Tech Stack:** Python 3.10+, dataclasses, existing AgentRL core trainers/runtime, pytest, Jupyter notebooks stored as `.ipynb`

---

## File Structure

- Create: `agentrl/byod.py`
- Modify: `agentrl/__init__.py`
- Modify: `examples/byod_task.py`
- Create: `tests/test_byod_api.py`
- Modify: `docs/bring_your_own_task.md`
- Modify: `README.md`
- Create: `notebooks/gsm8k_end_to_end.ipynb`
- Create: `notebooks/byod_onboarding.ipynb`

### Task 1: Add The Official BYOD API

**Files:**
- Create: `agentrl/byod.py`
- Modify: `agentrl/__init__.py`
- Test: `tests/test_byod_api.py`

- [ ] **Step 1: Write the failing public-surface tests**

```python
from __future__ import annotations

from agentrl.byod import BYODRecord, make_single_turn_task


def test_make_single_turn_task_builds_environment_and_verifier() -> None:
    records = [
        BYODRecord(
            input="Solve 2 + 2.",
            reference_answer="Final answer: 4",
            supervised_target="2 + 2 = 4\n\nFinal answer: 4",
            metadata={"difficulty": "easy"},
        )
    ]

    task = make_single_turn_task(
        records=records,
        prompt_formatter=lambda record, tokenizer: f"Prompt: {record.input}",
        reward_fn=lambda response, state: 1.0 if response.strip() == state["reference_answer"] else 0.0,
        supervised_target_fn=lambda record: record.supervised_target,
    )

    prompt = task.environment.reset()
    state = task.environment.state()
    reward = task.verifier.verify("Final answer: 4", state)
    samples = task.supervised_samples(tokenizer=None)

    assert prompt == "Prompt: Solve 2 + 2."
    assert state["reference_answer"] == "Final answer: 4"
    assert reward == 1.0
    assert samples == [("Prompt: Solve 2 + 2.", "2 + 2 = 4\n\nFinal answer: 4")]


def test_make_single_turn_task_requires_non_empty_records() -> None:
    try:
        make_single_turn_task(
            records=[],
            prompt_formatter=lambda record, tokenizer: record.input,
            reward_fn=lambda response, state: 0.0,
        )
    except ValueError as exc:
        assert "at least one record" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty records.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_byod_api.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentrl.byod'` or missing symbol errors from the new public API.

- [ ] **Step 3: Write the minimal BYOD adapter implementation**

```python
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from agentrl.core.base import BaseEnvironment, BaseVerifier


RecordT = TypeVar("RecordT")


@dataclass(frozen=True, slots=True)
class BYODRecord:
    input: str
    reference_answer: str
    supervised_target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PromptFormatter = Callable[[RecordT, Any | None], str]
RewardFn = Callable[[str, dict[str, Any]], float]
SupervisedTargetFn = Callable[[RecordT], str | None]


@dataclass(slots=True)
class BYODTask:
    environment: BaseEnvironment
    verifier: BaseVerifier
    _supervised_samples_fn: Callable[[Any | None], list[tuple[str, str]]]

    def supervised_samples(self, tokenizer: Any | None = None) -> list[tuple[str, str]]:
        return self._supervised_samples_fn(tokenizer)


class _SingleTurnEnvironment(BaseEnvironment, Generic[RecordT]):
    def __init__(
        self,
        records: Sequence[RecordT],
        prompt_formatter: PromptFormatter[RecordT],
        state_builder: Callable[[RecordT], dict[str, Any]],
        seed: int = 0,
    ) -> None:
        if not records:
            raise ValueError("BYOD tasks require at least one record.")
        self._records = list(records)
        self._prompt_formatter = prompt_formatter
        self._state_builder = state_builder
        self._rng = random.Random(seed)
        self._current_record: RecordT | None = None

    def reset(self) -> str:
        self._current_record = self._rng.choice(self._records)
        return self._prompt_formatter(self._current_record, None)

    def step(self, action: str) -> tuple[str, bool]:
        del action
        return ("done", True)

    def state(self) -> dict[str, Any]:
        if self._current_record is None:
            raise RuntimeError("reset() must be called before state().")
        return self._state_builder(self._current_record)


class _HookVerifier(BaseVerifier):
    def __init__(self, reward_fn: RewardFn) -> None:
        self._reward_fn = reward_fn

    def verify(self, response: str, env_state: dict[str, Any]) -> float:
        reward = float(self._reward_fn(response, env_state))
        return min(1.0, max(0.0, reward))


def make_single_turn_task(
    *,
    records: Sequence[BYODRecord],
    prompt_formatter: PromptFormatter[BYODRecord],
    reward_fn: RewardFn,
    supervised_target_fn: SupervisedTargetFn[BYODRecord] | None = None,
    seed: int = 0,
) -> BYODTask:
    environment = _SingleTurnEnvironment(
        records=records,
        prompt_formatter=prompt_formatter,
        state_builder=lambda record: {
            "input": record.input,
            "reference_answer": record.reference_answer,
            "metadata": dict(record.metadata),
        },
        seed=seed,
    )
    verifier = _HookVerifier(reward_fn)

    def build_samples(tokenizer: Any | None) -> list[tuple[str, str]]:
        if supervised_target_fn is None:
            raise ValueError("No supervised target hook configured for this task.")
        rows: list[tuple[str, str]] = []
        for record in records:
            target = supervised_target_fn(record)
            if target is None:
                continue
            rows.append((prompt_formatter(record, tokenizer), target))
        if not rows:
            raise ValueError("No supervised targets produced by the configured hook.")
        return rows

    return BYODTask(
        environment=environment,
        verifier=verifier,
        _supervised_samples_fn=build_samples,
    )
```

- [ ] **Step 4: Export the public API**

```python
from agentrl.byod import BYODRecord, BYODTask, make_single_turn_task

__all__ = [
    "AgentRLDebugger",
    "BYODRecord",
    "BYODTask",
    "BaseEnvironment",
    "BaseVerifier",
    "ConfigurationError",
    "GRPOConfig",
    "GRPOTrainer",
    "MetricsLogger",
    "ReplayBuffer",
    "SFTBootstrapTrainer",
    "SystemsProfiler",
    "TrajectoryBuffer",
    "TrajectoryStore",
    "make_single_turn_task",
]
```

- [ ] **Step 5: Run the focused tests and make sure they pass**

Run: `pytest tests/test_byod_api.py -q`
Expected: PASS with both new tests green.

- [ ] **Step 6: Commit**

```bash
git add agentrl/byod.py agentrl/__init__.py tests/test_byod_api.py
git commit -m "feat: add official BYOD task API"
```

### Task 2: Align The Example Module With The Official API

**Files:**
- Modify: `examples/byod_task.py`
- Test: `tests/test_byod_task.py`
- Test: `tests/test_byod_api.py`

- [ ] **Step 1: Write the failing compatibility test**

```python
from __future__ import annotations

from examples.byod_task import BYODEnvironment, ExactMatchVerifier, TaskRecord, build_demo_task


def test_build_demo_task_wraps_example_records_with_official_api() -> None:
    records = [
        TaskRecord(
            prompt="Return exactly: ok",
            expected_answer="ok",
            target="ok",
        )
    ]

    task = build_demo_task(records=records)
    prompt = task.environment.reset()
    reward = task.verifier.verify("ok", task.environment.state())
    samples = task.supervised_samples(tokenizer=None)

    assert "Return exactly: ok" in prompt
    assert reward == 1.0
    assert samples[0][1] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_byod_task.py::test_build_demo_task_wraps_example_records_with_official_api -q`
Expected: FAIL because `build_demo_task` does not exist yet.

- [ ] **Step 3: Refactor the example module to use the official API as the canonical path**

```python
from agentrl.byod import BYODRecord, BYODTask, make_single_turn_task


def _prompt_formatter(record: BYODRecord, tokenizer):
    del tokenizer
    return (
        "System:\nYou are solving a user-defined task.\n"
        "Respond in the format expected by the verifier.\n\n"
        f"User:\n{record.input}\n\nAssistant:\n"
    )


def build_demo_task(records: list[TaskRecord], seed: int = 0) -> BYODTask:
    adapted = [
        BYODRecord(
            input=record.prompt,
            reference_answer=record.expected_answer,
            supervised_target=record.target,
            metadata=dict(record.metadata),
        )
        for record in records
    ]
    return make_single_turn_task(
        records=adapted,
        prompt_formatter=_prompt_formatter,
        reward_fn=lambda response, state: 1.0 if response.strip() == state["reference_answer"] else 0.0,
        supervised_target_fn=lambda record: record.supervised_target,
        seed=seed,
    )
```

- [ ] **Step 4: Preserve the old minimal example classes as thin compatibility wrappers**

```python
class BYODEnvironment(BaseEnvironment):
    def __init__(self, records: list[TaskRecord], seed: int = 0) -> None:
        self._task = build_demo_task(records=records, seed=seed)

    def reset(self) -> str:
        return self._task.environment.reset()

    def step(self, action: str) -> tuple[str, bool]:
        return self._task.environment.step(action)

    def state(self) -> dict[str, Any]:
        return self._task.environment.state()

    def supervised_samples(self, tokenizer: Any | None = None) -> list[tuple[str, str]]:
        return self._task.supervised_samples(tokenizer=tokenizer)


class ExactMatchVerifier(BaseVerifier):
    def verify(self, response: str, env_state: dict[str, Any]) -> float:
        return 1.0 if response.strip() == str(env_state["reference_answer"]).strip() else 0.0
```

- [ ] **Step 5: Run the example-oriented tests and make sure they pass**

Run: `pytest tests/test_byod_task.py tests/test_byod_api.py -q`
Expected: PASS with the example module still working while the new API is canonical.

- [ ] **Step 6: Commit**

```bash
git add examples/byod_task.py tests/test_byod_task.py tests/test_byod_api.py
git commit -m "refactor: align BYOD example with official API"
```

### Task 3: Document The Official V1 BYOD Workflow

**Files:**
- Modify: `docs/bring_your_own_task.md`
- Modify: `README.md`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Add a failing doc-surface assertion**

```python
from pathlib import Path


def test_readme_mentions_official_byod_api() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "make_single_turn_task" in readme
    assert "BYODRecord" in readme
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_examples.py::test_readme_mentions_official_byod_api -q`
Expected: FAIL because the README does not mention the new BYOD API yet.

- [ ] **Step 3: Update `docs/bring_your_own_task.md` to teach the new high-level path first**

```md
## Official High-Level API

For the common single-turn case, use the library-owned BYOD adapter layer:

```python
from agentrl import BYODRecord, make_single_turn_task

records = [
    BYODRecord(
        input="Return exactly: ok",
        reference_answer="ok",
        supervised_target="ok",
    )
]

task = make_single_turn_task(
    records=records,
    prompt_formatter=lambda record, tokenizer: f"User:\n{record.input}\n\nAssistant:\n",
    reward_fn=lambda response, state: 1.0 if response.strip() == state["reference_answer"] else 0.0,
    supervised_target_fn=lambda record: record.supervised_target,
)
```

This path is the recommended v1 onboarding workflow. Drop down to `BaseEnvironment` and `BaseVerifier` directly when you need custom multi-turn state transitions.
```

- [ ] **Step 4: Update `README.md` to present the BYOD API and two-notebook demo story**

```md
## Bring Your Own Task / Dataset

The intended v1 onboarding path is now:

1. bring your dataset
2. define prompt formatting or environment behavior
3. define verifier or reward logic
4. bootstrap with supervised targets when available
5. run verifier-based GRPO
6. evaluate the saved adapter

For the common single-turn case, use `BYODRecord` plus `make_single_turn_task` instead of writing framework base classes directly.
```

- [ ] **Step 5: Run the doc regression tests and make sure they pass**

Run: `pytest tests/test_examples.py::test_readme_mentions_official_byod_api -q`
Expected: PASS and existing example tests remain green if the public docs match the shipped API.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/bring_your_own_task.md tests/test_examples.py
git commit -m "docs: publish official BYOD onboarding workflow"
```

### Task 4: Add The GSM8K Demo Notebook

**Files:**
- Create: `notebooks/gsm8k_end_to_end.ipynb`
- Modify: `README.md`

- [ ] **Step 1: Create the notebook skeleton**

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# GSM8K End-to-End Demo\n",
        "\n",
        "This notebook demonstrates the intended v1 workflow: bootstrap, diagnostic evaluation, GRPO, and final evaluation."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from pathlib import Path\n",
        "from agentrl import GRPOConfig\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- [ ] **Step 2: Populate the notebook with the real GSM8K workflow cells**

```python
from examples.bootstrap_gsm8k_subset import main as bootstrap_main
from examples.eval_gsm8k_subset import main as eval_main

bootstrap_args = [
    "--model", "Qwen/Qwen2.5-1.5B-Instruct",
    "--epochs", "3",
    "--batch-size", "4",
    "--subset-size", "128",
    "--max-question-words", "45",
    "--curriculum", "easy",
    "--adapter-dir", "./bootstrap_gsm8k_adapter_15b",
]

eval_args = [
    "--model", "Qwen/Qwen2.5-1.5B-Instruct",
    "--init-adapter-path", "./bootstrap_gsm8k_adapter_15b",
    "--output-dir", "./eval_gsm8k_subset_15b_diag",
    "--split", "train",
    "--subset-size", "128",
]
```

- [ ] **Step 3: Link the notebook from the README demo section**

```md
- `notebooks/gsm8k_end_to_end.ipynb` walks through the bundled GSM8K benchmark path from bootstrap to post-RL evaluation.
```

- [ ] **Step 4: Sanity-check the notebook JSON**

Run: `python -m json.tool notebooks/gsm8k_end_to_end.ipynb >/dev/null`
Expected: command exits successfully with no output.

- [ ] **Step 5: Commit**

```bash
git add notebooks/gsm8k_end_to_end.ipynb README.md
git commit -m "docs: add GSM8K end-to-end demo notebook"
```

### Task 5: Add The Canonical BYOD Onboarding Notebook

**Files:**
- Create: `notebooks/byod_onboarding.ipynb`
- Modify: `README.md`
- Modify: `docs/bring_your_own_task.md`

- [ ] **Step 1: Create the notebook skeleton**

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# BYOD Onboarding Demo\n",
        "\n",
        "This notebook shows the canonical AgentRL v1 path for bringing your own dataset, formatting hook, verifier hook, bootstrap targets, and training loop."
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- [ ] **Step 2: Fill the notebook with the canonical high-level BYOD flow**

```python
from agentrl import BYODRecord, GRPOConfig, GRPOTrainer, SFTBootstrapTrainer, make_single_turn_task

records = [
    BYODRecord(
        input="Return exactly: ok",
        reference_answer="ok",
        supervised_target="ok",
        metadata={"split": "train"},
    )
]

task = make_single_turn_task(
    records=records,
    prompt_formatter=lambda record, tokenizer: f"User:\n{record.input}\n\nAssistant:\n",
    reward_fn=lambda response, state: 1.0 if response.strip() == state["reference_answer"] else 0.0,
    supervised_target_fn=lambda record: record.supervised_target,
)

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    steps=2,
    batch_size=1,
    group_size=2,
    max_new_tokens=32,
)
```

- [ ] **Step 3: Add explicit notebook guidance about when to drop to low-level contracts**

```md
## When To Use `BaseEnvironment` Directly

Use the high-level BYOD API for single-turn or lightly customized tasks.

Drop down to `BaseEnvironment` and `BaseVerifier` directly when:

- you need true multi-turn state transitions
- your verifier requires richer environment state than the default record adapter
- your task sampling logic is more complex than record-based sampling
```

- [ ] **Step 4: Link the notebook from the README and BYOD guide**

```md
- `notebooks/byod_onboarding.ipynb` is the canonical v1 notebook for custom datasets and reward hooks.
```

- [ ] **Step 5: Sanity-check the notebook JSON**

Run: `python -m json.tool notebooks/byod_onboarding.ipynb >/dev/null`
Expected: command exits successfully with no output.

- [ ] **Step 6: Commit**

```bash
git add notebooks/byod_onboarding.ipynb README.md docs/bring_your_own_task.md
git commit -m "docs: add canonical BYOD onboarding notebook"
```

### Task 6: Final Verification For V1 BYOD Closure

**Files:**
- Modify: any files touched above only if verification exposes mismatches
- Test: `tests/test_byod_api.py`
- Test: `tests/test_byod_task.py`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Run the targeted regression suite**

Run: `pytest tests/test_byod_api.py tests/test_byod_task.py tests/test_examples.py -q`
Expected: PASS with the new BYOD API, example compatibility, and doc assertions all green.

- [ ] **Step 2: Run the broader existing checks that cover rollout-facing regressions**

Run: `pytest tests/test_base.py tests/test_trainer.py tests/test_runtime.py tests/test_continuous.py -q`
Expected: PASS with no regressions in core trainer/runtime behavior from the BYOD adapter additions.

- [ ] **Step 3: Validate notebook files are well-formed**

Run: `python -m json.tool notebooks/gsm8k_end_to_end.ipynb >/dev/null`
Expected: PASS with no output.

Run: `python -m json.tool notebooks/byod_onboarding.ipynb >/dev/null`
Expected: PASS with no output.

- [ ] **Step 4: Review the final user-facing scope**

Run: `rg -n "BYODRecord|make_single_turn_task|byod_onboarding|gsm8k_end_to_end" README.md docs agentrl examples`
Expected: hits in the public package surface, README, BYOD docs, and notebook references.

- [ ] **Step 5: Commit**

```bash
git add agentrl/byod.py agentrl/__init__.py examples/byod_task.py tests/test_byod_api.py tests/test_byod_task.py tests/test_examples.py README.md docs/bring_your_own_task.md notebooks/gsm8k_end_to_end.ipynb notebooks/byod_onboarding.ipynb
git commit -m "feat: finish v1 BYOD onboarding and demo notebooks"
```
