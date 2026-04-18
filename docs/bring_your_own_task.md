# Bring Your Own Task / Dataset

AgentRL does not need to own your dataset. The recommended v1 onboarding path is:

1. you bring task records from a dataset, JSONL, or in-memory source
2. you build a small task record shape with `BYODRecord`
3. you use `make_single_turn_task(...)` for the common single-turn path
4. you keep reward logic in the verifier hook and bootstrap data in `supervised_target_fn`
5. you drop to `BaseEnvironment` and `BaseVerifier` directly only when you need full custom control

That keeps the framework focused on single-GPU post-training instead of growing into a dataset platform.

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

This is the recommended v1 onboarding workflow. Use it first when the task is a single-turn prompt, a deterministic verifier, and an optional supervised bootstrap target.

Drop down to `BaseEnvironment` and `BaseVerifier` directly when you need custom multi-turn state transitions, nonstandard rollout state, or a task shape that does not fit the single-turn adapter.

## Minimum Task-Side Contract

For RL:

- implement `BaseEnvironment`
- implement `BaseVerifier`

That matches the existing public surface:

- [agentrl/core/base.py](/Users/kalyaanrao/newProject/agentrl/core/base.py:1)
- [agentrl/core/sft.py](/Users/kalyaanrao/newProject/agentrl/core/sft.py:1)

## Bootstrap Adapter Convention

When a task supports supervised warm start, a common adapter convention is to expose:

- `supervised_samples(tokenizer=None) -> list[tuple[str, str]]`

This is not part of the core `BaseEnvironment` / `BaseVerifier` contract. It is the bootstrap-side shape used by `SFTBootstrapTrainer`, which expects prompt/target pairs from the task layer.

## Example Input Styles

The task can load examples from:

- local `jsonl`
- Hugging Face datasets
- in-memory records

Subset filtering, curriculum, and prompt formatting remain task-specific decisions inside the environment.

## Lightweight Example

See [examples/byod_task.py](/Users/kalyaanrao/newProject/examples/byod_task.py:1) for a minimal reference implementation.

It shows:

- `TaskRecord` as a simple raw-example shape
- `BYODEnvironment` loading records from memory or JSONL
- `supervised_samples()` producing bootstrap prompt/target pairs
- `ExactMatchVerifier` defining reward without any extra reward DSL

## JSONL Schema

The example JSONL loader expects one object per line with:

```json
{
  "prompt": "Solve 2 + 2.",
  "expected_answer": "Final answer: 4",
  "target": "Reasoning...\n\nFinal answer: 4",
  "metadata": {
    "difficulty": "easy"
  }
}
```

Required keys:

- `prompt`
- `expected_answer`

Optional keys:

- `target`
- `metadata`

## Minimal Usage

```python
from agentrl import GRPOConfig, GRPOTrainer, SFTBootstrapTrainer
from examples.byod_task import BYODEnvironment, ExactMatchVerifier, TaskRecord

records = [
    TaskRecord(
        prompt="Solve 2 + 2. Reply with exactly one line: Final answer: <integer>",
        expected_answer="Final answer: 4",
        target="2 + 2 = 4\n\nFinal answer: 4",
    )
]

environment = BYODEnvironment(records=records)
verifier = ExactMatchVerifier()
```

For bootstrap, call `environment.supervised_samples(tokenizer=...)` and pass the resulting pairs to `SFTBootstrapTrainer`.

For RL, pass the same environment and verifier to `GRPOTrainer`.

## Recommended Scope for v1

Do not add a generic dataset registry or plugin system yet.

The intended abstraction is:

- users bring records
- task environments interpret them
- verifiers define reward
- AgentRL handles the single-GPU runtime and post-training workflow
