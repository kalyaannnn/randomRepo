# V1 BYOD And Notebook Design

## Context

AgentRL v1 should finish as a coherent single-GPU post-training stack with:

- a stable runtime and trainer story
- a clear bootstrap to RL workflow
- a first-class bring-your-own-dataset story
- demo material that shows both the bundled GSM8K path and the intended external-task onboarding path

The current repo already has:

- the low-level task contracts via `BaseEnvironment` and `BaseVerifier`
- bootstrap training via `SFTBootstrapTrainer`
- GRPO training via `GRPOTrainer`
- a lightweight BYOD example in `examples/byod_task.py`

What is missing for v1 is a more ergonomic official onboarding API and notebook demos that present the intended user workflow cleanly.

## Problem

The current low-level API is technically sufficient, but it does not yet feel like a canonical user-facing workflow for:

1. bringing a custom dataset
2. defining prompt formatting or environment behavior
3. defining verifier or reward logic
4. running bootstrap, diagnostics, GRPO, and evaluation

That gap matters because the intended product story is not "use our baked-in benchmark paths." It is "bring your own data and logic, and AgentRL provides the runtime and training abstraction."

## Goals

- Finish v1 around the current product story before expanding the v2 paged-KV track.
- Add a small official high-level BYOD API for notebook-first onboarding.
- Preserve the existing core contracts as the underlying abstraction.
- Make the user workflow resemble:
  `bring dataset -> define formatting/environment hook -> define verifier/reward hook -> bootstrap -> diagnose -> GRPO -> evaluate`
- Ship two notebooks:
  - GSM8K end-to-end demo
  - canonical BYOD onboarding demo for custom datasets and custom reward logic

## Non-Goals

- Do not add a baked-in code benchmark or first-class code dataset integration for v1.
- Do not replace `BaseEnvironment`, `BaseVerifier`, `GRPOTrainer`, or `SFTBootstrapTrainer`.
- Do not introduce a plugin marketplace, dataset registry, or hosted task service.
- Do not fold paged-KV runtime-engine work into the v1 closure effort.

## Approaches Considered

### Approach 1: Keep only low-level APIs and improve docs

Pros:

- lowest implementation risk
- no new public API to support

Cons:

- does not materially improve notebook ergonomics
- leaves users to assemble too much framework knowledge themselves
- weakens the "bring your own data" product story

### Approach 2: Add a small official BYOD high-level API

Pros:

- improves the public onboarding path without destabilizing core internals
- keeps the existing low-level abstractions as the source of truth
- fits notebook usage and tutorial material well

Cons:

- introduces a small public API surface that needs tests and docs
- requires careful scoping to avoid overdesign

### Approach 3: Add a fully declarative callback-heavy training surface

Pros:

- potentially very polished for notebook users

Cons:

- too much new surface area for v1
- risks locking in the wrong abstractions
- harder to validate and maintain quickly

## Recommendation

Choose Approach 2.

AgentRL should expose a small official BYOD API that composes existing primitives instead of replacing them. This gives users a clear, ergonomic notebook story while keeping the current runtime and trainer contracts stable.

## Proposed Design

### 1. Official BYOD module

Add a small library-owned module, likely `agentrl/byod.py`, that provides notebook-friendly builders and wrappers.

The high-level API should let users provide:

- dataset records or a dataset loader
- a prompt formatting hook
- a verifier or reward hook
- an optional supervised target hook for bootstrap
- optional state transition logic for multi-turn tasks where needed

Internally, the module should adapt these hooks into concrete `BaseEnvironment` and `BaseVerifier` implementations.

### 2. Public API shape

The public API should stay small and explicit. The exact naming can change during implementation, but the design should include constructs equivalent to:

- a task or builder object that accepts dataset data plus hooks
- a default single-turn path for the common case
- an optional multi-turn path for advanced tasks
- helpers to produce bootstrap samples from user-provided supervised targets

The common user path should not require subclassing framework base classes directly unless the user wants lower-level control.

### 3. Layering

The high-level BYOD API is an adapter layer only.

It should compile down to the existing stable contracts:

- `BaseEnvironment`
- `BaseVerifier`
- `SFTBootstrapTrainer`
- `GRPOTrainer`

This keeps the implementation low-risk and preserves existing internal architecture.

### 4. Notebook story

Notebook 1 should demonstrate the existing GSM8K benchmark workflow:

- bootstrap
- diagnostic eval
- GRPO
- final eval

Notebook 2 should demonstrate the canonical BYOD onboarding path:

- load or define a custom dataset
- define a formatting/environment hook
- define a verifier/reward hook
- optionally define supervised targets
- construct the high-level BYOD task
- run bootstrap, diagnostics, GRPO, and eval

The second notebook should teach the abstraction, not a baked-in benchmark.

### 5. V1 closure framing

V1 should be considered complete when:

- the current README and docs reflect the intended product story cleanly
- the official BYOD high-level API exists and is tested
- the GSM8K and BYOD notebooks provide the recommended public demo shape
- the v2 paged-KV work remains clearly separated as a later runtime-engine track

## User Experience Requirements

The BYOD workflow should make the following easy:

- pass in records from a local file, in-memory list, or external dataset loader
- specify how raw records become model prompts
- specify how generated responses are scored
- specify optional supervised targets for bootstrap
- use the same task definition across bootstrap, RL, and evaluation

The API should avoid hidden magic. It should remain obvious how user hooks map onto environment and verifier behavior.

## Testing Requirements

Implementation should include tests for:

- single-turn BYOD task construction from user hooks
- bootstrap sample generation from supervised target hooks
- verifier hook integration into reward computation
- compatibility with the existing trainers
- basic notebook-facing examples or doc snippets remaining in sync with the public API

## Documentation Requirements

Update:

- `README.md` to present the BYOD onboarding flow as an official path
- `docs/bring_your_own_task.md` to explain the new high-level API and when to drop down to low-level contracts
- notebook assets and supporting docs so the two-demo story is easy to follow

## Risks

- Overdesigning the high-level API and accidentally creating a second framework inside the first one.
- Hiding too much of the task abstraction and making advanced customization harder.
- Letting notebook convenience leak runtime-specific assumptions into the public task model.

## Risk Mitigations

- Keep the public surface narrow.
- Keep the adapter layer thin and explicit.
- Preserve the low-level contracts untouched.
- Treat the high-level API as a convenience path, not a replacement architecture.

## Open Decisions

These should be resolved during implementation planning:

- exact public names in the BYOD module
- whether single-turn and multi-turn flows share one builder or two small entrypoints
- whether evaluation helpers should be part of the BYOD module or remain notebook-level composition

## Success Criteria

This work is successful if:

- a new user can follow the BYOD notebook without reading internal framework code first
- the official onboarding flow clearly communicates the intended AgentRL abstraction
- the repo has a complete v1 demo story before additional v2 runtime-engine work resumes
