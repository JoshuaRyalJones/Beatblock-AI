# AGENTS.md

Instructions for coding agents working in this repository.

## Before you start

1. Read [SPEC.md](SPEC.md) completely. It is the source of truth for architecture, schemas,
   milestones, and acceptance criteria.
2. Identify the current milestone (see [SPEC.md §31](SPEC.md#31-milestones)).
3. Implement **only that milestone** unless explicitly instructed otherwise.

## Implementation contract

The full contract is [SPEC.md §32](SPEC.md#32-agent-implementation-contract). In short:

- Implement only the current milestone.
- No frontend frameworks until the CLI milestones are complete.
- No new infrastructure without a concrete requirement.
- Prefer deterministic functions and typed domain models (Pydantic).
- Every nontrivial music-theory rule requires tests.
- Model IDs and hyperparameters live in `configs/`, never hard-coded.
- Do not silently repair invalid model output — validate it and make failures observable.
- No train/test leakage. Transpositions of the same source progression stay in one split.
- Do not claim model improvement without running the frozen evaluation suite.
- Never commit model weights, checkpoints, tokens, or private feedback databases.

## Before finishing any task

```bash
uv run ruff check .
uv run pytest
```

Both must pass. Report actual output — do not fabricate metrics or test results.

## Commits

Keep commits milestone-scoped. Explain behavioral changes in the commit message.
Record significant architectural decisions as a short note in `docs/decisions/`.
