# BeatBlock AI

**A local, offline chord recommendation system that learns your taste.**

BeatBlock helps producers and songwriters answer one question: *what chord should come next?*
You give it a key, the progression so far, a genre, a mood, and how much harmonic tension you
want. It generates musically valid candidate chords with a deterministic music-theory engine,
then uses a small local language model to rank those candidates against your requested vibe.

Everything runs on your machine. No cloud inference, no account, no progression data leaving
your laptop.

> **Status:** v0.1 in development. See [SPEC.md](SPEC.md) for the full technical specification
> and [milestone roadmap](SPEC.md#31-milestones).

---

## The core idea

BeatBlock deliberately separates **musical validity** from **musical preference**:

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| **A — Music engine** | What chords are *possible* here? | Deterministic, testable rules over `music21` |
| **B — SLM ranker** | Which of those fits the *vibe*? | `Qwen/Qwen3-1.7B` running locally |

The model never invents chord symbols outside the generated candidate set. That constraint is
what makes the system evaluable — and what turns "does this sound good?" into a **ranking**
problem with measurable metrics.

```text
 context ──▶ validation ──▶ theory engine ──▶ candidates ──▶ SLM ranker ──▶ ranked result
                                              (10–30)                        + reasons
```

---

## Why this project exists

BeatBlock is a useful tool *and* an end-to-end study of local model post-training. The roadmap
walks a small model through the full lifecycle, measuring at every step:

1. **Baseline** — evaluate `Qwen3-1.7B` out of the box
2. **SFT** — LoRA/QLoRA supervised fine-tuning on a hand-reviewed golden dataset
3. **Preference data** — capture real selections and rejections from actual use
4. **DPO** — direct preference optimization on those pairs
5. **Reward modeling & RL** — explicit, versioned, ablatable reward functions

Each stage is compared against the same frozen test set using Top-1, Top-3, MRR, candidate
validity, and structured-output success rate. The guiding question:

> Can a small local model become measurably better at ranking musically valid next-chord
> options for a requested vibe after supervised and preference-based post-training?

---

## Planned usage

```bash
uv run beatblock recommend \
  --key "D minor" \
  --progression "Dm9,Gm9" \
  --genre jazz_rap \
  --mood dark \
  --mood soulful \
  --tension 0.6 \
  --bpm 84
```

```text
BeatBlock
Key: D minor
Progression: Dm9 -> Gm9
Genre: jazz_rap

1. A7#9       score: 0.91
   Strong altered dominant tension while preserving the dark jazz vocabulary.

2. Cmaj7      score: 0.82
   ...
```

Inspect the deterministic candidate set without loading the model at all:

```bash
uv run beatblock candidates --key "D minor" --progression "Dm9,Gm9"
```

---

## Repository layout

```text
configs/     model, candidate-rule, and training configuration (no hard-coded model IDs)
data/        sample datasets and the frozen evaluation set — small files only
src/         the beatblock package: domain, music, model, evaluation, feedback, cli
training/    SFT / DPO / reward / GRPO preparation and training scripts
tests/       unit, integration, fixtures
scripts/     smoke tests and evaluation runners
docs/        architectural decision notes
```

---

## Stack

Python 3.11 · [uv](https://docs.astral.sh/uv/) · PyTorch · Transformers · PEFT · TRL ·
Pydantic · Typer · Rich · music21 · pytest · Ruff

---

## Development

### Prerequisites

Install Python 3.11 and `uv`. On Apple Silicon macOS with Homebrew:

```bash
brew install python@3.11 uv
```

Then create the project environment and install the lightweight CLI dependencies:

```bash
uv sync
uv run beatblock --help
uv run ruff check .
uv run pytest
```

The base install intentionally excludes PyTorch and the model-training stack. Add those when
M2 (local inference) begins:

```bash
uv sync --extra ml
```

You do not need a Hugging Face account or token for M0 or M1. Model weights, local virtual
environments, and caches stay outside Git.

### Learning checkpoints

Treat each milestone as a small experiment with a question you should be able to answer:

1. **M0 — tooling:** Can you run, test, and lint a Python CLI reproducibly?
2. **M1 — music engine:** Can you explain and test every rule that proposes a chord?
3. **M2–M3 — inference and evaluation:** How does prompting become validated rankings, and how
   do we establish an honest baseline before training?
4. **M4–M5 — data and SFT:** How do reviewed labels, leakage-safe splits, LoRA, and
   hyperparameters affect the frozen metrics?
5. **M6–M9 — preferences:** How do user choices become DPO pairs or explicit rewards without
   confusing every click with musical quality?

AI can help scaffold code, explain APIs, propose tests, and review experiment results. You
should remain the decision-maker for music-theory rules, manually review golden examples,
choose evaluation criteria before seeing results, and write a short hypothesis before each
training run. That is where most of the ML learning lives.

`ruff check` and `pytest` must pass before any task is considered complete.

Contributors — human and agent alike — should read [SPEC.md](SPEC.md) in full before making
changes, and implement **only the current milestone**. The agent implementation contract is in
[§32](SPEC.md#32-agent-implementation-contract).

---

## What v0.1 is not

No web frontend, no DAW or VST plugin, no cloud inference, no accounts, no MIDI or audio
generation, no RAG. v0.1 is a CLI chord recommender with reproducible evaluation — everything
else waits until that works and is measured.

---

## License

MIT — see [LICENSE](LICENSE).

Model weights are **never** committed to this repository. Base models are downloaded from the
Hugging Face Hub under their own respective licenses; verify compatibility before publishing
any derivative adapter.
