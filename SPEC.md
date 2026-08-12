# BeatBlock AI
## Agent-Ready Technical Specification

**Status:** v0.1 specification  
**Primary language:** Python 3.11  
**Initial model:** `Qwen/Qwen3-1.7B`  
**Initial interface:** CLI  
**Primary goal:** Learn local SLM inference, supervised fine-tuning, preference learning, reward modeling, and feedback loops by building a useful chord recommendation system.

---

# 1. Product Summary

BeatBlock is a local AI-assisted chord recommendation system for producers and songwriters who are stuck deciding what chord should come next.

The user provides:

- musical key
- existing chord progression
- genre or vibe
- mood descriptors
- desired harmonic tension
- optionally tempo and song section

BeatBlock then:

1. generates a deterministic set of musically plausible next-chord candidates;
2. asks a small local language model to rank those candidates according to the musical context and requested vibe;
3. returns the top recommendations with structured metadata;
4. eventually records user selections as preference data;
5. later uses that feedback for DPO, reward-model training, and RL experiments.

The core ML problem is **ranking**, not unconstrained chord generation.

---

# 2. Learning Goals

The project should explicitly teach and demonstrate:

1. Local SLM inference.
2. Structured prompting.
3. Deterministic candidate generation.
4. Dataset design and JSONL pipelines.
5. Offline evaluation.
6. LoRA / QLoRA supervised fine-tuning.
7. Preference-pair collection.
8. Direct Preference Optimization.
9. Reward-model training.
10. Custom reward functions.
11. Online feedback loops.
12. Model/version comparison.
13. Reproducible ML experiments.
14. Public open-source project organization.

The project should make it possible to demonstrate measurable improvement between:

- base model
- SFT model
- SFT + DPO model
- optional reward-model / RL variants

---

# 3. Non-Goals for v0.1

Do NOT build these during the first milestone:

- React or Next.js frontend
- DAW plugin
- VST/AU plugin
- account system
- cloud inference
- live collaboration
- MIDI generation
- audio generation
- automatic key detection from audio
- reinforcement learning
- DPO
- reward models
- vector databases
- RAG
- arbitrary natural-language music production assistant

The first version is a CLI chord recommender with reproducible evaluation.

---

# 4. Core Design Principle

Separate **musical validity** from **musical preference**.

## Layer A: deterministic music engine

Responsible for:

- parsing key
- parsing current progression
- deriving scale degrees
- generating plausible candidate chords
- transposition
- avoiding exact duplicates
- attaching functional metadata

## Layer B: SLM ranker

Responsible for:

- ranking valid candidates
- matching candidates to vibe
- matching desired tension
- matching genre vocabulary
- explaining recommendations

The SLM should not generate arbitrary chord symbols outside the candidate list during v0.1.

---

# 5. High-Level Architecture

```text
User Input
    |
    v
Context Validation
    |
    v
Music Theory Engine
    |
    v
Candidate Generator
    |
    | 10-30 plausible chords
    v
SLM Ranker
    |
    v
Structured RankedResult
    |
    v
CLI Output
    |
    v
Optional Feedback Store
```

Future:

```text
Feedback Store
    |
    +--> chosen/rejected preference pairs
    |
    +--> DPO training
    |
    +--> reward-model training
    |
    +--> RL experiments
```

---

# 6. Initial Technology Stack

## Required

- Python 3.11
- `uv` for environment and dependency management
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- Accelerate
- Pydantic
- Typer
- Rich
- music21
- pytest
- Ruff

## Later

- bitsandbytes for 4-bit QLoRA where supported
- SQLite for local preference events
- Gradio or a small web UI
- MIDI playback library
- Hugging Face Hub

---

# 7. Default Model

Use:

```text
Qwen/Qwen3-1.7B
```

Requirements:

- model identifier must live in configuration, not be hard-coded throughout the codebase;
- inference must support CPU, CUDA, and MPS where available;
- use non-thinking mode for ranking;
- model output must be parsed into validated structured output;
- malformed model output must fail gracefully.

Example config:

```yaml
model:
  id: Qwen/Qwen3-1.7B
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  enable_thinking: false
```

---

# 8. Repository Structure

```text
beatblock-ai/
|
|-- README.md
|-- SPEC.md
|-- AGENTS.md
|-- LICENSE
|-- pyproject.toml
|-- uv.lock
|-- .gitignore
|
|-- configs/
|   |-- model.yaml
|   |-- candidate_rules.yaml
|   `-- training/
|       |-- sft.yaml
|       |-- dpo.yaml
|       `-- reward.yaml
|
|-- data/
|   |-- README.md
|   |-- samples/
|   |   |-- golden_sample.jsonl
|   |   `-- preference_sample.jsonl
|   |
|   `-- eval/
|       `-- eval_sample.jsonl
|
|-- src/
|   `-- beatblock/
|       |-- __init__.py
|       |
|       |-- domain/
|       |   |-- models.py
|       |   `-- enums.py
|       |
|       |-- music/
|       |   |-- parsing.py
|       |   |-- theory.py
|       |   |-- candidates.py
|       |   |-- transpose.py
|       |   `-- scoring.py
|       |
|       |-- model/
|       |   |-- loader.py
|       |   |-- prompt.py
|       |   |-- ranker.py
|       |   `-- parser.py
|       |
|       |-- evaluation/
|       |   |-- metrics.py
|       |   `-- runner.py
|       |
|       |-- feedback/
|       |   |-- models.py
|       |   `-- repository.py
|       |
|       `-- cli.py
|
|-- training/
|   |-- prepare_sft.py
|   |-- train_sft.py
|   |-- prepare_dpo.py
|   |-- train_dpo.py
|   |-- train_reward.py
|   `-- train_grpo.py
|
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- fixtures/
|
`-- scripts/
    |-- smoke_test_model.py
    `-- evaluate_model.py
```

---

# 9. Domain Models

Use Pydantic models for external and persisted structures.

## RecommendationContext

```python
class RecommendationContext(BaseModel):
    key: str
    progression: list[str]
    genre: str
    moods: list[str] = []
    tension: float = Field(ge=0.0, le=1.0)
    bpm: int | None = Field(default=None, ge=20, le=300)
    section: str | None = None
```

Example:

```json
{
  "key": "D minor",
  "progression": ["Dm9", "Gm9"],
  "genre": "jazz_rap",
  "moods": ["dark", "soulful"],
  "tension": 0.6,
  "bpm": 84,
  "section": "verse"
}
```

## CandidateChord

```python
class CandidateChord(BaseModel):
    symbol: str
    degree: str
    function: str
    source_rule: str
    theory_score: float
```

Example:

```json
{
  "symbol": "A7#9",
  "degree": "V7alt",
  "function": "dominant",
  "source_rule": "minor_dominant_altered",
  "theory_score": 0.91
}
```

## RankedCandidate

```python
class RankedCandidate(BaseModel):
    symbol: str
    rank: int
    model_score: float
    reason: str
```

## RecommendationResult

```python
class RecommendationResult(BaseModel):
    context: RecommendationContext
    candidates_generated: int
    recommendations: list[RankedCandidate]
    model_id: str
    model_version: str | None = None
```

---

# 10. Candidate Generation v0.1

The candidate generator must be deterministic.

Input:

```text
Key: D minor
Progression: Dm9 -> Gm9
```

Candidate families should initially include:

1. diatonic triads
2. diatonic seventh chords
3. diatonic ninth extensions where sensible
4. harmonic-minor dominant
5. secondary dominants
6. common modal-mixture options
7. passing diminished chords
8. common jazz extensions

Do not attempt to cover every known harmonic device.

Each generated candidate must contain:

- chord symbol
- roman numeral / functional degree
- harmonic function
- rule that produced it
- deterministic theory score

Remove:

- duplicate chord symbols
- exact repetition of the final progression chord unless explicitly allowed
- impossible/unparseable chords
- candidates outside configured rule families

Default target:

```text
10 <= candidate count <= 30
```

---

# 11. music21 Usage

Use music21 for representation and validation where useful.

Good use cases:

- `ChordSymbol`
- key representation
- RomanNumeral
- pitch handling
- transposition

Do not hide all domain logic inside music21 calls.

BeatBlock must own explicit candidate-generation rules so they can be inspected, tested, scored, and modified.

---

# 12. Ranking Prompt Contract

The model receives:

- system instruction
- normalized musical context
- candidate list
- strict output schema

Example conceptual prompt:

```text
You are BeatBlock's harmonic candidate ranker.

Rank ONLY the supplied candidate chords.

Optimize for:
1. compatibility with the current progression;
2. requested genre;
3. requested moods;
4. requested tension;
5. musical usefulness.

Do not invent additional chord symbols.

Return valid JSON only.

CONTEXT
Key: D minor
Progression: Dm9 -> Gm9
Genre: jazz_rap
Mood: dark, soulful
Desired tension: 0.60

CANDIDATES
1. A7#9
2. Bbmaj7
3. Cmaj7
4. Em7b5
5. Gm11
...
```

Expected output:

```json
{
  "recommendations": [
    {
      "symbol": "A7#9",
      "rank": 1,
      "model_score": 0.91,
      "reason": "Strong altered dominant tension while preserving the requested dark jazz vocabulary."
    }
  ]
}
```

Validation requirements:

- all returned symbols must exist in the supplied candidate set;
- rank values must be unique;
- scores must be between 0 and 1;
- parser must reject extra invented candidates;
- parser should permit one repair attempt if JSON is malformed;
- if repair fails, return a typed application error rather than silently continuing.

---

# 13. CLI Contract

Command:

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

Output should display:

```text
BeatBlock
Key: D minor
Progression: Dm9 -> Gm9
Genre: jazz_rap

1. A7#9       score: 0.91
   Strong altered dominant tension...

2. Cmaj7      score: 0.82
   ...

3. Bbmaj7     score: 0.76
   ...
```

Additional development command:

```bash
uv run beatblock candidates ...
```

This must show deterministic candidates without loading the SLM.

---

# 14. Golden SFT Dataset

Use JSONL.

Do not begin by generating thousands of examples.

First target:

```text
100 manually reviewed examples
```

Then expand only after the schema and evaluation methodology are stable.

Recommended record:

```json
{
  "id": "gold_000001",
  "context": {
    "key": "D minor",
    "progression": ["Dm9", "Gm9"],
    "degrees": ["i9", "iv9"],
    "genre": "jazz_rap",
    "moods": ["dark", "soulful"],
    "tension": 0.6
  },
  "candidates": [
    "A7#9",
    "Bbmaj7",
    "Cmaj7",
    "Em7b5"
  ],
  "target_ranking": [
    "A7#9",
    "Cmaj7",
    "Bbmaj7",
    "Em7b5"
  ],
  "best": "A7#9",
  "explanation": "..."
}
```

---

# 15. Transposition Augmentation

Golden examples should support transposition augmentation.

The musical relationship:

```text
i9 -> iv9 -> V7alt
```

should be representable in multiple keys.

Rules:

- preserve functional relationships;
- preserve chord quality;
- preserve labels;
- correctly spell enharmonics where possible;
- generated transpositions must be tagged as augmented data;
- evaluation examples must never be created by transposing a training example into another key.

This is important to prevent data leakage.

---

# 16. Dataset Splits

Maintain immutable IDs.

Recommended split:

```text
train       70%
validation  15%
test        15%
```

For the small initial dataset, exact percentages can vary.

Critical requirement:

Related transpositions of the same source progression must remain in the same split.

Never put a transposed derivative of a training example into the test set.

---

# 17. Baseline Evaluation

Evaluation must exist BEFORE fine-tuning.

Required metrics:

## Top-1 accuracy

Does the model's first choice equal the human-preferred target?

## Top-3 accuracy

Does the target occur in the top three?

## Mean Reciprocal Rank

For preferred target rank `r`:

```text
RR = 1 / r
```

MRR is the average reciprocal rank across evaluation examples.

## Candidate validity rate

Percentage of returned chords that are present in the supplied candidate list.

Target:

```text
100%
```

## Structured-output success rate

Percentage of inference requests that return valid schema-compatible output.

Track all metrics by:

- overall
- genre
- tension bucket
- progression length

---

# 18. Evaluation Output

Every model evaluation should create an artifact similar to:

```json
{
  "model": "Qwen/Qwen3-1.7B",
  "experiment": "baseline-001",
  "dataset_version": "eval-v1",
  "metrics": {
    "top_1": 0.31,
    "top_3": 0.59,
    "mrr": 0.47,
    "candidate_validity": 1.0,
    "structured_output_success": 0.98
  }
}
```

Do not hard-code or fabricate metric results.

Commit small result summaries to Git.

Do not commit large model checkpoints.

---

# 19. SFT Phase

SFT begins only after:

- candidate generator works;
- baseline inference works;
- golden schema is stable;
- eval set is frozen;
- baseline metrics have been recorded.

Use:

- Transformers
- TRL `SFTTrainer`
- PEFT LoRA
- optionally QLoRA / 4-bit loading when supported

Initial training objective:

Teach the model to rank the supplied candidates according to context and produce reliable structured output.

Do not try to teach broad music theory from scratch.

---

# 20. Preference Feedback Schema

Future interactive application records user decisions.

Example:

```json
{
  "event_id": "evt_123",
  "timestamp": "2026-08-12T14:00:00Z",
  "context_id": "ctx_456",
  "context": {},
  "shown": ["A7#9", "Cmaj7", "Bbmaj7"],
  "selected": "A7#9",
  "rejected": ["Cmaj7", "Bbmaj7"],
  "model_id": "beatblock-sft-v0.1",
  "event_type": "selection"
}
```

Potential event types:

- impression
- audition
- selection
- rejection
- undo
- favorite
- request_more
- kept_in_progression

Never interpret every event as equivalent reward.

---

# 21. DPO Dataset

Convert feedback into pairs:

```json
{
  "prompt": "...serialized musical context...",
  "chosen": "A7#9",
  "rejected": "Cmaj7"
}
```

Train using TRL `DPOTrainer`.

DPO should be a separate experiment from SFT.

Always compare:

```text
base
vs
SFT
vs
SFT + DPO
```

on the exact same frozen test set.

---

# 22. Reward Model

Later create:

```text
R(context, candidate) -> scalar score
```

Potential small reward model:

```text
Qwen3-0.6B
```

Potential reward components:

```text
human preference
theory compatibility
voice-leading quality
vibe compatibility
desired tension
novelty/diversity
```

Reward function design must be explicit and versioned.

Example:

```text
reward_v1 =
0.50 * human_preference
+ 0.15 * theory
+ 0.10 * voice_leading
+ 0.15 * vibe
+ 0.10 * tension_match
```

Weights above are placeholders, not validated defaults.

---

# 23. RL / GRPO Phase

GRPO is an experimental later milestone.

Do not introduce it until:

- SFT provides a meaningful improvement;
- sufficient preference data exists;
- reward functions have unit tests;
- reward hacking checks exist;
- frozen evaluation remains available.

The project should support multiple reward functions so their effect can be ablated independently.

---

# 24. Feedback Loop

Long-term loop:

```text
Model recommends candidates
        |
        v
User auditions/selects
        |
        v
Feedback events stored
        |
        v
Preference pairs created
        |
        v
Offline dataset snapshot
        |
        v
DPO / reward training
        |
        v
Offline evaluation
        |
        v
Promote model only if metrics improve
```

Never automatically retrain and replace the active model from raw live events.

Training should happen from versioned snapshots.

---

# 25. Experiment Tracking

Every training experiment must record:

- experiment ID
- git commit SHA
- base model
- dataset version
- training config
- random seed
- LoRA config
- epochs
- batch size
- learning rate
- hardware information
- resulting metrics
- output adapter path

A simple JSON experiment manifest is sufficient initially.

Do not require MLflow or Weights & Biases for v0.1.

---

# 26. Testing Strategy

## Unit tests

Required for:

- context validation
- key parsing
- chord parsing
- candidate deduplication
- theory rule outputs
- transposition
- candidate filtering
- model output parsing
- evaluation metrics
- preference conversion

## Integration tests

Required for:

- candidate generation end to end
- ranker with a mocked model
- CLI
- loading 3-5 sample dataset records

## Model smoke test

Optional in CI.

Do not require downloading a 1.7B model for normal GitHub Actions tests.

---

# 27. CI

GitHub Actions should initially run:

```text
ruff check
pytest
```

Optional:

```text
mypy
```

CI must not:

- download model weights;
- run SFT;
- require GPU;
- publish artifacts automatically.

---

# 28. Public Repository Rules

Commit:

- source code
- tests
- configs
- documentation
- small sample datasets
- small evaluation summaries
- training scripts
- lockfile

Do NOT commit:

- model checkpoints
- `.safetensors`
- `.gguf`
- Hugging Face cache
- raw large datasets
- secrets
- access tokens
- private user feedback database
- experiment outputs larger than necessary

Recommended `.gitignore` entries:

```gitignore
.venv/
.env
__pycache__/
.pytest_cache/
.ruff_cache/

models/
checkpoints/
outputs/
artifacts/
wandb/

*.safetensors
*.gguf
*.pt
*.pth

data/private/
data/raw/
*.db
*.sqlite
```

---

# 29. Licensing

Recommended for project source code:

```text
MIT
```

Before publishing:

- verify license compatibility for each base model;
- do not redistribute training data without redistribution rights;
- publish attribution/model-card information for derivative adapters;
- maintain a `THIRD_PARTY_NOTICES.md` if needed.

---

# 30. Security and Privacy

Never commit:

- Hugging Face tokens
- GitHub tokens
- local file paths containing personal information
- private feedback event databases

Secrets belong in environment variables or authenticated local tooling.

The application should work without sending progression data to a third-party inference API in the default local configuration.

---

# 31. Milestones

## M0 - Repository Bootstrap

Deliver:

- Python project
- dependency management
- package structure
- linting
- tests
- CLI skeleton
- CI

Acceptance:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run beatblock --help
```

all succeed.

---

## M1 - Deterministic Candidate Generator

Deliver:

- input schema
- key parsing
- progression parsing
- candidate rules
- candidate metadata
- CLI `candidates` command

Acceptance:

```bash
uv run beatblock candidates \
  --key "D minor" \
  --progression "Dm9,Gm9"
```

returns 10-30 unique valid candidates.

No model is loaded.

---

## M2 - Base Model Ranker

Deliver:

- Qwen loader
- prompt serializer
- output parser
- CLI `recommend`
- configurable model ID

Acceptance:

Given a valid context, BeatBlock:

1. creates candidates;
2. ranks only those candidates;
3. returns at least top 3;
4. produces schema-valid output;
5. stores no chain-of-thought.

---

## M3 - Evaluation Harness

Deliver:

- frozen eval JSONL
- evaluation runner
- Top-1
- Top-3
- MRR
- validity
- structured-output rate

Acceptance:

```bash
uv run python scripts/evaluate_model.py \
  --model Qwen/Qwen3-1.7B \
  --dataset data/eval/eval_v1.jsonl
```

creates a versioned metrics JSON file.

---

## M4 - Golden Dataset v1

Deliver:

- 100 manually reviewed source examples
- schema validator
- split generator
- transposition augmentation
- leakage checks

Acceptance:

Every dataset record validates and no source-family ID occurs across multiple splits.

---

## M5 - SFT

Deliver:

- SFT preparation script
- LoRA configuration
- training script
- adapter output
- post-training evaluation

Acceptance:

The same frozen evaluation harness can compare base and SFT.

Promotion criterion:

Do not call SFT "better" unless predefined primary metrics improve.

---

## M6 - Feedback Capture

Deliver:

- SQLite event store
- selection/rejection events
- deterministic preference-pair generator

Acceptance:

A recommendation interaction can be transformed into reproducible chosen/rejected DPO records.

---

## M7 - DPO

Deliver:

- DPO dataset snapshot
- DPO config
- training script
- comparison report

Acceptance:

Compare base vs SFT vs DPO using the same test set.

---

## M8 - Reward Model

Deliver:

- paired reward dataset
- reward training script
- reward evaluator

Acceptance:

Reward model ranks chosen above rejected at a measured rate on a held-out set.

---

## M9 - RL Experiment

Deliver:

- explicit reward functions
- reward-function unit tests
- GRPO experiment
- ablation report

This milestone is experimental and may be abandoned if it does not outperform simpler approaches.

---

# 32. Agent Implementation Contract

Coding agents working on this repository must follow these rules.

## Rule 1

Implement only the current milestone unless explicitly instructed otherwise.

## Rule 2

Do not add frontend frameworks before the CLI milestones are complete.

## Rule 3

Do not introduce new infrastructure without a concrete requirement.

## Rule 4

Prefer deterministic functions and typed domain models.

## Rule 5

Every nontrivial music-theory rule requires tests.

## Rule 6

Every persisted schema uses a version field once the project reaches dataset v1.

## Rule 7

Model IDs and training hyperparameters belong in configuration.

## Rule 8

Do not silently repair invalid model behavior. Validate it and make failures observable.

## Rule 9

No training/test leakage.

## Rule 10

Do not claim model improvement without running the frozen evaluation suite.

## Rule 11

Do not commit model weights.

## Rule 12

Before finishing a task, run:

```bash
uv run ruff check .
uv run pytest
```

## Rule 13

For significant architectural decisions, add a short decision note under:

```text
docs/decisions/
```

## Rule 14

Keep commits milestone-scoped and explain behavioral changes in commit messages.

---

# 33. Definition of v0.1 Done

BeatBlock v0.1 is complete when a developer can clone the repo and run:

```bash
uv sync
uv run beatblock recommend \
  --key "D minor" \
  --progression "Dm9,Gm9" \
  --genre jazz_rap \
  --mood dark \
  --mood soulful \
  --tension 0.6
```

and receive at least three ranked candidate chords from a locally loaded Qwen3-1.7B model.

The system must:

- generate the candidate list deterministically;
- prevent the model from returning candidates outside that list;
- validate all structured outputs;
- run locally;
- have unit tests;
- have baseline evaluation infrastructure;
- keep model weights out of Git.

Fine-tuning is explicitly NOT required for v0.1.

---

# 34. First Coding-Agent Prompt

Use this as the first implementation instruction after placing this specification in the repository:

```text
Read SPEC.md completely before making changes.

Implement milestone M0 only.

Create the Python 3.11 project structure described in the specification using uv.
Add the package skeleton, Typer CLI, Pydantic domain-model placeholder, Ruff configuration,
pytest setup, .gitignore, MIT license placeholder, and GitHub Actions workflow.

The following commands must pass before completion:

uv sync
uv run beatblock --help
uv run ruff check .
uv run pytest

Do not implement candidate generation, model inference, training, frontend code, or feedback storage yet.

When finished, summarize:
1. files created;
2. architectural decisions made;
3. commands executed;
4. test results;
5. anything deferred to M1.
```

---

# 35. Second Coding-Agent Prompt

After M0 is complete:

```text
Read SPEC.md and inspect the current repository.

Implement milestone M1 only: Deterministic Candidate Generator.

Add:
- RecommendationContext;
- CandidateChord;
- key/progression validation;
- music21-backed chord and key representation;
- explicit candidate-generation rules;
- deterministic candidate scores;
- candidate deduplication;
- CLI `beatblock candidates`.

Support at minimum:
- major and minor keys;
- diatonic triads;
- diatonic sevenths;
- common ninth extensions;
- harmonic-minor V7;
- secondary dominants;
- a limited documented modal-mixture set;
- passing diminished candidates.

Add unit tests for all implemented rule families.

Do not load or call an SLM yet.

Run:
uv run ruff check .
uv run pytest

Then report acceptance-criteria results from M1.
```

---

# 36. Guiding Principle

The project is successful if it answers this question with evidence:

> Can a small local model become measurably better at ranking musically valid next-chord options for a requested vibe after supervised and preference-based post-training?

Every architectural decision should support answering that question.
