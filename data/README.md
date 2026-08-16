# Data

Only small, redistributable files belong in this directory. Raw datasets, private feedback
databases, and anything under `data/private/` or `data/raw/` are git-ignored.

## Layout

```text
samples/golden_sample.jsonl      illustrative SFT records (M4)
samples/preference_sample.jsonl  illustrative chosen/rejected pairs (M6/M7)
eval/eval_sample.jsonl           illustrative evaluation records (M3)
eval/eval_v1.jsonl               frozen baseline evaluation records (M3)
```

The sample files are placeholders until the schemas are frozen. The real frozen evaluation set
is versioned separately (`eval_v1.jsonl`, etc.) and must never change once baseline metrics
have been recorded against it.

`eval_v1.jsonl` is intentionally small and human-reviewable. Each record freezes the exact
candidate set used for comparison so candidate-generator changes cannot silently move the
benchmark. Expand it by creating a new dataset version, never by editing it after baseline
results exist.

## Golden record schema (SPEC.md §14)

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
  "candidates": ["A7#9", "Bbmaj7", "Cmaj7", "Em7b5"],
  "target_ranking": ["A7#9", "Cmaj7", "Bbmaj7", "Em7b5"],
  "best": "A7#9",
  "explanation": "..."
}
```

## Preference pair schema (SPEC.md §21)

```json
{
  "prompt": "...serialized musical context...",
  "chosen": "A7#9",
  "rejected": "Cmaj7"
}
```

## Splits and leakage

Target split is roughly 70 / 15 / 15 (train / validation / test) with immutable IDs.

**Transposed derivatives of a source progression must stay in the same split.** An evaluation
example may never be produced by transposing a training example. See SPEC.md §15–§16.
