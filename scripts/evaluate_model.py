"""Evaluate a configured local model against a frozen BeatBlock JSONL dataset."""

import argparse
from pathlib import Path

from beatblock.evaluation.runner import evaluate, load_evaluation_dataset, write_artifact
from beatblock.model.loader import load_inference_config, load_local_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID recorded in the artifact.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-version", default="eval-v1")
    parser.add_argument("--experiment", default="baseline-001")
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline-001.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_inference_config(args.config)
    if args.model != config.model.id:
        raise ValueError("--model must match the configured model ID")
    local_model = load_local_model(config)
    artifact = evaluate(
        load_evaluation_dataset(args.dataset),
        local_model.generate,
        model_id=args.model,
        experiment=args.experiment,
        dataset_version=args.dataset_version,
        enable_thinking=config.model.enable_thinking,
    )
    write_artifact(artifact, args.output)
    print(artifact.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
