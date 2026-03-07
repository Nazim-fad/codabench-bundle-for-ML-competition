import json
import sys
import time
from pathlib import Path

import pandas as pd


EVAL_SETS = ["test", "private_test"]


def load_features(data_dir: Path, split: str) -> pd.DataFrame:
    split_dir = data_dir / split
    features_path = split_dir / f"{split}_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    return pd.read_csv(features_path)


def load_train_labels(data_dir: Path) -> pd.DataFrame:
    labels_path = data_dir / "train" / "train_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing train labels file: {labels_path}")
    return pd.read_csv(labels_path)


def normalize_prediction(pred: dict) -> dict:
    """
    Required prediction format for each detected segment:
        {"start": float, "end": float}
    """
    if not isinstance(pred, dict):
        raise ValueError(
            "Each predicted segment must be a dict like {'start': ..., 'end': ...}."
        )

    if "start" not in pred or "end" not in pred:
        raise ValueError(
            "Each predicted segment dict must contain 'start' and 'end' keys."
        )

    start = float(pred["start"])
    end = float(pred["end"])
    return {"start": start, "end": end}


def sanitize_predictions(sample_id: str, preds) -> list[dict]:
    """
    Convert model output into a clean list of rows:
        [{"sample_id": ..., "start": ..., "end": ...}, ...]

    Rules:
    - model.predict(audio_path) must return a list
    - each item must be {"start": ..., "end": ...}
    - invalid segments (end <= start) are skipped
    """
    if preds is None:
        preds = []

    if not isinstance(preds, list):
        raise ValueError(
            "model.predict(audio_path) must return a list of segment dicts."
        )

    rows = []
    for pred in preds:
        seg = normalize_prediction(pred)

        start = max(0.0, float(seg["start"]))
        end = float(seg["end"])

        if end <= start:
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "start": round(start, 4),
                "end": round(end, 4),
            }
        )

    rows.sort(key=lambda x: (x["sample_id"], x["start"], x["end"]))
    return rows


def evaluate_model(model, features_df: pd.DataFrame, split_dir: Path) -> pd.DataFrame:
    """
    For each row in features_df:
      - read sample_id and audio_path
      - call model.predict(audio_path)
      - collect all predicted segments into one dataframe

    Expected features_df columns:
      - sample_id
      - audio_path
    """
    required_cols = {"sample_id", "audio_path"}
    missing = required_cols - set(features_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in features dataframe: {sorted(missing)}"
        )

    all_rows = []

    for _, row in features_df.iterrows():
        sample_id = str(row["sample_id"])
        audio_rel_path = str(row["audio_path"])
        audio_path = split_dir / audio_rel_path

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found for sample_id={sample_id}: {audio_path}"
            )

        preds = model.predict(audio_path)
        rows = sanitize_predictions(sample_id, preds)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=["sample_id", "start", "end"])


def main(data_dir: Path, output_dir: Path) -> None:
    from submission import get_model

    print("Loading training data")
    train_features = load_features(data_dir, "train")
    train_labels = load_train_labels(data_dir)

    print("Initializing model")
    model = get_model()

    if not hasattr(model, "fit"):
        raise AttributeError(
            "Submission model must implement "
            "'fit(train_features_df, train_labels_df, data_dir)'."
        )

    if not hasattr(model, "predict"):
        raise AttributeError("Submission model must implement 'predict(audio_path)'.")

    print("Training the model")
    start = time.time()
    model.fit(train_features, train_labels, data_dir / "train")
    train_time = time.time() - start

    print("-" * 10)
    print("Evaluating the model")

    start = time.time()
    results = {}

    for eval_set in EVAL_SETS:
        print(f"Running predictions for {eval_set}")
        features_df = load_features(data_dir, eval_set)
        split_dir = data_dir / eval_set
        results[eval_set] = evaluate_model(model, features_df, split_dir)

    test_time = time.time() - start
    duration = train_time + test_time

    print("-" * 10)
    print(f"Completed prediction. Total duration: {duration:.2f}s")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "train_time": train_time,
                "test_time": test_time,
            },
            f,
        )

    for eval_set in EVAL_SETS:
        out_path = output_dir / f"{eval_set}_predictions.csv"
        results[eval_set].to_csv(out_path, index=False)

    print("\nIngestion program finished. Moving on to scoring.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for Codabench emergency audio event detection"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="Path to input_data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="Path where predictions will be written",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="Path to participant submission directory",
    )

    args = parser.parse_args()

    sys.path.append(args.submission_dir)
    sys.path.append(str(Path(__file__).parent.resolve()))

    main(Path(args.data_dir), Path(args.output_dir))
