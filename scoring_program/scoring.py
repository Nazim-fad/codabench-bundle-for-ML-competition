import json
from pathlib import Path

import pandas as pd


EVAL_SETS = ["test", "private_test"]
IOU_THRESHOLD = 0.5


def load_segments(csv_path: Path) -> pd.DataFrame:
    """
    Load segment CSV with columns:
      sample_id,start,end

    Return empty dataframe with correct columns if file does not exist
    or if the CSV is empty.
    """
    cols = ["sample_id", "start", "end"]

    if not csv_path.exists():
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)

    if df.empty:
        return pd.DataFrame(columns=cols)

    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df[cols].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)

    # Keep only valid segments
    df = df[df["end"] > df["start"]].copy()
    df = df.sort_values(["sample_id", "start", "end"]).reset_index(drop=True)
    return df


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def group_segments(df: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    grouped = {}
    if df.empty:
        return grouped

    for sample_id, g in df.groupby("sample_id"):
        grouped[str(sample_id)] = [
            (float(row.start), float(row.end)) for row in g.itertuples(index=False)
        ]
    return grouped


def match_segments(
    pred_segments: list[tuple[float, float]],
    true_segments: list[tuple[float, float]],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[int, int, int, list[float]]:
    """
    Greedy one-to-one matching between predicted and true segments.

    Returns:
      tp, fp, fn, matched_ious
    """
    if not pred_segments and not true_segments:
        return 0, 0, 0, []

    matched_pred = set()
    matched_true = set()
    candidate_pairs = []

    for i, (ps, pe) in enumerate(pred_segments):
        for j, (ts, te) in enumerate(true_segments):
            iou = interval_iou(ps, pe, ts, te)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, i, j))

    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    matched_ious = []
    for iou, i, j in candidate_pairs:
        if i in matched_pred or j in matched_true:
            continue
        matched_pred.add(i)
        matched_true.add(j)
        matched_ious.append(iou)

    tp = len(matched_pred)
    fp = len(pred_segments) - tp
    fn = len(true_segments) - tp
    return tp, fp, fn, matched_ious


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_segment_metrics(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, float]:
    pred_map = group_segments(pred_df)
    true_map = group_segments(true_df)

    all_sample_ids = sorted(set(pred_map) | set(true_map))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious = []

    for sample_id in all_sample_ids:
        pred_segments = pred_map.get(sample_id, [])
        true_segments = true_map.get(sample_id, [])

        tp, fp, fn, matched_ious = match_segments(
            pred_segments,
            true_segments,
            iou_threshold=iou_threshold,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_matched_ious.extend(matched_ious)

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = f1_score(precision, recall)
    mean_matched_iou = (
        float(sum(all_matched_ious) / len(all_matched_ious))
        if all_matched_ious
        else 0.0
    )

    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "segment_tp": float(total_tp),
        "segment_fp": float(total_fp),
        "segment_fn": float(total_fn),
        "segment_mean_iou": mean_matched_iou,
    }


def compute_presence_metrics(
    pred_df: pd.DataFrame, true_df: pd.DataFrame
) -> dict[str, float]:
    pred_positive = (
        set(pred_df["sample_id"].astype(str).unique()) if not pred_df.empty else set()
    )
    true_positive = (
        set(true_df["sample_id"].astype(str).unique()) if not true_df.empty else set()
    )

    tp = len(pred_positive & true_positive)
    fp = len(pred_positive - true_positive)
    fn = len(true_positive - pred_positive)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = f1_score(precision, recall)

    return {
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_tp": float(tp),
        "presence_fp": float(fp),
        "presence_fn": float(fn),
    }


def compute_presence_accuracy(
    features_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
) -> float:
    all_sample_ids = set(features_df["sample_id"].astype(str).unique())
    pred_positive = (
        set(pred_df["sample_id"].astype(str).unique()) if not pred_df.empty else set()
    )
    true_positive = (
        set(true_df["sample_id"].astype(str).unique()) if not true_df.empty else set()
    )

    correct = 0
    for sample_id in all_sample_ids:
        pred_is_pos = sample_id in pred_positive
        true_is_pos = sample_id in true_positive
        if pred_is_pos == true_is_pos:
            correct += 1

    return safe_div(correct, len(all_sample_ids))


def load_features(reference_dir: Path, eval_set: str) -> pd.DataFrame:
    """
    For presence accuracy we need the full set of sample_ids.
    Since scoring normally only receives reference_data and predictions,
    we store a copy of the public/private feature manifests in reference_data too.
    """
    path = reference_dir / f"{eval_set}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            "Reference data must include test/private_test feature manifests for presence metrics."
        )
    return pd.read_csv(path)


def main(reference_dir: Path, prediction_dir: Path, output_dir: Path):
    scores = {}

    for eval_set in EVAL_SETS:
        print(f"Scoring {eval_set}")

        pred_df = load_segments(prediction_dir / f"{eval_set}_predictions.csv")
        true_df = load_segments(reference_dir / f"{eval_set}_labels.csv")
        features_df = load_features(reference_dir, eval_set)

        segment_metrics = compute_segment_metrics(
            pred_df, true_df, iou_threshold=IOU_THRESHOLD
        )
        presence_metrics = compute_presence_metrics(pred_df, true_df)
        presence_accuracy = compute_presence_accuracy(features_df, pred_df, true_df)

        for key, value in segment_metrics.items():
            scores[f"{eval_set}_{key}"] = float(value)

        for key, value in presence_metrics.items():
            scores[f"{eval_set}_{key}"] = float(value)

        scores[f"{eval_set}_presence_accuracy"] = float(presence_accuracy)

    metadata_path = prediction_dir / "metadata.json"
    if metadata_path.exists():
        durations = json.loads(metadata_path.read_text())
        scores.update(**durations)

    print(scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for Codabench emergency audio event detection"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="Path to reference data directory",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="Path to ingestion outputs directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="Path where scores.json will be written",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir),
    )
