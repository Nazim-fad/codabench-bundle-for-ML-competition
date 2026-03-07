from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split


PHASE = "dev_phase"
TARGET_SR = 16000


ESC50_EMERGENCY = {
    "siren",
    "glass_breaking",
    "fireworks",
}

URBAN_EMERGENCY = {
    "siren",
    "gun_shot",
}

FSD50K_EMERGENCY = {
    "Alarm",
    "Boom",
    "Explosion",
    "Fire",
    "Fireworks",
    "Gunshot_and_gunfire",
    "Screaming",
    "Shatter",
    "Shout",
    "Siren",
    "Yell",
}

ESC50_HARD_NEG = {
    "clock_alarm",
    "door_wood_knock",
    "crying_baby",
    "coughing",
    "car_horn",
    "engine",
}

URBAN_HARD_NEG = {
    "car_horn",
    "jackhammer",
    "engine_idling",
    "drilling",
}

ESC50_EASY_NEG = {
    "rain",
    "chirping_birds",
    "wind",
    "dog",
    "cat",
    "footsteps",
    "clock_tick",
    "sea_waves",
    "water_drops",
    "pouring_water",
    "frog",
    "cow",
    "sheep",
    "crow",
    "crickets",
    "insects",
    "airplane",
    "train",
    "church_bells",
}

URBAN_EASY_NEG = {
    "street_music",
    "children_playing",
    "dog_bark",
    "air_conditioner",
}


@dataclass(frozen=True)
class SourceClip:
    sample_id: str
    dataset: str
    source_path: Path
    category: str
    split_group: str  # emergency / hard_negative / easy_negative
    event_start: Optional[float]
    event_end: Optional[float]


def stable_id(*parts: str) -> str:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak
    return x.astype(np.float32)


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)

    return normalize_audio(audio)


def write_wav(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def get_duration_sec(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    return float(len(audio) / sr)


def find_one_csv(folder: Path, contains: Optional[str] = None) -> Path:
    csvs = list(folder.rglob("*.csv"))
    if contains is not None:
        csvs = [p for p in csvs if contains.lower() in p.name.lower()]
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder} with filter={contains!r}")
    csvs.sort(key=lambda p: (len(str(p)), str(p)))
    return csvs[0]


def load_esc50(root: Path) -> list[SourceClip]:
    meta_csv = root / "meta" / "esc50.csv"
    if not meta_csv.exists():
        meta_csv = find_one_csv(root / "meta")

    df = pd.read_csv(meta_csv)
    clips: list[SourceClip] = []

    for _, row in df.iterrows():
        category = str(row["category"]).strip()
        filename = str(row["filename"]).strip()
        src_path = root / "audio" / filename
        if not src_path.exists():
            continue

        if category in ESC50_EMERGENCY:
            split_group = "emergency"
            event_start, event_end = 0.0, None
        elif category in ESC50_HARD_NEG:
            split_group = "hard_negative"
            event_start, event_end = None, None
        elif category in ESC50_EASY_NEG:
            split_group = "easy_negative"
            event_start, event_end = None, None
        else:
            continue

        clips.append(
            SourceClip(
                sample_id=stable_id("esc50", filename),
                dataset="esc50",
                source_path=src_path,
                category=category,
                split_group=split_group,
                event_start=event_start,
                event_end=event_end,
            )
        )

    return clips


def load_urbansound8k(root: Path) -> list[SourceClip]:
    meta_csv = root / "metadata" / "UrbanSound8K.csv"
    if not meta_csv.exists():
        meta_csv = find_one_csv(root / "metadata")

    df = pd.read_csv(meta_csv)
    clips: list[SourceClip] = []

    for _, row in df.iterrows():
        category = str(row["class"]).strip()
        fold = int(row["fold"])
        filename = str(row["slice_file_name"]).strip()

        src_path = root / "audio" / f"fold{fold}" / filename
        if not src_path.exists():
            continue

        if category in URBAN_EMERGENCY:
            split_group = "emergency"
            event_start = float(row["start"])
            event_end = float(row["end"])
        elif category in URBAN_HARD_NEG:
            split_group = "hard_negative"
            event_start, event_end = None, None
        elif category in URBAN_EASY_NEG:
            split_group = "easy_negative"
            event_start, event_end = None, None
        else:
            continue

        clips.append(
            SourceClip(
                sample_id=stable_id("urbansound8k", f"fold{fold}", filename),
                dataset="urbansound8k",
                source_path=src_path,
                category=category,
                split_group=split_group,
                event_start=event_start,
                event_end=event_end,
            )
        )

    return clips


def split_fsd_labels(value: str) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def load_fsd50k_emergency_only(
    root: Path,
    max_labels_allowed: int = 1,
) -> list[SourceClip]:
    gt_root = root / "FSD50K.ground_truth"
    audio_root = root / "FSD50K.dev_audio"

    gt_csvs = list(gt_root.rglob("*.csv"))
    if not gt_csvs:
        raise FileNotFoundError(f"No ground-truth CSV found in {gt_root}")

    gt_csv = None
    for p in gt_csvs:
        try:
            tmp = pd.read_csv(p, nrows=5)
            cols = set(tmp.columns)
            if ("labels" in cols or "label" in cols) and (
                "fname" in cols or "filename" in cols
            ):
                gt_csv = p
                break
        except Exception:
            continue

    if gt_csv is None:
        gt_csv = gt_csvs[0]

    df = pd.read_csv(gt_csv)
    fname_col = "fname" if "fname" in df.columns else "filename"
    labels_col = "labels" if "labels" in df.columns else "label"

    clips: list[SourceClip] = []

    for _, row in df.iterrows():
        fname = str(row[fname_col]).strip()
        labels_list = split_fsd_labels(row[labels_col])

        if not labels_list:
            continue
        if len(labels_list) > max_labels_allowed:
            continue

        labels = set(labels_list)
        emergency_labels = labels & FSD50K_EMERGENCY
        if not emergency_labels:
            continue

        candidates = [
            audio_root / "FSD50K.dev_audio" / f"{fname}.wav",
            audio_root / "FSD50K.dev_audio" / f"{fname}.flac",
            audio_root / "FSD50K.dev_audio" / f"{fname}.ogg",
            audio_root / "FSD50K.dev_audio" / f"{fname}.mp3",
            audio_root / f"{fname}.wav",
            audio_root / f"{fname}.flac",
            audio_root / f"{fname}.ogg",
            audio_root / f"{fname}.mp3",
        ]
        src_path = next((p for p in candidates if p.exists()), None)
        if src_path is None:
            continue

        category = sorted(emergency_labels)[0]
        clips.append(
            SourceClip(
                sample_id=stable_id("fsd50k", fname),
                dataset="fsd50k",
                source_path=src_path,
                category=category,
                split_group="emergency",
                event_start=0.0,
                event_end=None,
            )
        )

    return clips


def cap_fsd50k_emergency(
    clips: list[SourceClip], cap: Optional[int], seed: int
) -> list[SourceClip]:
    if cap is None:
        return clips

    rng = np.random.default_rng(seed)

    fsd = [c for c in clips if c.dataset == "fsd50k"]
    non_fsd = [c for c in clips if c.dataset != "fsd50k"]

    if len(fsd) <= cap:
        return clips

    idx = rng.choice(len(fsd), size=cap, replace=False)
    fsd_selected = [fsd[i] for i in idx]

    return non_fsd + fsd_selected


def rel_audio_path(sample_id: str) -> str:
    return f"audio/{sample_id}.wav"


def stratified_split(clips: list[SourceClip], seed: int) -> dict[str, list[SourceClip]]:
    groups = pd.Series([c.split_group for c in clips])
    idx = np.arange(len(clips))

    idx_train, idx_rest = train_test_split(
        idx,
        test_size=0.30,
        random_state=seed,
        stratify=groups,
    )
    rest_groups = groups.iloc[idx_rest]

    idx_test, idx_private = train_test_split(
        idx_rest,
        test_size=0.50,
        random_state=seed,
        stratify=rest_groups,
    )

    return {
        "train": [clips[i] for i in idx_train],
        "test": [clips[i] for i in idx_test],
        "private_test": [clips[i] for i in idx_private],
    }


def process_split(
    split_name: str,
    clips: list[SourceClip],
    input_root: Path,
    reference_root: Path,
) -> None:
    split_dir = input_root / split_name
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = []
    label_rows = []

    for clip in clips:
        out_wav = audio_dir / f"{clip.sample_id}.wav"

        try:
            audio = load_audio(clip.source_path, target_sr=TARGET_SR)
        except Exception as e:
            print(f"Skipping unreadable audio: {clip.source_path} ({e})")
            continue

        if len(audio) == 0:
            print(f"Skipping empty audio: {clip.source_path}")
            continue

        write_wav(out_wav, audio, TARGET_SR)
        clip_duration = get_duration_sec(audio, TARGET_SR)

        feature_rows.append(
            {
                "sample_id": clip.sample_id,
                "audio_path": rel_audio_path(clip.sample_id),
            }
        )

        if clip.split_group == "emergency":
            start = (
                0.0 if clip.event_start is None else max(0.0, float(clip.event_start))
            )
            end = clip_duration if clip.event_end is None else float(clip.event_end)

            start = min(start, clip_duration)
            end = min(end, clip_duration)

            # Repair zero-length intervals instead of dropping them
            if end <= start:
                min_duration = min(0.1, clip_duration)
                start = max(0.0, min(start, clip_duration - min_duration))
                end = start + min_duration

            label_rows.append(
                {
                    "sample_id": clip.sample_id,
                    "start": round(start, 4),
                    "end": round(end, 4),
                }
            )

    features_df = (
        pd.DataFrame(feature_rows).sort_values("sample_id").reset_index(drop=True)
    )
    labels_df = (
        pd.DataFrame(label_rows, columns=["sample_id", "start", "end"])
        .sort_values(["sample_id", "start", "end"])
        .reset_index(drop=True)
    )

    features_path = split_dir / f"{split_name}_features.csv"
    features_df.to_csv(features_path, index=False)

    if split_name == "train":
        labels_df.to_csv(split_dir / "train_labels.csv", index=False)
    else:
        reference_root.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(reference_root / f"{split_name}_labels.csv", index=False)

        features_df.to_csv(reference_root / f"{split_name}_features.csv", index=False)

    print(
        f"{split_name}: {len(features_df)} clips, {len(labels_df)} emergency segments"
    )


def write_summary(clips: list[SourceClip], out_csv: Path) -> None:
    df = pd.DataFrame(
        {
            "dataset": [c.dataset for c in clips],
            "category": [c.category for c in clips],
            "split_group": [c.split_group for c in clips],
            "source_path": [str(c.source_path) for c in clips],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def print_counts(title: str, clips: list[SourceClip]) -> None:
    df = pd.DataFrame(
        {
            "dataset": [c.dataset for c in clips],
            "split_group": [c.split_group for c in clips],
        }
    )
    print(f"\n{title}:")
    print(df.groupby(["dataset", "split_group"]).size())
    total = len(clips)
    n_pos = int((df["split_group"] == "emergency").sum())
    print(f"Emergency ratio: {n_pos}/{total} = {n_pos / total:.3f}")


def print_split_distribution(split_map: dict[str, list[SourceClip]]) -> None:
    print("\nSplit-wise class distribution:")
    for split_name, clips in split_map.items():
        df = pd.DataFrame(
            {
                "dataset": [c.dataset for c in clips],
                "split_group": [c.split_group for c in clips],
            }
        )
        print(f"\n{split_name}:")
        print(df.groupby(["dataset", "split_group"]).size())
        total = len(clips)
        n_pos = int((df["split_group"] == "emergency").sum())
        print(f"Emergency ratio: {n_pos}/{total} = {n_pos / total:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dev_phase for emergency audio event detection"
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default="dataset_source",
        help="Folder containing esc50/, urbansound8k/, fsd50k/. Default: dataset_source",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Delete existing dev_phase first"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fsd-max-labels",
        type=int,
        default=1,
        help="Keep only FSD50K clips with at most this many labels. 1 = single-label only.",
    )
    parser.add_argument(
        "--fsd-emergency-cap",
        type=int,
        default=1800,
        help="Max number of FSD50K emergency clips after filtering. Use 0 or negative to keep all.",
    )
    args = parser.parse_args()

    data_root = Path(args.repo_root).resolve()
    project_root = data_root.parent if data_root.name == "dataset_source" else data_root

    phase_root = project_root / PHASE
    input_root = phase_root / "input_data"
    reference_root = phase_root / "reference_data"

    if args.clean and phase_root.exists():
        shutil.rmtree(phase_root)

    esc_root = data_root / "esc50"
    urban_root = data_root / "urbansound8k"
    fsd_root = data_root / "fsd50k"

    all_clips: list[SourceClip] = []

    if esc_root.exists():
        print(f"Loading ESC-50 from {esc_root}")
        all_clips.extend(load_esc50(esc_root))
    else:
        print(f"ESC-50 folder not found: {esc_root}")

    if urban_root.exists():
        print(f"Loading UrbanSound8K from {urban_root}")
        all_clips.extend(load_urbansound8k(urban_root))
    else:
        print(f"UrbanSound8K folder not found: {urban_root}")

    fsd_clips: list[SourceClip] = []
    if fsd_root.exists():
        print(f"Loading FSD50K emergency-only from {fsd_root}")
        fsd_clips = load_fsd50k_emergency_only(
            fsd_root,
            max_labels_allowed=args.fsd_max_labels,
        )
    else:
        print(f"FSD50K folder not found: {fsd_root}")

    if fsd_clips:
        print_counts("FSD50K after label filtering", fsd_clips)

    cap = None if args.fsd_emergency_cap <= 0 else args.fsd_emergency_cap
    fsd_clips = cap_fsd50k_emergency(fsd_clips, cap=cap, seed=args.seed)

    if fsd_clips:
        print_counts("FSD50K after optional cap", fsd_clips)

    all_clips.extend(fsd_clips)

    if not all_clips:
        raise RuntimeError(
            "No usable clips found. Check dataset folders and label mappings."
        )

    print_counts("All usable clips", all_clips)

    split_map = stratified_split(all_clips, seed=args.seed)
    print_split_distribution(split_map)

    print("\nNatural split sizes:")
    for split_name, clips in split_map.items():
        print(f"{split_name}: {len(clips)}")

    for split_name, clips in split_map.items():
        process_split(
            split_name=split_name,
            clips=clips,
            input_root=input_root,
            reference_root=reference_root,
        )

    write_summary(all_clips, phase_root / "source_summary.csv")
    print(f"\nDone. Dataset written to: {phase_root}")


if __name__ == "__main__":
    main()
