from pathlib import Path
import pandas as pd
from collections import Counter

ROOT = Path("dataset_source")


def inspect_esc50():
    print("\n================ ESC50 ================")

    meta_file = ROOT / "esc50" / "meta" / "esc50.csv"

    df = pd.read_csv(meta_file)

    counts = df["category"].value_counts()

    for label, count in counts.items():
        print(f"{label:25s} {count}")


def inspect_urbansound():
    print("\n============ UrbanSound8K =============")

    meta_file = ROOT / "urbansound8k" / "metadata" / "UrbanSound8K.csv"

    df = pd.read_csv(meta_file)

    counts = df["class"].value_counts()

    for label, count in counts.items():
        print(f"{label:25s} {count}")


def inspect_fsd50k():
    print("\n=============== FSD50K ================")

    gt_folder = ROOT / "fsd50k" / "FSD50K.ground_truth"

    csv_files = list(gt_folder.rglob("*.csv"))

    if not csv_files:
        print("No CSV file found in FSD50K ground truth")
        return

    df = pd.read_csv(csv_files[0])

    label_counter = Counter()

    for labels in df["labels"]:
        for l in str(labels).split(","):
            label_counter[l.strip()] += 1

    for label, count in sorted(label_counter.items()):
        print(f"{label:25s} {count}")


def main():
    inspect_esc50()
    inspect_urbansound()
    inspect_fsd50k()


if __name__ == "__main__":
    main()
