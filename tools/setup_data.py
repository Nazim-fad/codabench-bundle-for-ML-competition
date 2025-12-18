# Script to download the data from a given source and create the splits
# This is a mock version that generate fake problems
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

PHASE = 'dev_phase'

DATA_DIR = Path(PHASE) / 'input_data'
REF_DIR = Path(PHASE) / 'reference_data'


def make_csv(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(filepath, index=False)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Load or generate data for the benchmark'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data generation')
    args = parser.parse_args()

    # Generate and split the data
    rng = np.random.RandomState(args.seed)
    X, y = make_classification(n_samples=500, n_features=5, random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=rng
    )
    X_test, X_private_test, y_test, y_private_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=rng
    )

    # Store the data in the correct folders:
    # - input_data contains train data (both features and labels) and only
    #   test features so the test labels are kept secret
    # - reference_data contains the test labels for scoring
    for split, X_split, y_split in [
        ('train', X_train, y_train),
        ('test', X_test, y_test),
        ('private_test', X_private_test, y_private_test),
    ]:
        split_dir = DATA_DIR / split
        make_csv(X_split, split_dir / f'{split}_features.csv')
        label_dir = split_dir if split == "train" else REF_DIR
        make_csv(y_split, label_dir / f'{split}_labels.csv')