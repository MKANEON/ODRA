import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from EEGTool import EEGDataLoader
from ODRA import ODRA


def main():
    parser = argparse.ArgumentParser(description="Minimal single-subject 5-CV example for ODRA.")
    parser.add_argument("--x", required=True, help="Path to X.npy: (samples, channels, time).")
    parser.add_argument("--y", required=True, help="Path to y.npy: (samples,).")
    parser.add_argument("--output", default="odra_5cv_summary.json")
    args = parser.parse_args()

    X = np.asarray(np.load(args.x))
    y = np.asarray(np.load(args.y)).reshape(-1)
    if X.ndim != 3 or len(X) != len(y):
        raise ValueError("X must be 3-D and have the same number of samples as y.")
    if np.min(np.unique(y, return_counts=True)[1]) < 5:
        raise ValueError("Each class needs at least five samples for 5-fold cross-validation.")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        # ODRA expects explicitly standardized inputs; no test statistics are reused.
        X_train = EEGDataLoader.scale_data(X[train_idx])
        X_test = EEGDataLoader.scale_data(X[test_idx])
        model = ODRA(num_channels=X.shape[1], num_classes=len(np.unique(y)))
        model.fit(X_train, y[train_idx], verbose=False)
        scores = model.score(X_test, y[test_idx])
        result = {
            "fold": fold,
            "balanced_accuracy": float(scores["balanced_accuracy"]),
            "recall": [float(value) for value in scores["recall-per-class"]],
        }
        folds.append(result)
        print(f"Fold {fold}: BA={result['balanced_accuracy']:.4f}")

    ba = np.asarray([fold["balanced_accuracy"] for fold in folds])
    recalls = np.asarray([fold["recall"] for fold in folds])
    summary = {
        "n_splits": 5,
        "folds": folds,
        "mean_balanced_accuracy": float(ba.mean()),
        "std_balanced_accuracy": float(ba.std()),
        "mean_recall": [float(value) for value in recalls.mean(axis=0)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
