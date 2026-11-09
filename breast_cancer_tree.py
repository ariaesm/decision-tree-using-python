"""Decision tree classifier for the Breast Cancer Wisconsin (Diagnostic) dataset.

Predicts whether a breast mass is malignant (M) or benign (B) from 30
features computed from a digitized image of a fine-needle aspirate.

Steps:
1. load and clean the CSV,
2. train an entropy-based decision tree on a stratified split,
3. report accuracy, precision/recall and the confusion matrix,
4. pick max_depth with 5-fold cross-validation and plot the curve,
5. save a picture of the tree and the most important features.

Run:  python breast_cancer_tree.py   (figures are written to output/)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data_breast_cancer_wisconsin.csv"
OUT = HERE / "output"
SEED = 42


def load_data(path: Path = DATA) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and a 0/1 target (1 = malignant)."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == "id"])
    y = (df.pop("diagnosis") == "M").astype(int).rename("malignant")
    return df, y


def depth_curve(X: pd.DataFrame, y: pd.Series, depths: range) -> pd.DataFrame:
    """Mean and std of 5-fold CV accuracy for each max_depth."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for d in depths:
        model = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=SEED)
        scores = cross_val_score(model, X, y, cv=cv)
        rows.append({"max_depth": d, "mean": scores.mean(), "std": scores.std()})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    X, y = load_data()
    print(f"{len(X)} samples, {X.shape[1]} features, {y.mean():.1%} malignant\n")

    curve = depth_curve(X, y, range(1, 13))
    best = int(curve.loc[curve["mean"].idxmax(), "max_depth"])
    print(curve.round(3).to_string(index=False))
    print(f"\nBest max_depth by cross-validation: {best}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    model = DecisionTreeClassifier(criterion="entropy", max_depth=best, random_state=SEED)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    print(classification_report(y_test, pred, target_names=["benign", "malignant"]))
    print("Confusion matrix [rows = true, cols = predicted]:")
    print(confusion_matrix(y_test, pred))

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    print("\nTop features:\n" + importance.tail(5)[::-1].round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(curve["max_depth"], curve["mean"], yerr=curve["std"], marker="o", capsize=3)
    ax.axvline(best, ls="--", c="gray")
    ax.set(xlabel="max_depth", ylabel="5-fold CV accuracy", title="Choosing the tree depth")
    fig.tight_layout()
    fig.savefig(OUT / "breast_cancer_depth_curve.png", dpi=120)

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        model,
        feature_names=list(X.columns),
        class_names=["benign", "malignant"],
        filled=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    fig.savefig(OUT / "breast_cancer_tree.png", dpi=120, bbox_inches="tight")
    print(f"\nFigures saved to {OUT}/")
    np.set_printoptions()


if __name__ == "__main__":
    main()
