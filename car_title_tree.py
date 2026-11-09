"""Decision tree on the US Cars dataset: is a listed car a salvage / insurance-loss vehicle?

Target: ``title_status`` == "salvage insurance" (vs "clean vehicle").
Features: price, mileage, year, brand, colour, state and days left on the auction.

Run:  python car_title_tree.py   (figures are written to output/)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data_usa_cars.csv"
OUT = HERE / "output"
SEED = 42


def days_left(text: str) -> float:
    """Convert '10 days left' / '22 hours left' / 'Listing Expired' to days."""
    parts = str(text).split()
    if len(parts) >= 2 and parts[0].isdigit():
        n = int(parts[0])
        return n / 24 if parts[1].startswith("hour") else float(n)
    return 0.0


def load_data(path: Path = DATA) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, index_col=0)
    df = df[df["price"] > 0].copy()  # a few listings have no price
    y = (df["title_status"] != "clean vehicle").astype(int).rename("salvage")
    df["days_left"] = df["condition"].map(days_left)
    df["color"] = df["color"].where(~df["color"].isin(["no_color", "color:"]), "unknown")
    for col in ("brand", "color", "state"):  # keep the most common levels, bucket the rest
        top = df[col].value_counts().index[:15]
        df[col] = df[col].where(df[col].isin(top), "other")
    X = pd.get_dummies(
        df[["price", "mileage", "year", "days_left", "brand", "color", "state"]],
        columns=["brand", "color", "state"],
        dtype=int,
    )
    return X, y


def main() -> None:
    OUT.mkdir(exist_ok=True)
    X, y = load_data()
    print(f"{len(X)} listings, {y.mean():.1%} salvage titles\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    # salvage titles are rare, so weight the classes instead of chasing raw accuracy
    model = DecisionTreeClassifier(
        criterion="entropy", max_depth=5, class_weight="balanced", random_state=SEED
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=["clean", "salvage"]))

    importance = pd.Series(model.feature_importances_, index=X.columns).nlargest(8)
    print("Top features:\n" + importance.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, group in pd.concat([X[["price", "mileage"]], y], axis=1).groupby("salvage"):
        name = "salvage" if label else "clean"
        axes[0].scatter(group["mileage"], group["price"], s=6, alpha=0.5, label=name)
    axes[0].set(xlabel="mileage", ylabel="price ($)", title="Price vs mileage")
    axes[0].legend()
    importance[::-1].plot.barh(ax=axes[1], title="Feature importance")
    fig.tight_layout()
    fig.savefig(OUT / "cars_overview.png", dpi=120)
    print(f"\nFigure saved to {OUT}/")


if __name__ == "__main__":
    main()
