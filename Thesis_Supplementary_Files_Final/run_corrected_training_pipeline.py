from collections import OrderedDict
from pathlib import Path
from time import time
import json

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
METHOD_CATEGORIES = [
    "Hydrothermal",
    "Mechanochemical",
    "Molten-salt",
    "Sol-gel",
    "Solid-state",
]
METHOD_CODE_TO_NAME = dict(enumerate(METHOD_CATEGORIES))


def make_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    method_numeric = pd.to_numeric(df["Method"], errors="coerce")
    invalid = ~method_numeric.isin(METHOD_CODE_TO_NAME)
    if invalid.any():
        raise ValueError(
            f"Invalid Method codes: {df.loc[invalid, 'Method'].tolist()}"
        )

    temp_col = next(c for c in df.columns if c.startswith("T ("))
    condition_df = pd.DataFrame(index=df.index)
    condition_df["Method"] = method_numeric.astype(int)
    condition_df["T (°C)"] = pd.to_numeric(df[temp_col], errors="coerce")
    condition_df["t (h)"] = pd.to_numeric(df["t (h)"], errors="coerce")
    return condition_df.reset_index(drop=True)


def load_inputs():
    feature_sets = {}
    split_sets = {}
    for split in ("train", "val", "test"):
        feature_sets[split] = pd.read_csv(ROOT / f"X_{split}_filtered.csv")
        split_sets[split] = pd.read_csv(ROOT / "data_splits" / f"drx_{split}.csv")

    conditions = {
        split: make_condition_features(split_sets[split])
        for split in ("train", "val", "test")
    }
    inputs = {
        split: pd.concat(
            [
                feature_sets[split].reset_index(drop=True),
                conditions[split].reset_index(drop=True),
            ],
            axis=1,
        )
        for split in ("train", "val", "test")
    }
    for split in ("val", "test"):
        inputs[split] = inputs[split].reindex(columns=inputs["train"].columns, fill_value=0)

    medians = inputs["train"].median(numeric_only=True)
    inputs = {split: frame.fillna(medians) for split, frame in inputs.items()}
    labels = {
        split: split_sets[split]["DRX label"].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    return inputs, labels, split_sets


def metric_row(name, model, fit_time, y_train, pred_train, y_val, pred_val):
    return {
        "model_name": name,
        "fit_time": fit_time,
        "accuracy_train": accuracy_score(y_train, pred_train),
        "accuracy_val": accuracy_score(y_val, pred_val),
        "balanced_accuracy_train": balanced_accuracy_score(y_train, pred_train),
        "balanced_accuracy_val": balanced_accuracy_score(y_val, pred_val),
        "f1_macro_train": f1_score(y_train, pred_train, average="macro", zero_division=0),
        "f1_macro_val": f1_score(y_val, pred_val, average="macro", zero_division=0),
        "precision_macro_val": precision_score(
            y_val, pred_val, average="macro", zero_division=0
        ),
        "recall_macro_val": recall_score(
            y_val, pred_val, average="macro", zero_division=0
        ),
    }


def save_evaluation(prefix, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual unsuccessful", "Actual successful"],
        columns=["Predicted unsuccessful", "Predicted successful"],
    )
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["unsuccessful", "successful"],
            zero_division=0,
            output_dict=True,
        )
    ).T
    cm_df.to_csv(ROOT / f"confusion_matrix_{prefix}.csv")
    report_df.to_csv(ROOT / f"classification_report_{prefix}.csv")
    return cm_df, report_df


def main():
    inputs, labels, split_sets = load_inputs()
    scaler = StandardScaler()
    scaled = {
        "train": scaler.fit_transform(inputs["train"]),
        "val": scaler.transform(inputs["val"]),
        "test": scaler.transform(inputs["test"]),
    }

    models = OrderedDict(
        {
            "dummy": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
            "logreg": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
            ),
            "rf": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED
            ),
            "gb": GradientBoostingClassifier(random_state=RANDOM_SEED),
            "ada": AdaBoostClassifier(random_state=RANDOM_SEED),
            "knn": KNeighborsClassifier(),
            "svc": SVC(class_weight="balanced", random_state=RANDOM_SEED),
            "linear_svc": LinearSVC(
                class_weight="balanced", max_iter=5000, random_state=RANDOM_SEED
            ),
        }
    )

    results = []
    trained = {}
    for name, model in models.items():
        start = time()
        model.fit(scaled["train"], labels["train"])
        elapsed = time() - start
        pred_train = model.predict(scaled["train"])
        pred_val = model.predict(scaled["val"])
        results.append(
            metric_row(
                name,
                model,
                elapsed,
                labels["train"],
                pred_train,
                labels["val"],
                pred_val,
            )
        )
        trained[name] = model

    comparison = pd.DataFrame(results).sort_values(
        ["f1_macro_val", "balanced_accuracy_val", "accuracy_val"],
        ascending=False,
    ).reset_index(drop=True)
    comparison.to_csv(ROOT / "model_comparison_classic_models.csv", index=False)

    best_name = comparison.loc[0, "model_name"]
    best_model = trained[best_name]
    predictions = {
        split: best_model.predict(scaled[split])
        for split in ("train", "val", "test")
    }
    validation_cm, _ = save_evaluation(
        f"{best_name}_validation", labels["val"], predictions["val"]
    )
    test_cm, _ = save_evaluation(
        f"{best_name}_test", labels["test"], predictions["test"]
    )

    for split in ("val", "test"):
        output = split_sets[split].copy()
        output["actual_label"] = labels[split]
        output["predicted_label"] = predictions[split]
        output["correct_prediction"] = labels[split] == predictions[split]
        output.to_csv(ROOT / f"{split}_predictions_{best_name}.csv", index=False)

    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": inputs["train"].columns,
                "importance": best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False, ignore_index=True)
        importance.to_csv(ROOT / f"feature_importance_{best_name}.csv", index=False)

    summary = {
        "best_model": best_name,
        "input_columns": len(inputs["train"].columns),
        "atmosphere_in_inputs": "Atmosphere" in inputs["train"].columns,
        "method_encoding": "single integer category code (0-4)",
        "method_counts_train": {
            METHOD_CODE_TO_NAME[code]: int((inputs["train"]["Method"] == code).sum())
            for code in METHOD_CODE_TO_NAME
        },
        "train": {
            "accuracy": accuracy_score(labels["train"], predictions["train"]),
            "balanced_accuracy": balanced_accuracy_score(
                labels["train"], predictions["train"]
            ),
            "macro_f1": f1_score(
                labels["train"], predictions["train"], average="macro", zero_division=0
            ),
        },
        "validation": {
            "accuracy": accuracy_score(labels["val"], predictions["val"]),
            "balanced_accuracy": balanced_accuracy_score(
                labels["val"], predictions["val"]
            ),
            "macro_f1": f1_score(
                labels["val"], predictions["val"], average="macro", zero_division=0
            ),
            "confusion_matrix": validation_cm.to_numpy().tolist(),
        },
        "test": {
            "accuracy": accuracy_score(labels["test"], predictions["test"]),
            "balanced_accuracy": balanced_accuracy_score(
                labels["test"], predictions["test"]
            ),
            "macro_f1": f1_score(
                labels["test"], predictions["test"], average="macro", zero_division=0
            ),
            "confusion_matrix": test_cm.to_numpy().tolist(),
        },
        "model_parameters": best_model.get_params(),
    }
    (ROOT / "best_model_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
