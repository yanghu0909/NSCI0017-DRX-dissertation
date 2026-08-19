from pathlib import Path

import pandas as pd
from CBFV import composition
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
METHOD_CATEGORIES = [
    "Hydrothermal",
    "Mechanochemical",
    "Molten-salt",
    "Sol-gel",
    "Solid-state",
]


def make_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    condition_df = pd.DataFrame(index=df.index)
    numeric_method = pd.to_numeric(df["Method"], errors="coerce")
    if (~numeric_method.isin(range(len(METHOD_CATEGORIES)))).any():
        raise ValueError("Method must contain category codes 0-4.")
    condition_df["Method"] = numeric_method.astype(int)

    temp_col = next(c for c in df.columns if c.startswith("T ("))
    condition_df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    condition_df["t (h)"] = pd.to_numeric(df["t (h)"], errors="coerce")
    return condition_df.reset_index(drop=True)


def main() -> None:
    X_train_filtered = pd.read_csv(ROOT / "X_train_filtered.csv")
    X_val_filtered = pd.read_csv(ROOT / "X_val_filtered.csv")
    X_test_filtered = pd.read_csv(ROOT / "X_test_filtered.csv")

    for X in [X_train_filtered, X_val_filtered, X_test_filtered]:
        if "Unnamed: 0" in X.columns:
            X.drop(columns=["Unnamed: 0"], inplace=True)

    df_train = pd.read_csv(ROOT / "data_splits" / "drx_train.csv")
    df_val = pd.read_csv(ROOT / "data_splits" / "drx_val.csv")
    df_test = pd.read_csv(ROOT / "data_splits" / "drx_test.csv")

    y_train = df_train["DRX label"].reset_index(drop=True)
    y_val = df_val["DRX label"].reset_index(drop=True)

    condition_train = make_condition_features(df_train)
    condition_val = make_condition_features(df_val)
    condition_test = make_condition_features(df_test)

    X_train_filtered = pd.concat([X_train_filtered.reset_index(drop=True), condition_train], axis=1)
    X_val_filtered = pd.concat([X_val_filtered.reset_index(drop=True), condition_val], axis=1)
    X_test_filtered = pd.concat([X_test_filtered.reset_index(drop=True), condition_test], axis=1)

    X_val_filtered = X_val_filtered.reindex(columns=X_train_filtered.columns, fill_value=0)
    X_test_filtered = X_test_filtered.reindex(columns=X_train_filtered.columns, fill_value=0)

    train_medians = X_train_filtered.median(numeric_only=True)
    X_train_filtered = X_train_filtered.fillna(train_medians)
    X_val_filtered = X_val_filtered.fillna(train_medians)
    X_test_filtered = X_test_filtered.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_filtered),
        columns=X_train_filtered.columns,
    )
    best_model = ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    best_model.fit(X_train_scaled, y_train)
    print("Using freshly trained Extra Trees model.")

    experimental_samples = pd.DataFrame({
        "Sample": list(range(1, 13)),
        "Composition": [
            "Li1.20Ni0.50Mn0.10Ti0.20O2",
            "Li1.20Ni0.40Mn0.20Ti0.20O2",
            "Li1.20Ni0.30Mn0.30Ti0.20O2",
            "Li1.20Ni0.20Mn0.40Ti0.20O2",
            "Li1.20Ni0.10Mn0.50Ti0.20O2",
            "Li1.20Ni0.40Mn0.10Ti0.30O2",
            "Li1.20Ni0.30Mn0.20Ti0.30O2",
            "Li1.20Ni0.20Mn0.30Ti0.30O2",
            "Li1.20Ni0.10Mn0.40Ti0.30O2",
            "Li1.20Ni0.30Mn0.10Ti0.40O2",
            "Li1.20Ni0.20Mn0.20Ti0.40O2",
            "Li1.20Ni0.10Mn0.30Ti0.40O2",
        ],
    })

    cbfv_experimental = pd.DataFrame({
        "formula": experimental_samples["Composition"],
        "target": 0,
    })
    X_exp, _, _, skipped = composition.generate_features(cbfv_experimental)
    if skipped:
        print("Skipped formulae:", skipped)

    X_exp_filtered = X_exp.reindex(columns=X_train_filtered.columns, fill_value=0)

    method_value = "Sol-gel"
    temperature_value = 900
    time_value = 4

    X_exp_filtered["Method"] = METHOD_CATEGORIES.index(method_value)

    temp_col = next(c for c in X_exp_filtered.columns if c.startswith("T ("))
    X_exp_filtered[temp_col] = temperature_value
    X_exp_filtered["t (h)"] = time_value
    X_exp_filtered = X_exp_filtered.fillna(train_medians)

    X_exp_scaled = scaler.transform(X_exp_filtered)
    exp_pred = best_model.predict(X_exp_scaled)
    exp_proba = best_model.predict_proba(X_exp_scaled)

    class_list = list(best_model.classes_)
    unsuccessful_index = class_list.index(0) if 0 in class_list else class_list.index("unsuccessful")
    successful_index = class_list.index(1) if 1 in class_list else class_list.index("successful")

    prediction_results = experimental_samples.copy()
    prediction_results["Method"] = method_value
    prediction_results["T (°C)"] = temperature_value
    prediction_results["t (h)"] = time_value
    prediction_results["predicted_label"] = exp_pred
    prediction_results["predicted_label_text"] = prediction_results["predicted_label"].map({
        0: "unsuccessful",
        1: "successful",
    }).fillna(prediction_results["predicted_label"].astype(str))
    prediction_results["prob_unsuccessful"] = exp_proba[:, unsuccessful_index]
    prediction_results["prob_successful"] = exp_proba[:, successful_index]

    prediction_results = prediction_results.sort_values("Sample").reset_index(drop=True)

    out = ROOT / "predictions_for_12_experimental_samples_extra_trees_with_conditions.csv"
    prediction_results.to_csv(out, index=False)
    print("Saved:", out)
    print(prediction_results)


if __name__ == "__main__":
    main()
