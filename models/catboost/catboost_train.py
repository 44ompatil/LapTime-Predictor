from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET = "LapTimeSeconds"
CATEGORICAL_FEATURES = [
    "Driver",
    "TeamName",
    "RaceName",
    "Circuit",
    "RaceSession",
    "SeasonYear",
]
CURRENT_LAP_FEATURES = [
    "Sector1TimeSeconds",
    "Sector2TimeSeconds",
    "Sector3TimeSeconds",
]
DATA_PATH = Path(__file__).parents[2] / "data" / "processed" / "featEngineeredData.csv"
MODEL_DIRECTORY = Path(__file__).parent


def report_metrics(name: str, actual: pd.Series, predicted) -> None:
    mae_ms = mean_absolute_error(actual, predicted) * 1000
    rmse_ms = mean_squared_error(actual, predicted) ** 0.5 * 1000
    print(f"{name} samples: {len(actual)}")
    print(f"{name} MAE: {mae_ms:.1f} ms")
    print(f"{name} RMSE: {rmse_ms:.1f} ms")
    print(f"{name} R2: {r2_score(actual, predicted):.4f}")


def main(session: str | None = None, residual: bool = False) -> None:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise SystemExit("CatBoost is not installed. Run: uv sync") from error

    data = pd.read_csv(DATA_PATH)
    data = data[data[TARGET].notna()].copy()
    if session is not None:
        data = data[data["RaceSession"] == session].copy()

    feature_columns = [
        column
        for column in data.columns
        if column not in [TARGET, *CURRENT_LAP_FEATURES]
    ]
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in feature_columns
    ]

    season_year = data["SeasonYear"]
    data[categorical_features] = data[categorical_features].fillna("__missing__").astype(str)

    train = data[season_year <= 2023]
    validation = data[season_year == 2024]
    test = data[season_year == 2025]

    if train.empty or validation.empty or test.empty:
        raise ValueError("Expected training years through 2023, validation year 2024, and test year 2025.")

    model_name = "lap_time_catboost"
    if session is not None:
        model_name = f"lap_time_{session.lower()}_catboost"
    if residual:
        model_name += "_residual"
    model_output = MODEL_DIRECTORY / f"{model_name}.pkl"

    fit_train = train
    fit_validation = validation
    if residual:
        fit_train = train.dropna(subset=["PrevLapTime"])
        fit_validation = validation.dropna(subset=["PrevLapTime"])
        if fit_train.empty or fit_validation.empty:
            raise ValueError("Residual training requires previous-lap values in train and validation data.")

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=8,
        random_strength=1.0,
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100,
    )
    model.fit(
        fit_train[feature_columns],
        fit_train[TARGET] - fit_train["PrevLapTime"] if residual else fit_train[TARGET],
        cat_features=categorical_features,
        eval_set=(
            fit_validation[feature_columns],
            fit_validation[TARGET] - fit_validation["PrevLapTime"]
            if residual
            else fit_validation[TARGET],
        ),
    )

    predictions = model.predict(test[feature_columns])
    if residual:
        residual_base = test["PrevLapTime"].fillna(test["SessionBestLapBefore"])
        residual_base = residual_base.fillna(fit_train[TARGET].median())
        predictions = residual_base.to_numpy() + predictions
    report_metrics("CatBoost test", test[TARGET], predictions)

    baseline = test.dropna(subset=["PrevLapTime"])
    report_metrics("Previous-lap baseline", baseline[TARGET], baseline["PrevLapTime"])

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Saved model to {model_output}")


if __name__ == "__main__":
    main()