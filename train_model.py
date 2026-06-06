from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import os
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "Divisi yang paling di minati?"
DROP_FEATURES = {"Nama Lengkap", "Kelas", "HEAD"}
EXPERIENCE_COL = "Pernah ikut organisasi Sebelumnya?"

DEFAULT_DATA_FILES = [
    "datasheet_rekomendasi_divisi_pramuka_500.xlsx",
    "datasheet_rekomendasi_divisi_pramuka_500.csv",
    "kuesionardivisipramuka smaba.xlsx",
    str(Path.home() / "Downloads" / "datasheet_rekomendasi_divisi_pramuka_500.xlsx"),
    str(Path.home() / "Downloads" / "datasheet_rekomendasi_divisi_pramuka_500.csv"),
]

MODEL_ARTIFACT_PATH = Path("model_artifacts.pkl")
LIKERT_MIN = 1
LIKERT_MAX = 5


class PipelineTrainingError(RuntimeError):
    pass


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return _strip_columns(pd.read_excel(data_path))
    if suffix == ".csv":
        return _strip_columns(pd.read_csv(data_path))

    raise ValueError("Format dataset tidak didukung. Gunakan .xlsx atau .csv.")


def _resolve_dataset_path(explicit_path: str | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    env_data_path = os.getenv("PRAMUKA_DATA_PATH")
    if env_data_path:
        candidates.append(env_data_path)

    candidates.extend(DEFAULT_DATA_FILES)

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset tidak ditemukan.\n"
        "Sediakan file: datasheet_rekomendasi_divisi_pramuka_500.xlsx atau .csv di proyek/Downloads.\n"
        "Atau atur PRAMUKA_DATA_PATH di environment/.env."
    )


def _coerce_numeric_like(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sanitize_category_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        .astype("object")
    )


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    df = _strip_columns(df)

    if TARGET_COL not in df.columns:
        raise PipelineTrainingError(f"Kolom target '{TARGET_COL}' tidak ditemukan.")

    df = df.dropna(subset=[TARGET_COL]).copy()
    target = df[TARGET_COL].astype("string").str.strip().astype(str)
    if target.isna().any():
        raise PipelineTrainingError("Terdapat nilai target kosong/invalid.")

    feature_columns = [c for c in df.columns if c not in DROP_FEATURES and c != TARGET_COL]
    x = df[feature_columns].copy()

    if x.shape[1] == 0:
        raise PipelineTrainingError("Tidak ada fitur yang tersisa setelah membuang kolom non-fiturnya.")

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    category_values: Dict[str, list[str]] = {}

    for col in feature_columns:
        if col == EXPERIENCE_COL:
            values = _coerce_numeric_like(x[col])
            invalid = values[~values.between(LIKERT_MIN, LIKERT_MAX)]
            if not invalid.dropna().empty:
                raise PipelineTrainingError(
                    f"Kolom '{col}' harus skala {LIKERT_MIN}-{LIKERT_MAX}, ditemukan: "
                    f"{sorted(set(invalid.dropna().astype(int).astype(str).tolist()))}"
                )
            x[col] = values.round().astype("Int64")
            numeric_columns.append(col)
            continue

        numeric_candidate = _coerce_numeric_like(x[col])
        non_na_count = numeric_candidate.notna().sum()
        # jika seluruh kolom numerik dan masuk skala Likert, anggap numerik
        if non_na_count >= max(10, int(0.25 * len(x))) and (
            numeric_candidate.fillna(LIKERT_MIN - 1).between(LIKERT_MIN, LIKERT_MAX).all()
            and set(numeric_candidate.dropna().astype(int).astype(str).unique()) <= {"1", "2", "3", "4", "5"}
        ):
            x[col] = numeric_candidate
            if x[col].isna().any():
                raise PipelineTrainingError(f"Kolom numerik '{col}' masih mengandung nilai tidak valid.")
            numeric_columns.append(col)
            continue

        # fallback kategorikal
        x[col] = _sanitize_category_series(x[col])
        cats = sorted({str(v) for v in x[col].dropna().astype(str).unique()})
        if len(cats) == 0:
            raise PipelineTrainingError(f"Kolom kategorikal '{col}' kosong setelah pembersihan.")
        x[col] = x[col].fillna(cats[0])
        categorical_columns.append(col)
        category_values[col] = cats

    # validasi ekstra pada kolom numerik (termasuk default skala Likert lain)
    for col in numeric_columns:
        x[col] = _coerce_numeric_like(x[col])
        if x[col].isna().any():
            raise PipelineTrainingError(f"Kolom numerik '{col}' mengandung nilai non-angka.")
        if not x[col].between(LIKERT_MIN, LIKERT_MAX).all():
            raise PipelineTrainingError(f"Kolom numerik '{col}' harus bernilai antara {LIKERT_MIN}-{LIKERT_MAX}.")

    class_counts = target.value_counts().to_dict()
    if len(class_counts) < 2:
        raise PipelineTrainingError("Target tidak memiliki cukup variasi kelas untuk model klasifikasi.")
    if min(class_counts.values()) < 2:
        raise PipelineTrainingError("Setiap kelas target perlu minimal 2 data.")

    schema = {
        "columns": list(df.columns),
        "rows": int(len(df)),
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "category_values": category_values,
        "class_distribution": class_counts,
        "dropped_columns": sorted([c for c in DROP_FEATURES if c in df.columns]),
        "target_column": TARGET_COL,
    }

    target_clean = target.astype(str)
    x = x[feature_columns]
    return x, target_clean, schema


def _build_model_candidates() -> list[tuple[str, Any]]:
    return [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            ),
        ),
    ]


def _build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", StandardScaler(), numeric_columns))

    if categorical_columns:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("categorical", encoder, categorical_columns))

    if not transformers:
        raise PipelineTrainingError("Model tidak punya fitur numerik/kategorikal untuk training.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _safe_float(v: float | int | np.number | np.ndarray) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def evaluate_model_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    candidates: list[tuple[str, Any]],
    random_state: int,
) -> tuple[tuple[str, Pipeline], list[dict[str, Any]]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    best_name = None
    best_score = -np.inf
    best_pipeline: Pipeline | None = None
    rows: list[dict[str, Any]] = []

    for model_name, model in candidates:
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        cv_result = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
            error_score=np.nan,
        )
        result = {
            "model_name": model_name,
            "accuracy_mean": _safe_float(np.nanmean(cv_result["test_accuracy"])),
            "precision_macro_mean": _safe_float(np.nanmean(cv_result["test_precision_macro"])),
            "recall_macro_mean": _safe_float(np.nanmean(cv_result["test_recall_macro"])),
            "f1_macro_mean": _safe_float(np.nanmean(cv_result["test_f1_macro"])),
            "folds": cv.get_n_splits(),
        }
        rows.append(result)

        if np.isfinite(result["f1_macro_mean"]) and result["f1_macro_mean"] > best_score:
            best_score = result["f1_macro_mean"]
            best_name = model_name
            best_pipeline = pipe

    if best_pipeline is None:
        raise PipelineTrainingError("Tidak ada model yang lolos evaluasi CV.")

    return (str(best_name), best_pipeline), rows


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series, class_names: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        "n_train": int((y_true.index.notna().sum())),
        "n_test": int(len(y_true)),
    }


def train_model(dataset_path: str | None = None, seed: int = 42) -> Path:
    path = _resolve_dataset_path(dataset_path)
    df = _load_dataframe(path)

    X, y, schema = _prepare_features(df)
    class_names = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    preprocessor = _build_preprocessor(schema["numeric_columns"], schema["categorical_columns"])
    candidates = _build_model_candidates()
    (best_name, best_pipeline), cv_rows = evaluate_model_candidates(
        X,
        y,
        preprocessor,
        candidates,
        random_state=seed,
    )

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    holdout = _classification_metrics(y_test, pd.Series(y_pred), class_names)

    artifact = {
        "pipeline": best_pipeline,
        "best_model_name": best_name,
        "target_col": TARGET_COL,
        "feature_columns": schema["feature_columns"],
        "numeric_columns": schema["numeric_columns"],
        "categorical_columns": schema["categorical_columns"],
        "class_names": class_names,
        "dataset_schema": {
            "columns": schema["columns"],
            "rows": schema["rows"],
            "feature_columns": schema["feature_columns"],
            "numeric_columns": schema["numeric_columns"],
            "categorical_columns": schema["categorical_columns"],
            "category_values": schema["category_values"],
            "dropped_columns": schema["dropped_columns"],
            "target_column": TARGET_COL,
            "class_distribution": schema["class_distribution"],
            "data_path": str(path),
        },
        "metrics": {
            "cv_comparison": cv_rows,
            "holdout": holdout,
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(MODEL_ARTIFACT_PATH, "wb") as file:
        pickle.dump(artifact, file)

    return MODEL_ARTIFACT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model rekomendasi divisi pramuka")
    parser.add_argument("--data", default=None, help="Path dataset .xlsx/.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    artifact_path = train_model(dataset_path=args.data, seed=args.seed)
    print(f"[OK] Artifact model disimpan ke: {artifact_path}")


if __name__ == "__main__":
    main()
