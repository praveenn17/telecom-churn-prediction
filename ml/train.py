"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        Telecom Customer Churn Prediction — ML Training Pipeline             ║
║  Models : LogisticRegression | DecisionTree | RandomForest | XGBoost        ║
║  Output : ml/model.pkl  •  ml/feature_schema.json                           ║
║           ml/shap_summary.png  •  ml/feature_importance.json                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
    python ml/train.py

Prerequisites
-------------
    pip install -r ml/requirements.txt
    Place WA_Fn-UseC_-Telco-Customer-Churn.csv inside ml/data/
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import json
import os
import sys
import warnings
from pathlib import Path
from textwrap import dedent

# Force UTF-8 output so box-drawing / arrow characters render correctly on
# Windows terminals that default to cp1252 (e.g. PowerShell without chcp 65001)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── third-party ────────────────────────────────────────────────────────────────
import joblib
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).resolve().parent        # ml/
DATA_PATH  = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH      = BASE_DIR / "model.pkl"
SCHEMA_PATH     = BASE_DIR / "feature_schema.json"
SHAP_PLOT_PATH  = BASE_DIR / "shap_summary.png"
FEAT_IMP_PATH   = BASE_DIR / "feature_importance.json"

SHAP_SAMPLE_SIZE = 200   # max rows for SHAP computation

RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    """Print a section banner to stdout."""
    width = 72
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║  {title:<{width - 2}}║")
    print(f"╚{border}╝")


def _step(msg: str) -> None:
    print(f"  ▶  {msg}")


def _ok(msg: str) -> None:
    print(f"  ✔  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    _banner("STAGE 1 — DATA LOADING")
    _step(f"Reading CSV from: {path}")

    if not path.exists():
        print(
            f"\n  ✖  ERROR: Dataset not found at:\n     {path}\n"
            "     Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in ml/data/\n"
        )
        sys.exit(1)

    df = pd.read_csv(path)
    _ok(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    _step(f"Columns: {df.columns.tolist()}")
    _step(f"Target distribution (raw):\n{df['Churn'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING  (pandas-level — before sklearn Pipeline)
# ══════════════════════════════════════════════════════════════════════════════

# Columns that require StandardScaler
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def preprocess_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    _banner("STAGE 2 — RAW PREPROCESSING")

    # ── 2a. Drop customerID (non-informative identifier) ──────────────────────
    _step("Dropping 'customerID' column")
    df = df.drop(columns=["customerID"])

    # ── 2b. TotalCharges → numeric; blanks → NaN ──────────────────────────────
    _step("Converting 'TotalCharges' to numeric (blanks → NaN)")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blanks = df["TotalCharges"].isna().sum()
    _step(f"  Found {n_blanks} blank TotalCharges rows")

    if n_blanks > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        _ok(f"  Imputed {n_blanks} blanks with median = {median_val:.2f}")

    # ── 2c. Encode target: Yes→1 / No→0 ──────────────────────────────────────
    _step("Encoding target 'Churn': Yes → 1,  No → 0")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    _ok(f"  Class counts  →  {dict(df['Churn'].value_counts().sort_index())}")

    # ── 2d. Separate features / target ───────────────────────────────────────
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    _ok(f"Feature matrix shape: {X.shape}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE SCHEMA  (for the inference API)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_schema(X: pd.DataFrame) -> dict:
    """
    Capture the raw column names, dtypes, and — for categoricals — the
    unique category values the training data contained.
    The inference API uses this to validate and align incoming requests.
    """
    _banner("STAGE 3 — FEATURE SCHEMA")

    # Identify categorical columns = all non-numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    schema: dict = {
        "feature_names": X.columns.tolist(),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "categories": {
            col: sorted(X[col].dropna().unique().tolist())
            for col in cat_cols
        },
        "dtypes": X.dtypes.astype(str).to_dict(),
        "scaled_numeric_cols": NUMERIC_COLS,
    }

    SCHEMA_PATH.write_text(json.dumps(schema, indent=2, default=str))
    _ok(f"Feature schema saved → {SCHEMA_PATH}")
    _step(f"  Numeric features  ({len(num_cols)}): {num_cols}")
    _step(f"  Categorical features ({len(cat_cols)}): {cat_cols}")
    return schema


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SKLEARN PIPELINE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(model, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """
    Build a full sklearn Pipeline:
        ColumnTransformer
            ├─ numeric  → StandardScaler
            └─ categorical → OneHotEncoder (drop_first via drop='first')
        └─ estimator (any sklearn-compatible model)
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        (
            "onehot",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_transformer,    num_cols),
            ("cat",  categorical_transformer, cat_cols),
        ],
        remainder="drop",   # safely discard any unexpected column
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        model),
    ])
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_models(scale_pos_weight: float) -> dict[str, object]:
    """
    Return four classifiers.
    All models that support class_weight receive 'balanced'.
    XGBoost uses scale_pos_weight = (n_negatives / n_positives).
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=10,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred  = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred),   4),
        "Precision": round(precision_score(y_test, y_pred),  4),
        "Recall":    round(recall_score(y_test, y_pred),     4),
        "F1":        round(f1_score(y_test, y_pred),         4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba),   4),
    }


def print_confusion_matrix(model_pipeline: Pipeline,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            model_name: str) -> None:
    _banner(f"CONFUSION MATRIX — Best Model: {model_name}")
    y_pred = model_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    table = [
        ["",             "Predicted: No (0)", "Predicted: Yes (1)"],
        ["Actual: No (0)",  f"TN = {tn}",       f"FP = {fp}"],
        ["Actual: Yes (1)", f"FN = {fn}",       f"TP = {tp}"],
    ]
    print(tabulate(table, tablefmt="rounded_outline"))
    print(f"\n  Recall (sensitivity): {tp / (tp + fn):.4f}  ←  fraction of churners caught")
    print(f"  Specificity:          {tn / (tn + fp):.4f}  ←  fraction of non-churners correct")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

# Tree-based model class names that support TreeExplainer
_TREE_MODELS = {"DecisionTree", "RandomForest", "XGBoost"}


def run_shap_analysis(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    best_name: str,
) -> None:
    """
    Compute SHAP values for the fitted pipeline on a sample of the test set.

    Strategy
    --------
    • Extract the preprocessed array from the pipeline's ColumnTransformer so
      SHAP operates on the same feature space the model was trained on.
    • Recover human-readable encoded feature names from the ColumnTransformer.
    • Use TreeExplainer for tree-based models (fast, exact).
    • Use LinearExplainer for LogisticRegression (linear kernel SHAP).
    • Save a bar-style summary plot to ml/shap_summary.png.
    • Save the top-10 features by mean |SHAP| to ml/feature_importance.json.
    """
    _banner("STAGE 11 — SHAP EXPLAINABILITY")

    # ── 11a. Sample test rows ─────────────────────────────────────────────────
    n_sample = min(SHAP_SAMPLE_SIZE, len(X_test))
    _step(f"Sampling {n_sample} rows from test set for SHAP computation")
    X_sample = X_test.sample(n=n_sample, random_state=RANDOM_STATE).reset_index(drop=True)

    # ── 11b. Transform via the pipeline preprocessor ──────────────────────────
    _step("Transforming sample through pipeline preprocessor …")
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_sample)     # numpy array

    # ── 11c. Recover encoded feature names ───────────────────────────────────
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean sklearn prefix: 'num__tenure' → 'tenure', 'cat__gender_Male' → 'gender_Male'
        feature_names = [
            n.split("__", 1)[-1] for n in feature_names
        ]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    _ok(f"Encoded feature space: {len(feature_names)} columns")

    # ── 11d. Build SHAP explainer ─────────────────────────────────────────────
    model = pipeline.named_steps["model"]
    is_tree = best_name in _TREE_MODELS

    if is_tree:
        _step(f"{best_name} is tree-based → using shap.TreeExplainer")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        # For binary classifiers some versions return list[array]; take class-1 slice
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        _step(f"{best_name} is linear → using shap.LinearExplainer")
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    _ok(f"SHAP values computed — shape: {shap_values.shape}")

    # ── 11e. Mean absolute SHAP per feature → top-10 importance ──────────────
    _step("Computing mean |SHAP| per feature …")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_series = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    top10 = importance_series.head(10)

    top10_dict = [
        {"rank": int(i + 1), "feature": feat, "mean_abs_shap": round(float(val), 6)}
        for i, (feat, val) in enumerate(top10.items())
    ]
    FEAT_IMP_PATH.write_text(json.dumps(top10_dict, indent=2))
    _ok(f"Top-10 feature importances saved → {FEAT_IMP_PATH}")

    print()
    print(tabulate(
        [[r["rank"], r["feature"], f"{r['mean_abs_shap']:.6f}"] for r in top10_dict],
        headers=["Rank", "Feature", "Mean |SHAP|"],
        tablefmt="rounded_outline",
    ))

    # ── 11f. SHAP summary bar plot ────────────────────────────────────────────
    _step("Generating SHAP summary bar plot …")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title(
        f"SHAP Feature Importance — {best_name}\n"
        f"(mean |SHAP value| over {n_sample} test samples)",
        fontsize=13,
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    _ok(f"SHAP summary plot saved → {SHAP_PLOT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Stage 2: Raw preprocessing ────────────────────────────────────────────
    X, y = preprocess_raw(df)

    # ── Stage 3: Feature schema ───────────────────────────────────────────────
    schema = build_feature_schema(X)

    cat_cols = schema["categorical_features"]
    num_cols = schema["numeric_features"]

    # ── Stage 4: Train / test split ───────────────────────────────────────────
    _banner("STAGE 4 — TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    _ok(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows  (80/20 stratified)")

    # ── Stage 5: Compute class imbalance ratio for XGBoost ───────────────────
    _banner("STAGE 5 — CLASS IMBALANCE")
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = round(n_neg / n_pos, 4)
    _step(f"Train set  →  No-churn: {n_neg:,}  |  Churn: {n_pos:,}")
    _ok(f"scale_pos_weight for XGBoost = {scale_pos_weight}  (n_neg / n_pos)")
    _step("All other models use class_weight='balanced'")

    # ── Stage 6: Train all models ─────────────────────────────────────────────
    _banner("STAGE 6 — MODEL TRAINING")
    models   = get_models(scale_pos_weight)
    results  = {}
    pipelines = {}

    for name, estimator in models.items():
        _step(f"Training {name} ...")
        pipe = build_pipeline(estimator, cat_cols, num_cols)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name]   = metrics
        pipelines[name] = pipe
        _ok(f"{name} done  —  F1={metrics['F1']:.4f}  |  ROC-AUC={metrics['ROC-AUC']:.4f}")

    # ── Stage 7: Comparison table ─────────────────────────────────────────────
    _banner("STAGE 7 — MODEL COMPARISON")
    table_rows = [
        [name] + list(metrics.values())
        for name, metrics in results.items()
    ]
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline", floatfmt=".4f"))

    # ── Stage 8: Select best model by F1 ─────────────────────────────────────
    _banner("STAGE 8 — BEST MODEL SELECTION  (by F1 score)")
    best_name = max(results, key=lambda m: results[m]["F1"])
    best_metrics = results[best_name]
    _ok(f"Best model: {best_name}  (F1 = {best_metrics['F1']:.4f})")

    # Print confusion matrix for best model
    print_confusion_matrix(pipelines[best_name], X_test, y_test, best_name)

    # ── Stage 9: Refit best model on FULL training data ───────────────────────
    _banner("STAGE 9 — REFIT ON FULL TRAINING SET")
    _step(f"Refitting {best_name} pipeline on all {len(X_train):,} training rows …")
    best_estimator = get_models(scale_pos_weight)[best_name]
    final_pipeline = build_pipeline(best_estimator, cat_cols, num_cols)
    final_pipeline.fit(X_train, y_train)
    _ok("Refit complete")

    # ── Stage 10: Save model ──────────────────────────────────────────────────
    _banner("STAGE 10 — SAVING MODEL")
    _step(f"Saving pipeline to: {MODEL_PATH}")
    joblib.dump(final_pipeline, MODEL_PATH)
    size_kb = MODEL_PATH.stat().st_size / 1024
    _ok(f"model.pkl saved  ({size_kb:.1f} KB)")
    _ok(f"feature_schema.json saved  ({SCHEMA_PATH.stat().st_size / 1024:.1f} KB)")

    # ── Stage 11: SHAP explainability ─────────────────────────────────────────
    run_shap_analysis(final_pipeline, X_test, best_name)

    # ── Stage 12: Final summary ───────────────────────────────────────────────
    _banner("STAGE 12 — FINAL SUMMARY")
    summary = dedent(f"""
        ┌─ Selected Model ─────────────────────────────────────────────────┐
        │  Name      : {best_name:<52} │
        │  Selection : Highest F1 score on hold-out test set              │
        │              (F1 is the right metric for imbalanced churn data) │
        ├─ Metrics on Test Set (20 % hold-out) ───────────────────────────┤
        │  Accuracy  : {best_metrics['Accuracy']:<52} │
        │  Precision : {best_metrics['Precision']:<52} │
        │  Recall    : {best_metrics['Recall']:<52} │
        │  F1 Score  : {best_metrics['F1']:<52} │
        │  ROC-AUC   : {best_metrics['ROC-AUC']:<52} │
        ├─ Artifacts ─────────────────────────────────────────────────────┤
        │  ml/model.pkl             ← Pipeline (preprocessing + model)   │
        │  ml/feature_schema.json   ← Raw feature names, dtypes, cats    │
        │  ml/shap_summary.png      ← SHAP bar chart (top 15 features)   │
        │  ml/feature_importance.json ← Top-10 features + mean |SHAP|    │
        └─────────────────────────────────────────────────────────────────┘
    """)
    print(summary)
    print("  Pipeline contents:")
    print(f"    • Preprocessor : ColumnTransformer")
    print(f"        ├─ StandardScaler  → {num_cols}")
    print(f"        └─ OneHotEncoder   → {cat_cols}")
    print(f"    • Estimator    : {best_name}\n")
    print("  ✔  Training pipeline complete!\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
