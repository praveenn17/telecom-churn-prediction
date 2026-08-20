"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      Telecom Churn Prediction — FastAPI Backend                             ║
║  Endpoints:                                                                  ║
║    GET  /health              → liveness probe                               ║
║    POST /predict             → churn probability + risk level + SHAP        ║
║    POST /predict-batch       → batch CSV prediction (returns CSV or JSON)   ║
║    GET  /feature-importance  → top-10 SHAP features from training          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.schemas import (
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
    SkippedRow,
    TopFactor,
)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS  — resolved relative to the project root (one level above backend/)
# ══════════════════════════════════════════════════════════════════════════════
_PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # CP/
_MODEL_PATH     = _PROJECT_ROOT / "ml" / "model.pkl"
_SCHEMA_PATH    = _PROJECT_ROOT / "ml" / "feature_schema.json"
_FEAT_IMP_PATH  = _PROJECT_ROOT / "ml" / "feature_importance.json"
_DATA_PATH      = _PROJECT_ROOT / "ml" / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

_SHAP_BACKGROUND_SIZE = 200   # rows used to initialise LinearExplainer
MAX_BATCH_ROWS        = 500   # maximum CSV rows allowed in POST /predict-batch


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — load model & schema once at process start
# ══════════════════════════════════════════════════════════════════════════════
_pipeline:          Any | None = None
_feature_schema:    dict | None = None
_feature_importance: list | None = None
_shap_explainer:    Any | None = None   # shap.LinearExplainer (or TreeExplainer)
_encoded_feat_names: list[str] = []     # human-readable post-encoding column names


def _load_artifacts() -> None:
    """Load model pipeline, feature schema, and build SHAP explainer at startup."""
    global _pipeline, _feature_schema, _feature_importance
    global _shap_explainer, _encoded_feat_names

    if not _MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {_MODEL_PATH}. "
            "Run `python ml/train.py` first to generate ml/model.pkl."
        )
    if not _SCHEMA_PATH.exists():
        raise RuntimeError(
            f"Feature schema not found at {_SCHEMA_PATH}. "
            "Run `python ml/train.py` first to generate ml/feature_schema.json."
        )

    _pipeline = joblib.load(_MODEL_PATH)
    _feature_schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    if _FEAT_IMP_PATH.exists():
        _feature_importance = json.loads(_FEAT_IMP_PATH.read_text(encoding="utf-8"))
    else:
        _feature_importance = []

    print(f"[startup] Model loaded from:          {_MODEL_PATH}")
    print(f"[startup] Feature schema loaded from: {_SCHEMA_PATH}")
    print(f"[startup] Feature importance loaded:  {len(_feature_importance)} entries")

    # ── Build SHAP explainer (once, at startup) ───────────────────────────────
    print("[startup] Building SHAP explainer …")
    preprocessor = _pipeline.named_steps["preprocessor"]
    model        = _pipeline.named_steps["model"]

    # Recover human-readable encoded feature names (strip num__/cat__ prefixes)
    raw_feat_names: list[str] = preprocessor.get_feature_names_out().tolist()
    _encoded_feat_names = [n.split("__", 1)[-1] for n in raw_feat_names]

    # Build a background sample for the explainer
    # Priority: raw CSV → fall back to a zero-matrix if CSV not present (e.g. on Render)
    if _DATA_PATH.exists():
        bg_df = pd.read_csv(_DATA_PATH)
        bg_df["TotalCharges"] = pd.to_numeric(bg_df["TotalCharges"], errors="coerce")
        bg_df["TotalCharges"] = bg_df["TotalCharges"].fillna(bg_df["TotalCharges"].median())
        bg_df = bg_df.drop(columns=["customerID", "Churn"], errors="ignore")
        sample = bg_df.sample(
            n=min(_SHAP_BACKGROUND_SIZE, len(bg_df)), random_state=42
        ).reset_index(drop=True)
        background = preprocessor.transform(sample)
        print(f"[startup] Background sample: {background.shape[0]} rows from CSV")
    else:
        # Fallback: zero-matrix based on encoded feature length (production / Render)
        background = np.zeros((1, len(_encoded_feat_names)))
        print(f"[startup] Background sample: zero matrix ({len(_encoded_feat_names)} features, CSV not present)")


    model_type = type(model).__name__
    tree_types = {"DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier"}
    if model_type in tree_types:
        _shap_explainer = shap.TreeExplainer(model)
        print(f"[startup] SHAP: TreeExplainer for {model_type}")
    else:
        _shap_explainer = shap.LinearExplainer(model, background)
        print(f"[startup] SHAP: LinearExplainer for {model_type}")

    print("[startup] SHAP explainer ready.")


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN  — replaces deprecated @app.on_event("startup")
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: "FastAPI"):  # noqa: ARG001
    """Load ML artifacts once at startup; nothing to clean up on shutdown."""
    _load_artifacts()
    yield


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "REST API that serves predictions from a trained sklearn Pipeline "
        "(LogisticRegression / RandomForest / XGBoost). "
        "POST a customer record to /predict to get churn probability and risk level."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_ALLOWED_ORIGINS = [
    "http://localhost:5173",    # Vite dev server
    "http://localhost:3000",    # CRA / Next.js dev fallback
    "https://YOUR-APP.vercel.app",  # Production Vercel domain placeholder
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Automatically allows all Vercel prod & preview deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-Total-Rows",
        "X-Valid-Rows",
        "X-Skipped-Rows",
    ],
)




# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _risk_level(prob: float) -> str:
    """Map a churn probability to a human-readable risk band."""
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    elif prob < 0.80:
        return "High"
    else:
        return "Very High"


def _get_top_factors(encoded_row: np.ndarray, top_n: int = 5) -> list[TopFactor]:
    """
    Compute per-prediction SHAP values for a single already-encoded row and
    return the top_n features sorted by |SHAP|.

    Args:
        encoded_row: 2-D numpy array of shape (1, n_features) — output of
                     the pipeline's ColumnTransformer for one customer.
        top_n:       how many top factors to return (default 5).

    Returns:
        List of TopFactor objects, sorted descending by |impact|.
    """
    if _shap_explainer is None:
        return []

    raw = _shap_explainer.shap_values(encoded_row)
    # Some explainer versions return list[array] for binary classifiers
    shap_vals: np.ndarray = raw[1] if isinstance(raw, list) else raw
    shap_row = shap_vals[0]   # shape: (n_features,)

    # Pair each feature with its signed SHAP value, sort by |value|
    pairs = sorted(
        zip(_encoded_feat_names, shap_row.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return [
        TopFactor(
            feature=feat,
            impact=round(float(val), 6),
            direction="increases_risk" if val >= 0 else "decreases_risk",
        )
        for feat, val in pairs[:top_n]
    ]


def _customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convert a validated Pydantic model into a single-row DataFrame whose
    column names and dtypes exactly match the raw training data.
    The sklearn Pipeline's ColumnTransformer handles encoding from here.
    """
    row = customer.model_dump()

    # Enforce dtypes that the pipeline's ColumnTransformer expects
    df = pd.DataFrame([row])
    df["SeniorCitizen"]  = df["SeniorCitizen"].astype("int64")
    df["tenure"]         = df["tenure"].astype("int64")
    df["MonthlyCharges"] = df["MonthlyCharges"].astype("float64")
    df["TotalCharges"]   = df["TotalCharges"].astype("float64")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check / liveness probe",
    tags=["Utility"],
)
async def health() -> HealthResponse:
    """Returns 200 OK when the service is up and the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        feature_schema_loaded=_feature_schema is not None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict customer churn",
    tags=["Prediction"],
)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Accept a single customer's raw feature values and return:
    - `churn_probability` — model's confidence the customer will churn
    - `prediction`        — "Churn" or "No Churn"
    - `risk_level`        — Low / Medium / High / Very High
    - `top_factors`       — top 5 SHAP drivers for this specific customer

    The request body is validated against all 19 training features.
    Invalid category values or missing fields return HTTP 422 with details.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again in a moment.",
        )

    try:
        df = _customer_to_dataframe(customer)

        # ── Step 1: encode only (ColumnTransformer) ───────────────────────────
        preprocessor = _pipeline.named_steps["preprocessor"]
        encoded_row: np.ndarray = preprocessor.transform(df)   # shape (1, n_features)

        # ── Step 2: predict using the full pipeline ───────────────────────────
        prob: float = float(_pipeline.predict_proba(df)[0, 1])

        # ── Step 3: per-prediction SHAP top-5 ────────────────────────────────
        top_factors = _get_top_factors(encoded_row)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(
        churn_probability=round(prob, 4),
        prediction="Churn" if prob >= 0.5 else "No Churn",
        risk_level=_risk_level(prob),
        top_factors=top_factors,
    )


@app.get(
    "/feature-importance",
    summary="Top-10 SHAP feature importances",
    tags=["Explainability"],
)
async def feature_importance() -> list[dict]:
    """
    Returns the top-10 most important features ranked by mean absolute SHAP
    value, as computed during training and saved in ml/feature_importance.json.

    Shape:
        [{"rank": 1, "feature": "tenure", "mean_abs_shap": 1.021179}, ...]
    """
    if _feature_importance is None:
        raise HTTPException(
            status_code=503,
            detail="Feature importance data not loaded.",
        )
    return _feature_importance


@app.post(
    "/predict-batch",
    summary="Batch churn prediction from uploaded CSV",
    tags=["Prediction"],
    response_model=None,
)
async def predict_batch(
    file: UploadFile = File(..., description="CSV file containing customer records"),
    format: Literal["csv", "json"] = Query(
        "csv",
        description="Output format: 'csv' for downloadable file, 'json' for JSON summary",
    ),
):
    """
    Accept a CSV file containing batch customer records and return predictions.

    - Validates presence of all 19 required columns from the training feature schema.
    - Validates row values against category constraints and types.
    - Invalid rows are skipped and reported in the response/headers.
    - Runs all valid rows through the pipeline in a single batch inference call.
    - Max limit: 500 rows per request (returns HTTP 413 if exceeded).
    - Query param `format=csv` (default) returns a downloadable CSV with appended prediction columns.
    - Query param `format=json` returns structured JSON with skipped row details and predictions.
    """
    if _pipeline is None or _feature_schema is None:
        raise HTTPException(
            status_code=503,
            detail="Model or schema is not loaded yet. Please try again in a moment.",
        )

    # ── 1. Read & Parse CSV ───────────────────────────────────────────────────
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse uploaded CSV file: {exc}",
        ) from exc

    # ── 2. Check Row Limit (413 Payload Too Large) ───────────────────────────
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds maximum batch limit of {MAX_BATCH_ROWS} rows "
                f"(received {len(df)} rows)."
            ),
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=422,
            detail="Uploaded CSV file is empty (0 data rows).",
        )

    # ── 3. Validate Required Columns (422 Unprocessable Entity) ───────────────
    required_cols: list[str] = _feature_schema["feature_names"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {', '.join(missing_cols)}",
        )

    # ── 4. Row-by-Row Validation (Skip-and-Report) ────────────────────────────
    valid_indices: list[int] = []
    valid_records: list[dict[str, Any]] = []
    skipped_details: list[SkippedRow] = []

    numeric_cols = set(_feature_schema["numeric_features"])

    for idx, row in df.iterrows():
        row_num = int(idx) + 1  # 1-indexed row number
        row_dict: dict[str, Any] = {}
        try:
            for col in required_cols:
                val = row[col]
                if col in numeric_cols:
                    if pd.isna(val) or str(val).strip() == "":
                        raise ValueError(f"Missing numeric value for '{col}'")
                    if col in ["SeniorCitizen", "tenure"]:
                        row_dict[col] = int(float(val))
                    else:
                        row_dict[col] = float(val)
                else:
                    if pd.isna(val):
                        raise ValueError(f"Missing value for categorical column '{col}'")
                    row_dict[col] = str(val).strip()

            # Pydantic validation handles category constraints & cross-field rules
            CustomerFeatures(**row_dict)
            valid_indices.append(int(idx))
            valid_records.append(row_dict)

        except Exception as err:
            if isinstance(err, ValidationError):
                err_msgs = [f"{e['loc'][0]}: {e['msg']}" for e in err.errors()]
                skipped_details.append(SkippedRow(row_number=row_num, reason="; ".join(err_msgs)))
            else:
                skipped_details.append(SkippedRow(row_number=row_num, reason=str(err)))

    # If no rows are valid
    if not valid_records:
        if format == "json":
            return BatchPredictionResponse(
                total_rows=len(df),
                valid_rows=0,
                skipped_rows=len(skipped_details),
                skipped_details=skipped_details,
                predictions=[],
            )
        else:
            first_err = skipped_details[0].reason if skipped_details else "No valid records"
            raise HTTPException(
                status_code=422,
                detail=(
                    f"All {len(df)} rows were invalid and skipped. "
                    f"Example error (Row {skipped_details[0].row_number if skipped_details else 1}): {first_err}"
                ),
            )

    # ── 5. Single Batch Pipeline Execution ────────────────────────────────────
    valid_df = pd.DataFrame(valid_records)
    valid_df["SeniorCitizen"]  = valid_df["SeniorCitizen"].astype("int64")
    valid_df["tenure"]         = valid_df["tenure"].astype("int64")
    valid_df["MonthlyCharges"] = valid_df["MonthlyCharges"].astype("float64")
    valid_df["TotalCharges"]   = valid_df["TotalCharges"].astype("float64")

    probs: np.ndarray = _pipeline.predict_proba(valid_df)[:, 1]
    churn_probs = [round(float(p), 4) for p in probs]
    predictions = ["Churn" if p >= 0.5 else "No Churn" for p in probs]
    risk_levels = [_risk_level(p) for p in probs]

    # ── 6. Prepare Output ─────────────────────────────────────────────────────
    output_df = df.loc[valid_indices].copy()
    output_df["churn_probability"] = churn_probs
    output_df["prediction"] = predictions
    output_df["risk_level"] = risk_levels

    if format == "json":
        # Replace NaN with None for valid JSON output
        clean_df = output_df.where(pd.notnull(output_df), None)
        records = clean_df.to_dict(orient="records")
        return BatchPredictionResponse(
            total_rows=len(df),
            valid_rows=len(valid_indices),
            skipped_rows=len(skipped_details),
            skipped_details=skipped_details,
            predictions=records,
        )

    # Return CSV file stream / response
    csv_buf = io.StringIO()
    output_df.to_csv(csv_buf, index=False)
    csv_str = csv_buf.getvalue()

    headers = {
        "Content-Disposition": 'attachment; filename="churn_predictions.csv"',
        "X-Total-Rows": str(len(df)),
        "X-Valid-Rows": str(len(valid_indices)),
        "X-Skipped-Rows": str(len(skipped_details)),
    }

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers=headers,
    )

