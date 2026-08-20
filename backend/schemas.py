"""
Pydantic request / response schemas for the Churn Prediction API.
All Literal[] values are derived directly from ml/feature_schema.json
so they stay in sync with the training data categories.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST  —  one customer record (raw, pre-encoding)
# ══════════════════════════════════════════════════════════════════════════════

class CustomerFeatures(BaseModel):
    """
    All 19 raw input features expected by the trained pipeline.
    Field names and allowed category values match the training CSV exactly.
    """

    # ── Categorical ────────────────────────────────────────────────────────────
    gender: Literal["Female", "Male"] = Field(
        ..., description="Customer gender"
    )
    Partner: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has a partner"
    )
    Dependents: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has dependents"
    )
    PhoneService: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer has phone service"
    )
    MultipleLines: Literal["No", "No phone service", "Yes"] = Field(
        ..., description="Whether the customer has multiple phone lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Type of internet service"
    )
    OnlineSecurity: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has online security add-on"
    )
    OnlineBackup: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has online backup add-on"
    )
    DeviceProtection: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has device protection add-on"
    )
    TechSupport: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has tech support add-on"
    )
    StreamingTV: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has streaming TV add-on"
    )
    StreamingMovies: Literal["No", "No internet service", "Yes"] = Field(
        ..., description="Whether the customer has streaming movies add-on"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Type of customer contract"
    )
    PaperlessBilling: Literal["No", "Yes"] = Field(
        ..., description="Whether the customer uses paperless billing"
    )
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., description="Customer payment method")

    # ── Numeric ────────────────────────────────────────────────────────────────
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Whether the customer is a senior citizen (0 = No, 1 = Yes)"
    )
    tenure: int = Field(
        ..., ge=0, le=120,
        description="Number of months the customer has been with the company"
    )
    MonthlyCharges: float = Field(
        ..., ge=0.0, le=200.0, description="Monthly bill amount in USD"
    )
    TotalCharges: float = Field(
        ..., ge=0.0, description="Total amount charged to the customer (USD)"
    )

    @model_validator(mode="after")
    def check_internet_service_consistency(self) -> "CustomerFeatures":
        """
        If InternetService is 'No', internet-dependent add-on fields must be
        'No internet service' — catch the most common input mistake early.
        """
        internet_addons = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        if self.InternetService == "No":
            for field in internet_addons:
                val = getattr(self, field)
                if val != "No internet service":
                    raise ValueError(
                        f"When InternetService is 'No', {field} must be "
                        f"'No internet service' (got '{val}')."
                    )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 840.20,
            }
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE  —  per-prediction SHAP factor
# ══════════════════════════════════════════════════════════════════════════════

class TopFactor(BaseModel):
    """
    A single feature's contribution to the current prediction.
    Positive SHAP → increases churn probability.
    Negative SHAP → decreases churn probability.
    """
    feature: str = Field(
        ..., description="Human-readable encoded feature name (sklearn prefixes stripped)"
    )
    impact: float = Field(
        ..., description="Signed SHAP value for this prediction row (positive = more churn)"
    )
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Whether this feature pushes the prediction toward or away from churn"
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE  —  prediction result
# ══════════════════════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    churn_probability: float = Field(
        ..., description="Probability that the customer will churn (0.0–1.0)"
    )
    prediction: Literal["Churn", "No Churn"] = Field(
        ..., description="Binary churn prediction"
    )
    risk_level: Literal["Low", "Medium", "High", "Very High"] = Field(
        ...,
        description=(
            "Risk band: Low (0–0.29) | Medium (0.30–0.59) | "
            "High (0.60–0.79) | Very High (0.80–1.00)"
        ),
    )
    top_factors: list[TopFactor] = Field(
        default_factory=list,
        description="Top 5 features by absolute SHAP value for this specific prediction",
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE  —  health check
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # suppress model_ namespace warning

    status: str = "ok"
    model_loaded: bool
    feature_schema_loaded: bool


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE  —  batch prediction
# ══════════════════════════════════════════════════════════════════════════════

class SkippedRow(BaseModel):
    row_number: int = Field(..., description="1-indexed row number in the uploaded CSV")
    reason: str = Field(..., description="Validation failure details")


class BatchPredictionResponse(BaseModel):
    total_rows: int = Field(..., description="Total rows in the uploaded CSV")
    valid_rows: int = Field(..., description="Number of valid rows processed")
    skipped_rows: int = Field(..., description="Number of skipped rows due to validation errors")
    skipped_details: list[SkippedRow] = Field(
        default_factory=list, description="Details of skipped rows"
    )
    predictions: list[dict] = Field(
        default_factory=list,
        description="List of predictions with all original columns plus prediction results",
    )

