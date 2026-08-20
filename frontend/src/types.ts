// All types shared between API calls and components

export interface CustomerFeatures {
  gender: "Female" | "Male";
  SeniorCitizen: 0 | 1;
  Partner: "No" | "Yes";
  Dependents: "No" | "Yes";
  tenure: number;
  PhoneService: "No" | "Yes";
  MultipleLines: "No" | "No phone service" | "Yes";
  InternetService: "DSL" | "Fiber optic" | "No";
  OnlineSecurity: "No" | "No internet service" | "Yes";
  OnlineBackup: "No" | "No internet service" | "Yes";
  DeviceProtection: "No" | "No internet service" | "Yes";
  TechSupport: "No" | "No internet service" | "Yes";
  StreamingTV: "No" | "No internet service" | "Yes";
  StreamingMovies: "No" | "No internet service" | "Yes";
  Contract: "Month-to-month" | "One year" | "Two year";
  PaperlessBilling: "No" | "Yes";
  PaymentMethod:
    | "Bank transfer (automatic)"
    | "Credit card (automatic)"
    | "Electronic check"
    | "Mailed check";
  MonthlyCharges: number;
  TotalCharges: number;
}

export interface TopFactor {
  feature: string;
  impact: number;                               // signed SHAP value
  direction: "increases_risk" | "decreases_risk";
}

export interface PredictionResult {
  churn_probability: number;
  prediction: "Churn" | "No Churn";
  risk_level: "Low" | "Medium" | "High" | "Very High";
  top_factors: TopFactor[];
}

export interface FeatureImportanceItem {
  rank: number;
  feature: string;
  mean_abs_shap: number;
}

export interface SkippedRow {
  row_number: number;
  reason: string;
}

export interface BatchPredictionRecord {
  [key: string]: any;
  customerID?: string;
  gender?: string;
  SeniorCitizen?: number;
  Partner?: string;
  Dependents?: string;
  tenure?: number;
  PhoneService?: string;
  MultipleLines?: string;
  InternetService?: string;
  OnlineSecurity?: string;
  OnlineBackup?: string;
  DeviceProtection?: string;
  TechSupport?: string;
  StreamingTV?: string;
  StreamingMovies?: string;
  Contract?: string;
  PaperlessBilling?: string;
  PaymentMethod?: string;
  MonthlyCharges?: number;
  TotalCharges?: number;
  Churn?: string;
  churn_probability: number;
  prediction: "Churn" | "No Churn";
  risk_level: "Low" | "Medium" | "High" | "Very High";
}

export interface BatchPredictionResponse {
  total_rows: number;
  valid_rows: number;
  skipped_rows: number;
  skipped_details: SkippedRow[];
  predictions: BatchPredictionRecord[];
}
