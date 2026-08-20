import type {
  BatchPredictionResponse,
  CustomerFeatures,
  FeatureImportanceItem,
  PredictionResult,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function predictChurn(features: CustomerFeatures): Promise<PredictionResult> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof detail.detail === "string"
        ? detail.detail
        : JSON.stringify(detail.detail)
    );
  }
  return res.json() as Promise<PredictionResult>;
}

export async function fetchFeatureImportance(): Promise<FeatureImportanceItem[]> {
  const res = await fetch(`${API_URL}/feature-importance`);
  if (!res.ok) throw new Error("Could not load feature importances");
  return res.json() as Promise<FeatureImportanceItem[]>;
}

export async function predictBatchJson(file: File): Promise<BatchPredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/predict-batch?format=json`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof detail.detail === "string"
        ? detail.detail
        : JSON.stringify(detail.detail)
    );
  }
  return res.json() as Promise<BatchPredictionResponse>;
}

export async function downloadBatchCsv(file: File): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/predict-batch?format=csv`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof detail.detail === "string"
        ? detail.detail
        : JSON.stringify(detail.detail)
    );
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `churn_predictions_${file.name.replace(/\.csv$/i, "")}.csv`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
