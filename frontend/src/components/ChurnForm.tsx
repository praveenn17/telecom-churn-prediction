import { useState } from "react";
import type { CustomerFeatures } from "../types";

interface Props {
  onSubmit: (data: CustomerFeatures) => void;
  loading: boolean;
  error: string | null;
}

const DEFAULTS: CustomerFeatures = {
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "No",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 70.35,
  TotalCharges: 840.2,
};

// Helper: internet add-on options depend on whether internet service is active
const internetAddonOptions = (hasInternet: boolean) =>
  hasInternet ? ["No", "Yes"] : ["No internet service"];

function Select<K extends keyof CustomerFeatures>({
  id, label, value, options, onChange, disabled,
}: {
  id: K; label: string; value: string; options: string[];
  onChange: (val: string) => void; disabled?: boolean;
}) {
  return (
    <div className="field">
      <label className="field-label" htmlFor={String(id)}>{label}</label>
      <select
        id={String(id)}
        className="field-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

export function ChurnForm({ onSubmit, loading, error }: Props) {
  const [form, setForm] = useState<CustomerFeatures>(DEFAULTS);

  const set = <K extends keyof CustomerFeatures>(key: K, val: CustomerFeatures[K]) =>
    setForm((prev) => {
      const next = { ...prev, [key]: val };
      // Auto-correct internet add-on fields when internet service changes to "No"
      if (key === "InternetService" && val === "No") {
        const addons: (keyof CustomerFeatures)[] = [
          "OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies",
        ];
        addons.forEach((a) => { (next as Record<string, unknown>)[a] = "No internet service"; });
      }
      return next;
    });

  const hasInternet = form.InternetService !== "No";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      <div className="card">
        <div className="card-header">
          <span className="card-header-icon">👤</span>
          <span className="card-header-title">Customer Profile</span>
          <span className="card-header-desc">Fill in all fields</span>
        </div>
        <div className="card-body">

          {/* ── Personal Info ───────────────────────────────────────────── */}
          <div className="section-label">Personal Info</div>
          <div className="section-group">
            <Select id="gender" label="Gender"
              value={form.gender} options={["Female", "Male"]}
              onChange={(v) => set("gender", v as CustomerFeatures["gender"])} />

            <div className="field toggle-field">
              <label className="field-label" htmlFor="senior">Senior Citizen</label>
              <div
                id="senior"
                className={`toggle-track ${form.SeniorCitizen === 1 ? "on" : ""}`}
                role="checkbox"
                aria-checked={form.SeniorCitizen === 1}
                tabIndex={0}
                onClick={() => set("SeniorCitizen", form.SeniorCitizen === 1 ? 0 : 1)}
                onKeyDown={(e) => e.key === " " && set("SeniorCitizen", form.SeniorCitizen === 1 ? 0 : 1)}
              >
                <div className="toggle-thumb" />
              </div>
              <span className="toggle-label">{form.SeniorCitizen === 1 ? "Yes" : "No"}</span>
            </div>

            <Select id="Partner" label="Partner"
              value={form.Partner} options={["No", "Yes"]}
              onChange={(v) => set("Partner", v as "No" | "Yes")} />

            <Select id="Dependents" label="Dependents"
              value={form.Dependents} options={["No", "Yes"]}
              onChange={(v) => set("Dependents", v as "No" | "Yes")} />
          </div>

          {/* ── Phone Service ───────────────────────────────────────────── */}
          <div className="section-label">Phone Service</div>
          <div className="section-group">
            <Select id="PhoneService" label="Phone Service"
              value={form.PhoneService} options={["No", "Yes"]}
              onChange={(v) => set("PhoneService", v as "No" | "Yes")} />

            <Select id="MultipleLines" label="Multiple Lines"
              value={form.MultipleLines}
              options={form.PhoneService === "Yes" ? ["No", "Yes"] : ["No phone service"]}
              onChange={(v) => set("MultipleLines", v as CustomerFeatures["MultipleLines"])}
              disabled={form.PhoneService === "No"} />
          </div>

          {/* ── Internet & Add-ons ──────────────────────────────────────── */}
          <div className="section-label">Internet & Add-ons</div>
          <div className="section-group">
            <Select id="InternetService" label="Internet Service"
              value={form.InternetService} options={["DSL", "Fiber optic", "No"]}
              onChange={(v) => set("InternetService", v as CustomerFeatures["InternetService"])} />

            <Select id="OnlineSecurity" label="Online Security"
              value={form.OnlineSecurity}
              options={internetAddonOptions(hasInternet)}
              onChange={(v) => set("OnlineSecurity", v as CustomerFeatures["OnlineSecurity"])}
              disabled={!hasInternet} />

            <Select id="OnlineBackup" label="Online Backup"
              value={form.OnlineBackup}
              options={internetAddonOptions(hasInternet)}
              onChange={(v) => set("OnlineBackup", v as CustomerFeatures["OnlineBackup"])}
              disabled={!hasInternet} />

            <Select id="DeviceProtection" label="Device Protection"
              value={form.DeviceProtection}
              options={internetAddonOptions(hasInternet)}
              onChange={(v) => set("DeviceProtection", v as CustomerFeatures["DeviceProtection"])}
              disabled={!hasInternet} />

            <Select id="TechSupport" label="Tech Support"
              value={form.TechSupport}
              options={internetAddonOptions(hasInternet)}
              onChange={(v) => set("TechSupport", v as CustomerFeatures["TechSupport"])}
              disabled={!hasInternet} />

            <Select id="StreamingTV" label="Streaming TV"
              value={form.StreamingTV}
              options={internetAddonOptions(hasInternet)}
              onChange={(v) => set("StreamingTV", v as CustomerFeatures["StreamingTV"])}
              disabled={!hasInternet} />

            <Select id="StreamingMovies" label="Streaming Movies"
              value={form.StreamingMovies}
              options={internetAddonOptions(hasInternet)}
              onChange={(v) => set("StreamingMovies", v as CustomerFeatures["StreamingMovies"])}
              disabled={!hasInternet} />
          </div>

          {/* ── Billing ─────────────────────────────────────────────────── */}
          <div className="section-label">Billing</div>
          <div className="section-group">
            <div className="field">
              <label className="field-label" htmlFor="tenure">Tenure (months)</label>
              <input id="tenure" type="number" className="field-input"
                min={0} max={120} step={1}
                value={form.tenure}
                onChange={(e) => set("tenure", Number(e.target.value))} />
            </div>

            <Select id="Contract" label="Contract"
              value={form.Contract}
              options={["Month-to-month", "One year", "Two year"]}
              onChange={(v) => set("Contract", v as CustomerFeatures["Contract"])} />

            <Select id="PaperlessBilling" label="Paperless Billing"
              value={form.PaperlessBilling} options={["No", "Yes"]}
              onChange={(v) => set("PaperlessBilling", v as "No" | "Yes")} />

            <Select id="PaymentMethod" label="Payment Method"
              value={form.PaymentMethod}
              options={["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]}
              onChange={(v) => set("PaymentMethod", v as CustomerFeatures["PaymentMethod"])} />

            <div className="field">
              <label className="field-label" htmlFor="MonthlyCharges">Monthly Charges ($)</label>
              <input id="MonthlyCharges" type="number" className="field-input"
                min={0} max={200} step={0.01}
                value={form.MonthlyCharges}
                onChange={(e) => set("MonthlyCharges", parseFloat(e.target.value) || 0)} />
            </div>

            <div className="field">
              <label className="field-label" htmlFor="TotalCharges">Total Charges ($)</label>
              <input id="TotalCharges" type="number" className="field-input"
                min={0} step={0.01}
                value={form.TotalCharges}
                onChange={(e) => set("TotalCharges", parseFloat(e.target.value) || 0)} />
            </div>
          </div>

          {/* ── Submit ───────────────────────────────────────────────────── */}
          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? (
              <><div className="spinner" /> Predicting…</>
            ) : (
              <><span>⚡</span> Predict Churn Risk</>
            )}
          </button>

          {error && <div className="error-banner">⚠ {error}</div>}
        </div>
      </div>
    </form>
  );
}
