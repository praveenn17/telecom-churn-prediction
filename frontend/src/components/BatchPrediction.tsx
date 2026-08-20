import { useState, useRef, useMemo } from "react";
import { predictBatchJson, downloadBatchCsv } from "../api";
import type { BatchPredictionResponse, BatchPredictionRecord } from "../types";

const RISK_WEIGHTS: Record<string, number> = {
  "Very High": 4,
  High: 3,
  Medium: 2,
  Low: 1,
};

function riskClass(risk: string) {
  return (
    {
      Low: "risk-low",
      Medium: "risk-medium",
      High: "risk-high",
      "Very High": "risk-very-high",
    }[risk] ?? "risk-low"
  );
}

function riskRowClass(risk: string) {
  return (
    {
      Low: "batch-row-low",
      Medium: "batch-row-medium",
      High: "batch-row-high",
      "Very High": "batch-row-very-high",
    }[risk] ?? ""
  );
}

export function BatchPrediction() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<BatchPredictionResponse | null>(null);

  // Sorting & Filtering
  const [sortField, setSortField] = useState<keyof BatchPredictionRecord | "risk_level_weight">("risk_level_weight");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [search, setSearch] = useState("");
  const [showSkipped, setShowSkipped] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileSelect = (selectedFile: File | null) => {
    if (!selectedFile) return;
    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a valid CSV file (.csv format only).");
      return;
    }
    setError(null);
    setData(null);
    setFile(selectedFile);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleRunBatch = async () => {
    if (!file) {
      setError("Please select or drop a CSV file first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictBatchJson(file);
      setData(res);
      // Auto-show skipped box if any rows failed
      if (res.skipped_rows > 0) {
        setShowSkipped(true);
      }
    } catch (err: unknown) {
      setError((err as Error).message ?? "Batch prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCsv = async () => {
    if (!file) return;
    setDownloading(true);
    try {
      await downloadBatchCsv(file);
    } catch (err: unknown) {
      setError((err as Error).message ?? "Failed to download results CSV.");
    } finally {
      setDownloading(false);
    }
  };

  const handleSort = (field: keyof BatchPredictionRecord | "risk_level_weight") => {
    if (sortField === field) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortOrder("desc"); // Default to desc for high-to-low
    }
  };

  // Summary Metrics
  const summaryMetrics = useMemo(() => {
    if (!data || data.predictions.length === 0) return null;
    const total = data.predictions.length;
    const churners = data.predictions.filter((p) => p.prediction === "Churn").length;
    const highRisk = data.predictions.filter(
      (p) => p.risk_level === "High" || p.risk_level === "Very High"
    ).length;
    const avgProb =
      data.predictions.reduce((acc, p) => acc + p.churn_probability, 0) / total;

    return {
      total,
      churners,
      churnRate: Math.round((churners / total) * 100),
      highRisk,
      avgProb: Math.round(avgProb * 100),
    };
  }, [data]);

  // Filtered & Sorted Rows
  const sortedRows = useMemo(() => {
    if (!data) return [];
    let rows = [...data.predictions];

    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter((r) => {
        return (
          (r.customerID && String(r.customerID).toLowerCase().includes(q)) ||
          (r.Contract && String(r.Contract).toLowerCase().includes(q)) ||
          (r.InternetService && String(r.InternetService).toLowerCase().includes(q)) ||
          (r.PaymentMethod && String(r.PaymentMethod).toLowerCase().includes(q)) ||
          (r.risk_level && String(r.risk_level).toLowerCase().includes(q))
        );
      });
    }

    rows.sort((a, b) => {
      let valA: any;
      let valB: any;

      if (sortField === "risk_level_weight") {
        valA = RISK_WEIGHTS[a.risk_level] ?? 0;
        valB = RISK_WEIGHTS[b.risk_level] ?? 0;
      } else {
        valA = a[sortField];
        valB = b[sortField];
      }

      if (valA === valB) return 0;
      if (valA == null) return 1;
      if (valB == null) return -1;

      const comparison = valA > valB ? 1 : -1;
      return sortOrder === "asc" ? comparison : -comparison;
    });

    return rows;
  }, [data, search, sortField, sortOrder]);

  return (
    <div className="batch-container">
      {/* ── Dropzone & Action Card ────────────────────────────────────────── */}
      <div className="card batch-upload-card">
        <div className="card-header">
          <span className="card-header-icon">📁</span>
          <span className="card-header-title">Batch Customer Evaluation</span>
          <span className="card-header-desc">Upload CSV (max 500 rows)</span>
        </div>
        <div className="card-body">
          <div
            className={`dropzone ${isDragging ? "dropzone-active" : ""} ${file ? "dropzone-has-file" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept=".csv"
              style={{ display: "none" }}
              onChange={(e) => handleFileSelect(e.target.files?.[0] ?? null)}
            />

            <div className="dropzone-icon">
              {file ? "📄" : "☁️"}
            </div>

            {file ? (
              <div className="dropzone-file-info">
                <div className="dropzone-filename">{file.name}</div>
                <div className="dropzone-filesize">
                  {(file.size / 1024).toFixed(1)} KB · CSV File
                </div>
                <span className="dropzone-change-hint">Click or drag a new file to replace</span>
              </div>
            ) : (
              <div className="dropzone-text">
                <p className="dropzone-main-text">
                  Drag and drop your customer CSV here, or <span>Browse Files</span>
                </p>
                <p className="dropzone-sub-text">
                  Must contain the 19 standard Telco features (tenure, Contract, MonthlyCharges, etc.)
                </p>
              </div>
            )}
          </div>

          <div className="batch-action-row">
            <button
              type="button"
              className="submit-btn batch-run-btn"
              disabled={!file || loading}
              onClick={handleRunBatch}
            >
              {loading ? (
                <>
                  <div className="spinner" /> Evaluating Batch Records…
                </>
              ) : (
                <>
                  <span>⚡</span> Run Batch Prediction
                </>
              )}
            </button>

            {data && (
              <button
                type="button"
                className="download-btn"
                disabled={downloading}
                onClick={handleDownloadCsv}
                title="Download full results as CSV with appended predictions"
              >
                {downloading ? (
                  <>
                    <div className="spinner spinner-cyan" /> Generating CSV…
                  </>
                ) : (
                  <>
                    <span>⬇</span> Download Results CSV
                  </>
                )}
              </button>
            )}
          </div>

          {error && <div className="error-banner">⚠ {error}</div>}

          {/* ── Skipped Rows Banner ────────────────────────────────────────── */}
          {data && data.skipped_rows > 0 && (
            <div className="skipped-banner">
              <div className="skipped-banner-header">
                <span>
                  ⚠ <strong>{data.skipped_rows} of {data.total_rows} rows</strong> were invalid and skipped.
                </span>
                <button
                  type="button"
                  className="skipped-toggle-btn"
                  onClick={() => setShowSkipped(!showSkipped)}
                >
                  {showSkipped ? "Hide Details ▲" : "View Skipped Rows ▼"}
                </button>
              </div>

              {showSkipped && (
                <div className="skipped-details-list">
                  <table className="skipped-table">
                    <thead>
                      <tr>
                        <th>Row #</th>
                        <th>Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.skipped_details.map((skip, i) => (
                        <tr key={i}>
                          <td className="skip-row-num">Row {skip.row_number}</td>
                          <td className="skip-reason">{skip.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Batch Results Section ─────────────────────────────────────────── */}
      {data && (
        <div className="batch-results-wrap">
          {/* Summary Metric Cards */}
          {summaryMetrics && (
            <div className="batch-summary-grid">
              <div className="metric-card">
                <div className="metric-icon">👥</div>
                <div className="metric-content">
                  <div className="metric-value">{data.valid_rows}</div>
                  <div className="metric-label">Evaluated Customers</div>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">🚨</div>
                <div className="metric-content">
                  <div className="metric-value" style={{ color: "var(--red)" }}>
                    {summaryMetrics.churners}{" "}
                    <span className="metric-sub">({summaryMetrics.churnRate}%)</span>
                  </div>
                  <div className="metric-label">Predicted Churners</div>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">🔥</div>
                <div className="metric-content">
                  <div className="metric-value" style={{ color: "var(--orange)" }}>
                    {summaryMetrics.highRisk}
                  </div>
                  <div className="metric-label">High / Very High Risk</div>
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-icon">📊</div>
                <div className="metric-content">
                  <div className="metric-value" style={{ color: "var(--cyan)" }}>
                    {summaryMetrics.avgProb}%
                  </div>
                  <div className="metric-label">Average Churn Probability</div>
                </div>
              </div>
            </div>
          )}

          {/* Results Table Card */}
          <div className="card batch-table-card">
            <div className="card-header batch-table-header">
              <div className="batch-table-title-group">
                <span className="card-header-icon">📋</span>
                <span className="card-header-title">Customer Prediction Records</span>
                <span className="batch-count-badge">
                  {sortedRows.length} {sortedRows.length === 1 ? "record" : "records"}
                </span>
              </div>

              <div className="batch-search-wrap">
                <input
                  type="text"
                  className="batch-search-input"
                  placeholder="Filter records (ID, contract, risk...)"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                {search && (
                  <button
                    type="button"
                    className="batch-search-clear"
                    onClick={() => setSearch("")}
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>

            <div className="batch-table-scroll">
              <table className="batch-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th
                      className="sortable-th"
                      onClick={() => handleSort("customerID")}
                    >
                      Customer ID {sortField === "customerID" && (sortOrder === "asc" ? "▲" : "▼")}
                    </th>
                    <th
                      className="sortable-th"
                      onClick={() => handleSort("tenure")}
                    >
                      Tenure {sortField === "tenure" && (sortOrder === "asc" ? "▲" : "▼")}
                    </th>
                    <th
                      className="sortable-th"
                      onClick={() => handleSort("Contract")}
                    >
                      Contract {sortField === "Contract" && (sortOrder === "asc" ? "▲" : "▼")}
                    </th>
                    <th
                      className="sortable-th"
                      onClick={() => handleSort("MonthlyCharges")}
                    >
                      Monthly ($) {sortField === "MonthlyCharges" && (sortOrder === "asc" ? "▲" : "▼")}
                    </th>
                    <th
                      className="sortable-th"
                      onClick={() => handleSort("churn_probability")}
                    >
                      Churn Probability {sortField === "churn_probability" && (sortOrder === "asc" ? "▲" : "▼")}
                    </th>
                    <th>Prediction</th>
                    <th
                      className="sortable-th"
                      onClick={() => handleSort("risk_level_weight")}
                    >
                      Risk Level {sortField === "risk_level_weight" && (sortOrder === "asc" ? "▲" : "▼")}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedRows.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="batch-table-empty">
                        No customer records match your filter criteria.
                      </td>
                    </tr>
                  ) : (
                    sortedRows.map((row, idx) => {
                      const probPct = Math.round(row.churn_probability * 100);
                      const displayId =
                        row.customerID ?? row.customer_id ?? `Row #${idx + 1}`;

                      return (
                        <tr key={idx} className={riskRowClass(row.risk_level)}>
                          <td className="col-idx">{idx + 1}</td>
                          <td className="col-id" title={String(displayId)}>
                            {displayId}
                          </td>
                          <td>{row.tenure != null ? `${row.tenure} mo` : "—"}</td>
                          <td className="col-contract">{row.Contract ?? "—"}</td>
                          <td className="col-num">
                            {row.MonthlyCharges != null
                              ? `$${Number(row.MonthlyCharges).toFixed(2)}`
                              : "—"}
                          </td>
                          <td className="col-prob">
                            <div className="table-prob-cell">
                              <span className="table-prob-val">{probPct}%</span>
                              <div className="table-prob-bar-track">
                                <div
                                  className={`table-prob-bar-fill ${riskClass(row.risk_level)}`}
                                  style={{ width: `${probPct}%` }}
                                />
                              </div>
                            </div>
                          </td>
                          <td>
                            <span
                              className={`prediction-badge-sm ${
                                row.prediction === "Churn" ? "badge-churn" : "badge-no-churn"
                              }`}
                            >
                              {row.prediction === "Churn" ? "Churn" : "No Churn"}
                            </span>
                          </td>
                          <td>
                            <span className={`risk-badge-sm ${riskClass(row.risk_level)}`}>
                              {row.risk_level}
                            </span>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
