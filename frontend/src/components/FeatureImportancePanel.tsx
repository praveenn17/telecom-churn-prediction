import type { FeatureImportanceItem } from "../types";

interface Props {
  items: FeatureImportanceItem[];
  loading: boolean;
  error: string | null;
}

// Format encoded feature names to be more readable
function formatFeatureName(raw: string): string {
  return raw
    .replace(/_/g, " ")
    .replace(/\bNo\b/g, "No")
    .replace(/\bYes\b/g, "Yes");
}

function SkeletonRow() {
  return (
    <div className="fi-row fi-skeleton">
      <div className="fi-label" />
      <div className="fi-bar-track" />
      <div className="fi-value" />
    </div>
  );
}

export function FeatureImportancePanel({ items, loading, error }: Props) {
  const maxShap = items.length > 0 ? items[0].mean_abs_shap : 1;

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-header-icon">🧠</span>
        <span className="card-header-title">Key Churn Drivers</span>
        <span className="card-header-desc">Mean |SHAP| value</span>
      </div>
      <div className="card-body">
        {loading && (
          <div className="fi-list">
            {Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)}
          </div>
        )}

        {error && (
          <div className="error-banner">⚠ {error}</div>
        )}

        {!loading && !error && items.length > 0 && (
          <div className="fi-list">
            {items.map((item) => {
              const widthPct = (item.mean_abs_shap / maxShap) * 100;
              return (
                <div key={item.feature} className="fi-row">
                  <span className="fi-label" title={item.feature}>
                    {formatFeatureName(item.feature)}
                  </span>
                  <div className="fi-bar-track">
                    <div
                      className="fi-bar-fill"
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  <span className="fi-value">{item.mean_abs_shap.toFixed(4)}</span>
                </div>
              );
            })}
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <p style={{ color: "var(--text-dim)", fontSize: "0.82rem", textAlign: "center", padding: "1.5rem 0" }}>
            Feature importances not available. Run <code>python ml/train.py</code> first.
          </p>
        )}
      </div>
    </div>
  );
}
