import type { PredictionResult, TopFactor } from "../types";

interface Props {
  result: PredictionResult | null;
  isSimulating?: boolean;
  originalResult?: PredictionResult | null;
}

const CIRCUMFERENCE = 2 * Math.PI * 54; // r=54

function riskClass(risk: string) {
  return (
    { Low: "risk-low", Medium: "risk-medium", High: "risk-high", "Very High": "risk-very-high" }[
      risk
    ] ?? "risk-low"
  );
}

function gaugeColor(prob: number) {
  if (prob < 0.3) return "#4ade80";
  if (prob < 0.6) return "#facc15";
  if (prob < 0.8) return "#fb923c";
  return "#f87171";
}

/** Strip sklearn OHE suffix noise; keep the meaningful part readable. */
function formatFeatureName(raw: string): string {
  const underscore = raw.indexOf("_");
  if (underscore === -1) {
    return raw.charAt(0).toUpperCase() + raw.slice(1);
  }
  const base  = raw.slice(0, underscore);
  const value = raw.slice(underscore + 1);
  return `${value} (${base})`;
}

function WaterfallRow({ factor, maxAbs }: { factor: TopFactor; maxAbs: number }) {
  const isUp   = factor.direction === "increases_risk";
  const dir    = isUp ? "up" : "down";
  const sign   = isUp ? "+" : "−";
  const pctStr = `${sign}${(Math.abs(factor.impact) * 100).toFixed(1)}%`;
  const barW   = Math.round((Math.abs(factor.impact) / maxAbs) * 100);

  return (
    <div className="why-row">
      <div className="why-left">
        <div className={`why-indicator ${dir}`} />
        <div className="why-bar-wrap">
          <div className="why-feat-name" title={factor.feature}>
            {formatFeatureName(factor.feature)}
          </div>
          <div className="why-bar-track">
            <div className={`why-bar-fill ${dir}`} style={{ width: `${barW}%` }} />
          </div>
        </div>
      </div>
      <span className={`why-impact ${dir}`}>{pctStr}</span>
    </div>
  );
}

export function ResultPanel({ result, isSimulating, originalResult }: Props) {
  if (!result) {
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-header-icon">📊</span>
          <span className="card-header-title">Prediction Result</span>
        </div>
        <div className="empty-result">
          <div className="empty-result-icon">🔮</div>
          <p className="empty-result-text">
            Fill in the customer form and click <strong>Predict Churn Risk</strong> to see
            results here.
          </p>
        </div>
      </div>
    );
  }

  const prob   = result.churn_probability;
  const pct    = Math.round(prob * 100);
  const offset = CIRCUMFERENCE * (1 - prob);
  const color  = gaugeColor(prob);

  const origProb = originalResult?.churn_probability ?? null;
  const diffPct =
    isSimulating && origProb !== null
      ? Math.round((prob - origProb) * 100)
      : null;

  const factors = result.top_factors ?? [];
  const maxAbs  = factors.length > 0 ? Math.max(...factors.map((f) => Math.abs(f.impact))) : 1;

  return (
    <div className={`card gauge-card ${isSimulating ? "card-simulated" : ""}`}>
      <div className="card-header">
        <span className="card-header-icon">{isSimulating ? "🎛️" : "📊"}</span>
        <span className="card-header-title">
          {isSimulating ? "Simulated Prediction" : "Prediction Result"}
        </span>
        {isSimulating ? (
          <span className="sim-live-badge">Simulated Result</span>
        ) : (
          <span className="card-header-desc">Based on model output</span>
        )}
      </div>
      <div className="card-body">


        {/* ── Probability ring ────────────────────────────────────────── */}
        <div className="gauge-ring-wrap">
          <svg className="gauge-ring-svg" width="160" height="160" viewBox="0 0 120 120">
            <circle className="gauge-track" cx="60" cy="60" r="54" />
            <circle
              className="gauge-fill"
              cx="60" cy="60" r="54"
              stroke={color}
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="gauge-center">
            <span className="gauge-pct">{pct}%</span>
            <span className="gauge-label">Churn Prob.</span>
          </div>
        </div>

        {/* ── Prediction + risk badges ─────────────────────────────────── */}
        <div className="badge-row">
          <span
            className={`prediction-badge ${
              result.prediction === "Churn" ? "badge-churn" : "badge-no-churn"
            }`}
          >
            {result.prediction === "Churn" ? "⚠ Churn" : "✓ No Churn"}
          </span>
          <span className={`risk-badge ${riskClass(result.risk_level)}`}>
            {result.risk_level} Risk
          </span>
        </div>

        {/* ── Stats ───────────────────────────────────────────────────── */}
        <div className="stats-row">
          <div className="stat-box">
            <div className="stat-value" style={{ color }}>
              {(prob * 100).toFixed(1)}%
              {diffPct !== null && (
                <span
                  className={`stat-diff-badge ${
                    diffPct <= 0 ? "stat-diff-good" : "stat-diff-bad"
                  }`}
                >
                  {diffPct > 0 ? `+${diffPct}%` : `${diffPct}%`}
                </span>
              )}
            </div>
            <div className="stat-label">
              {isSimulating ? "Simulated Probability" : "Churn Probability"}
            </div>
          </div>
          <div className="stat-box">
            <div
              className="stat-value"
              style={{ color: result.prediction === "Churn" ? "#f87171" : "#4ade80" }}
            >
              {result.prediction}
            </div>
            <div className="stat-label">Prediction</div>
          </div>
        </div>

        {/* ── Why this prediction ──────────────────────────────────────── */}
        {factors.length > 0 && (
          <>
            <div className="why-divider" />
            <div className="why-header">
              <span className="why-header-icon">⚡</span>
              Why this prediction
            </div>
            <div className="why-list">
              {factors.map((f) => (
                <WaterfallRow key={f.feature} factor={f} maxAbs={maxAbs} />
              ))}
            </div>
          </>
        )}

      </div>
    </div>
  );
}

