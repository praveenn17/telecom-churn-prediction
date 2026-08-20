import { useEffect, useState } from "react";
import "./index.css";
import { predictChurn, fetchFeatureImportance } from "./api";
import type { CustomerFeatures, FeatureImportanceItem, PredictionResult } from "./types";
import { ChurnForm } from "./components/ChurnForm";
import { ResultPanel } from "./components/ResultPanel";
import { FeatureImportancePanel } from "./components/FeatureImportancePanel";
import { BatchPrediction } from "./components/BatchPrediction";
import { WhatIfSimulator } from "./components/WhatIfSimulator";

type TabMode = "single" | "batch";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabMode>("single");

  // ── Prediction state ──────────────────────────────────────────────────────
  const [originalResult, setOriginalResult] = useState<PredictionResult | null>(null);
  const [displayResult, setDisplayResult] = useState<PredictionResult | null>(null);
  const [lastSubmittedFeatures, setLastSubmittedFeatures] = useState<CustomerFeatures | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);

  // ── Feature importance state ──────────────────────────────────────────────
  const [fiItems, setFiItems] = useState<FeatureImportanceItem[]>([]);
  const [fiLoading, setFiLoading] = useState(true);
  const [fiError, setFiError] = useState<string | null>(null);

  // Fetch feature importances once on mount
  useEffect(() => {
    setFiLoading(true);
    fetchFeatureImportance()
      .then(setFiItems)
      .catch((e: Error) => setFiError(e.message))
      .finally(() => setFiLoading(false));
  }, []);

  const handlePredict = async (features: CustomerFeatures) => {
    setPredLoading(true);
    setPredError(null);
    setOriginalResult(null);
    setDisplayResult(null);
    setIsSimulating(false);
    try {
      const res = await predictChurn(features);
      setLastSubmittedFeatures(features);
      setOriginalResult(res);
      setDisplayResult(res);
    } catch (e: unknown) {
      setPredError((e as Error).message ?? "Unknown error");
    } finally {
      setPredLoading(false);
    }
  };

  const handleSimulationChange = (simResult: PredictionResult, simulating: boolean) => {
    setDisplayResult(simResult);
    setIsSimulating(simulating);
  };

  return (
    <div className="app">
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="header-logo">
            <div className="header-logo-icon">📡</div>
            <div>
              <div className="header-title">ChurnPredict</div>
              <div className="header-subtitle">Telecom Customer Intelligence</div>
            </div>
          </div>

          {/* Tab Navigation in Header */}
          <nav className="header-nav-tabs">
            <button
              type="button"
              className={`nav-tab-btn ${activeTab === "single" ? "nav-tab-active" : ""}`}
              onClick={() => setActiveTab("single")}
            >
              <span>👤</span> Single Customer
            </button>
            <button
              type="button"
              className={`nav-tab-btn ${activeTab === "batch" ? "nav-tab-active" : ""}`}
              onClick={() => setActiveTab("batch")}
            >
              <span>📁</span> Batch Evaluation
            </button>
          </nav>

          <span className="header-badge">ML Powered</span>
        </div>
      </header>

      {/* ── Main Content ─────────────────────────────────────────────────── */}
      {activeTab === "single" ? (
        <main className="main">
          {/* Left column: form */}
          <ChurnForm
            onSubmit={handlePredict}
            loading={predLoading}
            error={predError}
          />

          {/* Right column: result + what-if simulator + feature importance */}
          <div className="result-panel">
            <ResultPanel
              result={displayResult}
              isSimulating={isSimulating}
              originalResult={originalResult}
            />

            {/* What-If Simulator — only shown after a real prediction */}
            {lastSubmittedFeatures && originalResult && (
              <WhatIfSimulator
                originalFeatures={lastSubmittedFeatures}
                originalResult={originalResult}
                onSimulatedResult={handleSimulationChange}
              />
            )}

            <FeatureImportancePanel
              items={fiItems}
              loading={fiLoading}
              error={fiError}
            />
          </div>
        </main>
      ) : (
        <main className="main-batch">
          <div className="batch-layout">
            <BatchPrediction />
            <div className="batch-sidebar">
              <FeatureImportancePanel
                items={fiItems}
                loading={fiLoading}
                error={fiError}
              />
            </div>
          </div>
        </main>
      )}

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="footer">
        ChurnPredict · Telecom Customer Churn Prediction ·{" "}
        <span>Powered by LogisticRegression + SHAP</span>
      </footer>
    </div>
  );
}


