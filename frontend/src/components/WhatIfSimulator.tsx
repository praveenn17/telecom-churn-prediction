import { useState, useEffect, useRef, useMemo } from "react";
import { predictChurn } from "../api";
import type { CustomerFeatures, PredictionResult } from "../types";

interface Props {
  originalFeatures: CustomerFeatures;
  originalResult: PredictionResult;
  onSimulatedResult: (result: PredictionResult, isSimulating: boolean) => void;
}

export function WhatIfSimulator({
  originalFeatures,
  originalResult,
  onSimulatedResult,
}: Props) {
  const [tenure, setTenure] = useState<number>(originalFeatures.tenure);
  const [monthlyCharges, setMonthlyCharges] = useState<number>(originalFeatures.MonthlyCharges);
  const [contract, setContract] = useState<CustomerFeatures["Contract"]>(originalFeatures.Contract);
  const [loading, setLoading] = useState(false);

  // Sync state whenever a new original prediction is submitted
  useEffect(() => {
    setTenure(originalFeatures.tenure);
    setMonthlyCharges(originalFeatures.MonthlyCharges);
    setContract(originalFeatures.Contract);
  }, [originalFeatures]);

  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Check what has changed compared to the submitted baseline
  const changes = useMemo(() => {
    const list: string[] = [];
    if (tenure !== originalFeatures.tenure) {
      list.push(`Tenure: ${originalFeatures.tenure}mo → ${tenure}mo`);
    }
    if (Math.abs(monthlyCharges - originalFeatures.MonthlyCharges) > 0.01) {
      list.push(
        `Monthly: $${originalFeatures.MonthlyCharges.toFixed(2)} → $${monthlyCharges.toFixed(2)}`
      );
    }
    if (contract !== originalFeatures.Contract) {
      list.push(`Contract: ${originalFeatures.Contract} → ${contract}`);
    }
    return list;
  }, [tenure, monthlyCharges, contract, originalFeatures]);

  const isSimulating = changes.length > 0;

  // Run debounced prediction when controls change
  useEffect(() => {
    if (!isSimulating) {
      onSimulatedResult(originalResult, false);
      return;
    }

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        // Estimate updated TotalCharges based on new tenure & monthly rate
        const estimatedTotal =
          tenure === 0
            ? monthlyCharges
            : Math.round(tenure * monthlyCharges * 100) / 100;

        const simulatedFeatures: CustomerFeatures = {
          ...originalFeatures,
          tenure,
          MonthlyCharges: monthlyCharges,
          Contract: contract,
          TotalCharges: estimatedTotal,
        };

        const res = await predictChurn(simulatedFeatures);
        onSimulatedResult(res, true);
      } catch (err) {
        console.error("What-if simulation prediction failed:", err);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [tenure, monthlyCharges, contract, isSimulating]);

  const handleReset = () => {
    setTenure(originalFeatures.tenure);
    setMonthlyCharges(originalFeatures.MonthlyCharges);
    setContract(originalFeatures.Contract);
    onSimulatedResult(originalResult, false);
  };

  return (
    <div className={`card whatif-card ${isSimulating ? "whatif-active" : ""}`}>
      <div className="card-header whatif-header">
        <div className="whatif-title-wrap">
          <span className="card-header-icon">🎛️</span>
          <span className="card-header-title">What-If Retention Simulator</span>
          {isSimulating && <span className="whatif-pill">Active Simulation</span>}
        </div>
        {isSimulating && (
          <button
            type="button"
            className="whatif-reset-btn"
            onClick={handleReset}
            title="Revert back to original submitted values"
          >
            Reset to Original ↺
          </button>
        )}
      </div>

      <div className="card-body whatif-body">
        <p className="whatif-desc">
          Adjust high-impact levers to test retention strategies in real-time.
        </p>

        {/* ── Control 1: Tenure Slider ─────────────────────────────────── */}
        <div className="whatif-control-group">
          <div className="whatif-control-header">
            <label htmlFor="whatif-tenure" className="whatif-label">
              Tenure Duration
            </label>
            <span className="whatif-val-badge">{tenure} months</span>
          </div>
          <input
            id="whatif-tenure"
            type="range"
            min={0}
            max={72}
            step={1}
            value={tenure}
            onChange={(e) => setTenure(Number(e.target.value))}
            className="whatif-slider"
          />
          <div className="whatif-slider-ticks">
            <span>0 mo</span>
            <span>12 mo</span>
            <span>24 mo</span>
            <span>36 mo</span>
            <span>48 mo</span>
            <span>72 mo</span>
          </div>
        </div>

        {/* ── Control 2: Monthly Charges Slider ────────────────────────── */}
        <div className="whatif-control-group">
          <div className="whatif-control-header">
            <label htmlFor="whatif-monthly" className="whatif-label">
              Monthly Charges
            </label>
            <span className="whatif-val-badge">${monthlyCharges.toFixed(2)}/mo</span>
          </div>
          <input
            id="whatif-monthly"
            type="range"
            min={18}
            max={120}
            step={0.5}
            value={monthlyCharges}
            onChange={(e) => setMonthlyCharges(Number(e.target.value))}
            className="whatif-slider"
          />
          <div className="whatif-slider-ticks">
            <span>$18</span>
            <span>$40</span>
            <span>$65</span>
            <span>$90</span>
            <span>$120</span>
          </div>
        </div>

        {/* ── Control 3: Contract Type Dropdown ────────────────────────── */}
        <div className="whatif-control-group">
          <div className="whatif-control-header">
            <label htmlFor="whatif-contract" className="whatif-label">
              Contract Commitment
            </label>
          </div>
          <select
            id="whatif-contract"
            className="field-select whatif-select"
            value={contract}
            onChange={(e) =>
              setContract(e.target.value as CustomerFeatures["Contract"])
            }
          >
            <option value="Month-to-month">Month-to-month (High churn rate)</option>
            <option value="One year">One year commitment</option>
            <option value="Two year">Two year commitment (Best retention)</option>
          </select>
        </div>

        {/* ── Active Simulation Banner ─────────────────────────────────── */}
        {isSimulating && (
          <div className="whatif-scenario-banner">
            <div className="whatif-scenario-icon">
              {loading ? <div className="spinner spinner-cyan" /> : "💡"}
            </div>
            <div className="whatif-scenario-text">
              <span className="whatif-scenario-prefix">
                {loading ? "Calculating..." : "Simulating Scenario:"}
              </span>
              <span className="whatif-scenario-changes">
                {changes.join(" · ")}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
