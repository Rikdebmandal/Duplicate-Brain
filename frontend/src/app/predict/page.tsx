"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";

import {
  DecisionLandscape,
  OutcomeRecorder,
  PredictionResult,
  ScenarioSimulator,
} from "@/components/prediction";
import {
  Banner,
  Button,
  Card,
  Field,
  Shell,
  Spinner,
  inputClass,
} from "@/components/ui";
import { api } from "@/lib/api";
import type { FactorMeta, Prediction, User } from "@/lib/types";

const EXAMPLE = {
  scenario:
    "You receive a job offer paying 2x your current salary, but you must move to another city and the company has only been operating for one year.",
  options: ["accept the offer", "reject the offer"],
};

export default function PredictPage() {
  return (
    <Shell>
      <Suspense fallback={<Spinner />}>
        <Predict />
      </Suspense>
    </Shell>
  );
}

function Predict() {
  const params = useSearchParams();
  const existingId = params.get("id");

  const [scenario, setScenario] = useState("");
  const [options, setOptions] = useState<string[]>(["", ""]);
  const [category, setCategory] = useState("general");
  const [useLlm, setUseLlm] = useState(true);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [factors, setFactors] = useState<FactorMeta[]>([]);
  const [user, setUser] = useState<User | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcomeRecorded, setOutcomeRecorded] = useState(false);

  useEffect(() => {
    api.schema().then((s) => setFactors(s.factors)).catch(() => undefined);
    api.me().then(setUser).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!existingId) return;
    setBusy(true);
    api
      .getPrediction(existingId)
      .then((p) => {
        setPrediction(p);
        // Repopulate the form from the stored scenario so the permalink is
        // editable, not just readable.
        setScenario(p.scenario ?? "");
        setOptions(p.options.map((option) => option.label));
        setCategory(p.category ?? "general");
      })
      .catch((err) => setError(err.message))
      .finally(() => setBusy(false));
  }, [existingId]);

  const submit = useCallback(async () => {
    const cleaned = options.map((o) => o.trim()).filter(Boolean);
    if (scenario.trim().length < 5) {
      setError("Describe the situation in a sentence or two.");
      return;
    }
    if (cleaned.length < 2) {
      setError("Give at least two distinct options.");
      return;
    }
    setBusy(true);
    setError(null);
    setOutcomeRecorded(false);
    try {
      const result = await api.predict({
        scenario: scenario.trim(),
        options: cleaned,
        category,
        use_llm: useLlm,
      });
      setPrediction(result);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }, [scenario, options, category, useLlm]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-ink-200">New scenario</h1>
        <p className="mt-1 max-w-2xl text-sm leading-relaxed text-ink-400">
          Describe a situation and the choices available. The model estimates what you
          would probably do, shows the evidence it used, and lets you change the
          conditions to see what would shift the answer.
        </p>
      </div>

      {error && <Banner tone="error">{error}</Banner>}

      <Card
        title="Describe the situation"
        action={
          <button
            onClick={() => {
              setScenario(EXAMPLE.scenario);
              setOptions(EXAMPLE.options);
              setCategory("career");
            }}
            className="text-xs text-accent hover:underline"
          >
            Use an example
          </button>
        }
      >
        <div className="space-y-4">
          <Field
            label="Situation"
            hint="Include the things that actually matter: what is at stake, what is uncertain, who else is affected."
          >
            <textarea
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              rows={4}
              className={inputClass}
              placeholder="You receive a job offer paying 2x your current salary, but you must move to another city..."
            />
          </Field>

          <div className="grid gap-4 sm:grid-cols-[1fr_12rem]">
            <Field label="Options" hint="Two to eight distinct choices.">
              <div className="space-y-2">
                {options.map((option, index) => (
                  <div key={index} className="flex gap-2">
                    <input
                      value={option}
                      onChange={(e) =>
                        setOptions((prev) =>
                          prev.map((v, i) => (i === index ? e.target.value : v)),
                        )
                      }
                      className={inputClass}
                      placeholder={index === 0 ? "accept the offer" : "reject the offer"}
                    />
                    {options.length > 2 && (
                      <button
                        onClick={() =>
                          setOptions((prev) => prev.filter((_, i) => i !== index))
                        }
                        className="rounded-lg border border-ink-700 px-2.5 text-ink-400 hover:border-negative/50 hover:text-negative"
                        aria-label="Remove option"
                      >
                        &times;
                      </button>
                    )}
                  </div>
                ))}
                {options.length < 8 && (
                  <button
                    onClick={() => setOptions((prev) => [...prev, ""])}
                    className="text-xs text-accent hover:underline"
                  >
                    + Add another option
                  </button>
                )}
              </div>
            </Field>

            <Field label="Category">
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className={inputClass}
              >
                {["general", "career", "finance", "family", "education", "lifestyle", "health", "social", "ethics", "work"].map(
                  (option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ),
                )}
              </select>
            </Field>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-ink-800 pt-4">
            <label
              className={`flex items-center gap-2 text-xs ${
                user?.allow_llm_processing ? "text-ink-400" : "text-ink-600"
              }`}
            >
              <input
                type="checkbox"
                checked={useLlm && !!user?.allow_llm_processing}
                disabled={!user?.allow_llm_processing}
                onChange={(e) => setUseLlm(e.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              Use the narrative reasoning layer
              {!user?.allow_llm_processing && " (not enabled for this account)"}
            </label>
            <Button onClick={submit} disabled={busy}>
              {busy ? "Predicting..." : "Predict"}
            </Button>
          </div>
        </div>
      </Card>

      {busy && !prediction && <Spinner label="Weighing the evidence" />}

      {prediction && (
        <div className="space-y-6">
          <PredictionResult prediction={prediction} />
          <ScenarioSimulator prediction={prediction} factors={factors} />
          <DecisionLandscape predictionId={prediction.id} />
          {!outcomeRecorded && (
            <OutcomeRecorder
              prediction={prediction}
              onRecorded={() => setOutcomeRecorded(true)}
            />
          )}

          <Card title="Known limitations">
            <ul className="space-y-1.5">
              {prediction.explanation.limitations.map((limitation, index) => (
                <li key={index} className="text-xs leading-relaxed text-ink-600">
                  &bull; {limitation}
                </li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  );
}
