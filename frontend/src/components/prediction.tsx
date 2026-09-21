"use client";

import { useEffect, useMemo, useState } from "react";

import { SweepChart } from "@/components/charts";
import {
  Banner,
  Button,
  Card,
  ConfidenceBadge,
  ContributionBar,
  Empty,
  Field,
  ProbabilityBar,
  inputClass,
} from "@/components/ui";
import { api } from "@/lib/api";
import type {
  CounterfactualResult,
  FactorMeta,
  Landscape,
  Prediction,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// headline result
// ---------------------------------------------------------------------------

export function PredictionResult({ prediction }: { prediction: Prediction }) {
  const top = prediction.explanation.important_factors;
  const maxContribution = Math.max(0.01, ...top.map((f) => Math.abs(f.contribution)));

  return (
    <div className="space-y-6">
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-xs uppercase tracking-wide text-ink-400">
              Predicted decision
            </div>
            <div className="mt-1 text-2xl font-semibold text-ink-200">
              {prediction.predicted_option}
            </div>
            <div className="mt-1 font-mono text-sm text-accent">
              {(prediction.predicted_probability * 100).toFixed(0)}% probability
            </div>
          </div>
          <div className="flex flex-col items-end gap-2">
            <ConfidenceBadge
              label={prediction.confidence_label}
              score={prediction.confidence_score}
            />
            <span className="text-xs text-ink-600">
              from {prediction.evidence_count} recorded decisions
            </span>
          </div>
        </div>

        <div className="mt-6 space-y-3">
          {prediction.options.map((option) => (
            <ProbabilityBar
              key={option.label}
              label={option.label}
              value={option.probability}
              highlight={option.label === prediction.predicted_option}
            />
          ))}
        </div>

        <p className="mt-6 border-t border-ink-800 pt-4 text-sm leading-relaxed text-ink-400">
          {prediction.explanation.summary}
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Important factors"
          subtitle={`Signed contribution to "${prediction.predicted_option}". Right of centre pushes towards it, left pushes away.`}
        >
          {top.length === 0 ? (
            <Empty>
              No decision factors were confidently detected in this text. The estimate
              rests almost entirely on your general profile.
            </Empty>
          ) : (
            <div className="divide-y divide-ink-800/60">
              {top.map((factor) => (
                <ContributionBar
                  key={factor.factor}
                  label={factor.label}
                  value={factor.contribution}
                  magnitude={factor.magnitude}
                  symbol={factor.symbol}
                  max={maxContribution}
                  evidence={factor.evidence_phrases}
                />
              ))}
            </div>
          )}
          <p className="mt-4 text-xs text-ink-600">
            The number on the right is how strongly the factor is present in the situation;
            the bar is how much it moved this particular person.
          </p>
        </Card>

        <Card
          title="Similar decisions you actually made"
          subtitle="Ranked by how alike the situation is, in wording and in the trade-off it poses."
        >
          {prediction.similar_decisions.length === 0 ? (
            <Empty>
              Nothing comparable is on record yet, so case-based evidence contributed
              nothing to this estimate.
            </Empty>
          ) : (
            <ul className="space-y-4">
              {prediction.similar_decisions.slice(0, 4).map((neighbour, index) => (
                <li key={neighbour.decision_id} className="border-l-2 border-ink-700 pl-3">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-xs font-medium text-ink-400">
                      Decision #{index + 1}
                    </span>
                    <span className="font-mono text-xs text-accent">
                      {(neighbour.similarity * 100).toFixed(0)}% similar
                    </span>
                  </div>
                  <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-ink-400">
                    {neighbour.scenario_text}
                  </p>
                  <p className="mt-1.5 text-sm text-ink-200">
                    Chose: <span className="text-ink-200">{neighbour.chosen_option}</span>
                  </p>
                  {neighbour.reason && (
                    <p className="mt-1 text-xs italic leading-relaxed text-ink-600">
                      &ldquo;{neighbour.reason}&rdquo;
                    </p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Reasoning trace" subtitle="How the number was produced, step by step.">
          <ol className="space-y-3">
            {prediction.explanation.reasoning_trace.map((step) => (
              <li key={step.step} className="flex gap-3">
                <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded bg-ink-800 font-mono text-[10px] text-ink-400">
                  {step.step}
                </span>
                <div>
                  <div className="text-xs font-medium text-ink-200">{step.title}</div>
                  <div className="mt-0.5 text-xs leading-relaxed text-ink-400">
                    {step.detail}
                  </div>
                </div>
              </li>
            ))}
          </ol>

          {prediction.explanation.llm_reasoning && (
            <div className="mt-5 rounded-lg border border-accent-dim bg-accent/5 p-3">
              <div className="mb-1.5 text-xs font-medium text-accent">
                Narrative reasoning layer
              </div>
              <p className="text-xs leading-relaxed text-ink-400">
                {prediction.explanation.llm_reasoning}
              </p>
            </div>
          )}
        </Card>

        <Card title="Evidence" subtitle="Kept separate on purpose: what was counted, versus what was estimated.">
          <div className="space-y-4">
            <div>
              <div className="mb-2 flex items-center gap-2">
                <span className="rounded bg-positive/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-positive">
                  Observed
                </span>
                <span className="text-xs text-ink-600">counts over your record</span>
              </div>
              <ul className="space-y-1.5">
                {prediction.explanation.observed.map((statement, index) => (
                  <li key={index} className="text-xs leading-relaxed text-ink-400">
                    {statement}
                  </li>
                ))}
              </ul>
            </div>

            <div className="border-t border-ink-800 pt-4">
              <div className="mb-2 flex items-center gap-2">
                <span className="rounded bg-caution/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-caution">
                  Inferred
                </span>
                <span className="text-xs text-ink-600">estimates, with uncertainty</span>
              </div>
              <ul className="space-y-1.5">
                {prediction.explanation.inferred.map((statement) => (
                  <li key={statement.trait} className="text-xs leading-relaxed text-ink-400">
                    {statement.statement}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      </div>

      <Card title="How the layers voted" subtitle="Each layer is weighted by how much evidence it actually had.">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {prediction.layers.map((layer) => (
            <div
              key={layer.name}
              className={`rounded-lg border p-3 ${
                layer.available ? "border-ink-700 bg-ink-950" : "border-ink-800 bg-ink-900/40"
              }`}
            >
              <div className="flex items-baseline justify-between">
                <span className="text-xs font-medium capitalize text-ink-200">
                  {layer.name}
                </span>
                <span className="font-mono text-xs text-ink-400">
                  {layer.available ? `${(layer.weight * 100).toFixed(0)}%` : "—"}
                </span>
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-ink-600">
                {layer.available
                  ? (layer.detail.description as string) ?? ""
                  : `Not used: ${layer.detail.reason ?? "unavailable"}`}
              </p>
              {layer.available && layer.probabilities.length > 0 && (
                <div className="mt-2 flex gap-1">
                  {layer.probabilities.map((p, i) => (
                    <div
                      key={i}
                      className="h-1 flex-1 rounded-full bg-ink-800"
                      title={`${prediction.options[i]?.label}: ${(p * 100).toFixed(0)}%`}
                    >
                      <div
                        className="h-full rounded-full bg-accent/70"
                        style={{ width: `${p * 100}%` }}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// counterfactual simulator
// ---------------------------------------------------------------------------

export function ScenarioSimulator({
  prediction,
  factors,
}: {
  prediction: Prediction;
  factors: FactorMeta[];
}) {
  const [values, setValues] = useState<Record<string, number>>(prediction.factors);
  const [result, setResult] = useState<CounterfactualResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setValues(prediction.factors);
    setResult(null);
  }, [prediction]);

  const changed = useMemo(
    () =>
      Object.entries(values).filter(
        ([key, value]) => Math.abs(value - (prediction.factors[key] ?? 0)) > 0.005,
      ),
    [values, prediction.factors],
  );

  async function run() {
    if (changed.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.counterfactual(
          prediction.id,
          Object.fromEntries(changed),
          false,
        ),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not run the simulation.");
    } finally {
      setBusy(false);
    }
  }

  // Only factors that are actually present, plus any the user has moved, to
  // avoid a wall of twelve sliders sitting at zero.
  const visible = factors.filter(
    (factor) =>
      (prediction.factors[factor.key] ?? 0) > 0.05 ||
      Math.abs((values[factor.key] ?? 0) - (prediction.factors[factor.key] ?? 0)) > 0.005,
  );
  const hidden = factors.filter((f) => !visible.includes(f));
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? factors : visible;

  return (
    <Card
      title="Scenario simulator"
      subtitle="Change the conditions and see what it would take to shift the answer."
      action={
        <div className="flex gap-2">
          {changed.length > 0 && (
            <Button
              variant="ghost"
              onClick={() => {
                setValues(prediction.factors);
                setResult(null);
              }}
            >
              Reset
            </Button>
          )}
          <Button onClick={run} disabled={busy || changed.length === 0}>
            {busy ? "Recalculating..." : `Recalculate${changed.length ? ` (${changed.length})` : ""}`}
          </Button>
        </div>
      }
    >
      {error && (
        <div className="mb-4">
          <Banner tone="error">{error}</Banner>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-3">
          {shown.map((factor) => {
            const original = prediction.factors[factor.key] ?? 0;
            const current = values[factor.key] ?? 0;
            const moved = Math.abs(current - original) > 0.005;
            return (
              <div key={factor.key}>
                <div className="mb-1 flex items-baseline justify-between gap-2">
                  <span className="text-xs text-ink-400" title={factor.description}>
                    {factor.label}
                  </span>
                  <span className="font-mono text-xs">
                    {moved && (
                      <span className="mr-1.5 text-ink-600 line-through">
                        {original.toFixed(2)}
                      </span>
                    )}
                    <span className={moved ? "text-accent" : "text-ink-600"}>
                      {current.toFixed(2)}
                    </span>
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={current}
                  onChange={(e) =>
                    setValues((prev) => ({
                      ...prev,
                      [factor.key]: Number(e.target.value),
                    }))
                  }
                />
              </div>
            );
          })}

          {hidden.length > 0 && (
            <button
              onClick={() => setShowAll((v) => !v)}
              className="text-xs text-accent hover:underline"
            >
              {showAll
                ? "Show only the factors present here"
                : `Show all ${factors.length} factors (${hidden.length} not detected)`}
            </button>
          )}
        </div>

        <div>
          {result ? (
            <div className="space-y-4">
              <div className="space-y-3">
                {prediction.options.map((option) => (
                  <ProbabilityBar
                    key={option.label}
                    label={option.label}
                    value={result.modified.probabilities[option.label] ?? 0}
                    compare={result.original.probabilities[option.label] ?? 0}
                    highlight={option.label === result.modified.predicted_option}
                  />
                ))}
              </div>

              <Banner tone={result.flipped ? "warn" : "info"}>
                {result.flipped ? (
                  <>
                    Under these conditions the predicted choice changes from{" "}
                    <strong>{result.original.predicted_option}</strong> to{" "}
                    <strong>{result.modified.predicted_option}</strong>.
                  </>
                ) : (
                  <>
                    The predicted choice stays{" "}
                    <strong>{result.modified.predicted_option}</strong>, though the
                    probabilities move.
                  </>
                )}
              </Banner>

              <p className="text-xs leading-relaxed text-ink-600">
                Only the factors you moved differ between the two runs &mdash; the scenario
                text, the option set and the profile are held fixed, so the change is
                attributable to the sliders alone.
              </p>
            </div>
          ) : (
            <div className="grid h-full place-items-center rounded-lg border border-dashed border-ink-700 p-8 text-center">
              <p className="max-w-xs text-sm text-ink-400">
                Move a slider and recalculate to see how the prediction responds.
              </p>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// decision landscape
// ---------------------------------------------------------------------------

export function DecisionLandscape({ predictionId }: { predictionId: string }) {
  const [landscape, setLandscape] = useState<Landscape | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLandscape(null);
    api
      .landscape(predictionId, 7)
      .then(setLandscape)
      .catch((err) => setError(err.message));
  }, [predictionId]);

  if (error) return <Banner tone="error">{error}</Banner>;
  if (!landscape) return null;

  const active = landscape.options[0];
  const meaningful = landscape.sweeps.filter((s) => s.sensitivity > 0.02).slice(0, 6);

  return (
    <Card
      title="Decision landscape"
      subtitle={`How the probability of "${active}" responds as each factor moves from 0 to 1. The orange line marks where the predicted choice flips.`}
    >
      {meaningful.length === 0 ? (
        <Empty>No single factor moves this prediction much on its own.</Empty>
      ) : (
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {meaningful.map((sweep) => (
            <div key={sweep.factor}>
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="truncate text-xs text-ink-400">{sweep.label}</span>
                <span className="font-mono text-[10px] text-ink-600">
                  now {sweep.current_value.toFixed(2)}
                </span>
              </div>
              <SweepChart
                points={sweep.points.map((p) => ({
                  value: p.value,
                  probability: p.probabilities[active] ?? 0,
                }))}
                optionLabel={active}
                currentValue={sweep.current_value}
                flipAt={
                  sweep.flip_point
                    ? (sweep.flip_point.between[0] + sweep.flip_point.between[1]) / 2
                    : null
                }
              />
              <p className="mt-1 text-[11px] leading-snug text-ink-600">
                {sweep.flip_point
                  ? `Flips to "${sweep.flip_point.to}" around ${sweep.flip_point.between[1].toFixed(2)}.`
                  : "Never flips the choice on its own."}
              </p>
            </div>
          ))}
        </div>
      )}
      <p className="mt-4 text-xs text-ink-600">{landscape.note}</p>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// outcome recorder
// ---------------------------------------------------------------------------

export function OutcomeRecorder({
  prediction,
  onRecorded,
}: {
  prediction: Prediction;
  onRecorded: () => void;
}) {
  const [actual, setActual] = useState(prediction.predicted_option);
  const [reason, setReason] = useState("");
  const [importance, setImportance] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [correct, setCorrect] = useState(false);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.recordOutcome(prediction.id, {
        actual_option: actual,
        reason,
        importance,
        promote_to_history: true,
      });
      setCorrect(result.was_correct);
      setDone(
        `${result.message} The model had assigned ${(
          result.probability_of_actual * 100
        ).toFixed(0)}% to what you actually did (Brier ${result.brier_score.toFixed(3)}).${
          result.retrained ? " The model has been retrained." : ""
        }`,
      );
      onRecorded();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not record the outcome.");
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <Card title="Outcome recorded">
        <Banner tone={correct ? "success" : "warn"}>{done}</Banner>
        <p className="mt-3 text-xs leading-relaxed text-ink-600">
          This situation is now part of your decision history, so it will inform future
          predictions like any other recorded decision.
        </p>
      </Card>
    );
  }

  return (
    <Card
      title="What did you actually decide?"
      subtitle="Recording the real outcome is what makes the model improve rather than just keep score."
    >
      {error && (
        <div className="mb-4">
          <Banner tone="error">{error}</Banner>
        </div>
      )}
      <div className="space-y-4">
        <Field label="Actual decision">
          <div className="space-y-2">
            {prediction.options.map((option) => (
              <label
                key={option.label}
                className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2 text-sm transition ${
                  actual === option.label
                    ? "border-accent bg-accent/10 text-ink-200"
                    : "border-ink-700 text-ink-400 hover:border-ink-600"
                }`}
              >
                <input
                  type="radio"
                  name="actual"
                  checked={actual === option.label}
                  onChange={() => setActual(option.label)}
                  className="h-3.5 w-3.5 accent-accent"
                />
                <span className="flex-1">{option.label}</span>
                <span className="font-mono text-xs text-ink-600">
                  predicted {(option.probability * 100).toFixed(0)}%
                </span>
              </label>
            ))}
          </div>
        </Field>

        <Field label="Why did you choose that?" hint="Optional, but this is the single most useful thing you can record.">
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            rows={3}
            className={inputClass}
            placeholder="What actually decided it for you?"
          />
        </Field>

        <Field label={`How much did this matter? ${importance}/10`}>
          <input
            type="range"
            min={1}
            max={10}
            value={importance}
            onChange={(e) => setImportance(Number(e.target.value))}
          />
        </Field>

        <Button onClick={submit} disabled={busy}>
          {busy ? "Recording..." : "Record outcome"}
        </Button>
      </div>
    </Card>
  );
}
