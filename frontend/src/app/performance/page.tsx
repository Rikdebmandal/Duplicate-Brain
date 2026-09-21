"use client";

import { useEffect, useState } from "react";

import { CalibrationChart, LineChart } from "@/components/charts";
import {
  Banner,
  Button,
  Card,
  Empty,
  Shell,
  Spinner,
  Stat,
} from "@/components/ui";
import { api } from "@/lib/api";
import type { Evaluation, Performance } from "@/lib/types";

export default function PerformancePage() {
  return (
    <Shell>
      <PerformanceView />
    </Shell>
  );
}

function PerformanceView() {
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  useEffect(() => {
    api
      .performance()
      .then(setPerformance)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <Banner tone="error">{error}</Banner>;
  if (!performance) return <Spinner label="Loading performance" />;

  const metrics = performance.metrics;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-ink-200">Model performance</h1>
        <p className="mt-1 max-w-2xl text-sm leading-relaxed text-ink-400">
          Measured against what actually happened. Accuracy is reported, but the model is
          judged on calibration: a twin that says 70% and is right 70% of the time is more
          useful than one that says 95% and is wrong when it matters.
        </p>
      </div>

      {performance.n === 0 ? (
        <Empty>
          {performance.message ??
            "No predictions have been checked against reality yet. Make a prediction, then record what you actually decided."}
        </Empty>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="Predictions checked"
              value={String(performance.n)}
              hint="Each one has a recorded real outcome"
            />
            <Stat
              label="Accuracy"
              value={`${((metrics?.accuracy ?? 0) * 100).toFixed(0)}%`}
              hint={`Base-rate baseline: ${((metrics?.baseline_accuracy ?? 0) * 100).toFixed(0)}%`}
              tone={
                (metrics?.accuracy ?? 0) > (metrics?.baseline_accuracy ?? 0)
                  ? "positive"
                  : "caution"
              }
            />
            <Stat
              label="Brier score"
              value={(metrics?.brier ?? 0).toFixed(3)}
              hint="Lower is better. Penalises confident mistakes."
            />
            <Stat
              label="Calibration error"
              value={(metrics?.ece ?? 0).toFixed(3)}
              hint="Gap between stated and actual reliability"
              tone={(metrics?.ece ?? 1) < 0.1 ? "positive" : "caution"}
            />
          </div>

          {performance.interpretation && (
            <Banner tone="info">{performance.interpretation}</Banner>
          )}

          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
            <Card
              title="Accuracy over time"
              subtitle="Running accuracy as outcomes accumulate, against the base-rate baseline."
            >
              <LineChart
                yLabel="accuracy"
                reference={metrics?.baseline_accuracy}
                series={[
                  {
                    name: "running accuracy",
                    color: "#5b8def",
                    points: performance.timeline.map((row, index) => ({
                      x: index,
                      y: row.running_accuracy,
                    })),
                  },
                  {
                    name: "probability assigned to reality",
                    color: "#3fb984",
                    points: performance.timeline.map((row, index) => ({
                      x: index,
                      y: row.probability_of_actual,
                    })),
                  },
                ]}
              />
              <div className="mt-3 flex flex-wrap gap-4 text-xs text-ink-600">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-5 bg-accent" /> running accuracy
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-5 bg-positive" /> probability it gave to what
                  you actually did
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-5 border-t border-dashed border-ink-400" />{" "}
                  baseline
                </span>
              </div>
            </Card>

            <Card
              title="Calibration"
              subtitle="Points on the diagonal mean the stated probabilities can be taken at face value."
            >
              {metrics && metrics.calibration_bins.length > 0 ? (
                <>
                  <CalibrationChart bins={metrics.calibration_bins} />
                  <p className="mt-3 text-xs leading-relaxed text-ink-600">
                    Above the line means under-confident; below means over-confident. The
                    fitted calibration temperature (
                    <span className="font-mono">
                      {(performance.calibration_temperature ?? 1).toFixed(2)}
                    </span>
                    ) corrects for this automatically as outcomes accumulate.
                  </p>
                </>
              ) : (
                <Empty>Not enough outcomes to plot a reliability diagram yet.</Empty>
              )}
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Full metrics">
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2.5 text-sm">
                {[
                  ["Accuracy", metrics?.accuracy],
                  ["Precision", metrics?.precision],
                  ["Recall", metrics?.recall],
                  ["F1", metrics?.f1],
                  ["ROC-AUC", metrics?.roc_auc],
                  ["Brier", metrics?.brier],
                  ["Log loss", metrics?.log_loss],
                  ["ECE", metrics?.ece],
                  ["Mean confidence", metrics?.mean_confidence],
                ].map(([label, value]) => (
                  <div key={label as string} className="flex justify-between gap-3">
                    <dt className="text-ink-400">{label as string}</dt>
                    <dd className="font-mono text-ink-200">
                      {value === null || value === undefined
                        ? "n/a"
                        : (value as number).toFixed(3)}
                    </dd>
                  </div>
                ))}
              </dl>

              {metrics && Object.keys(metrics.by_category).length > 0 && (
                <div className="mt-5 border-t border-ink-800 pt-4">
                  <div className="mb-2 text-xs uppercase tracking-wide text-ink-600">
                    By scenario category
                  </div>
                  <div className="space-y-1.5">
                    {Object.entries(metrics.by_category).map(([category, stats]) => (
                      <div
                        key={category}
                        className="flex justify-between gap-3 text-xs text-ink-400"
                      >
                        <span>
                          {category}{" "}
                          <span className="text-ink-600">(n={stats.n})</span>
                        </span>
                        <span className="font-mono">
                          acc {stats.accuracy.toFixed(2)} &middot; brier{" "}
                          {stats.brier.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>

            <Card title="Prediction vs reality" subtitle="Most recent first.">
              <ul className="divide-y divide-ink-800">
                {[...performance.timeline].reverse().slice(0, 10).map((row, index) => (
                  <li key={index} className="py-2.5 first:pt-0">
                    <div className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="truncate text-ink-400">
                        predicted{" "}
                        <span className="text-ink-200">{row.predicted_option}</span>
                      </span>
                      <span
                        className={`shrink-0 text-xs ${
                          row.was_correct ? "text-positive" : "text-negative"
                        }`}
                      >
                        {row.was_correct ? "match" : "miss"}
                      </span>
                    </div>
                    {!row.was_correct && (
                      <div className="mt-0.5 text-xs text-ink-600">
                        actually: {row.actual_option}
                      </div>
                    )}
                    <div className="mt-1 flex gap-3 font-mono text-[11px] text-ink-600">
                      <span>p={row.predicted_probability.toFixed(2)}</span>
                      <span>p(actual)={row.probability_of_actual.toFixed(2)}</span>
                      <span>brier={row.brier_score.toFixed(2)}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        </>
      )}

      <Card
        title="Offline comparison"
        subtitle="Splits your history by time and asks whether each approach can predict the later block from the earlier one."
        action={
          <Button
            disabled={evaluating}
            onClick={async () => {
              setEvaluating(true);
              setEvaluation(await api.evaluate().catch(() => null));
              setEvaluating(false);
            }}
          >
            {evaluating ? "Running..." : "Run evaluation"}
          </Button>
        }
      >
        {!evaluation ? (
          <Empty>
            Compare the base-rate baseline, the trait model, the statistical model, case
            retrieval, and the hybrid that pools them.
          </Empty>
        ) : evaluation.status !== "ok" ? (
          <Banner tone="warn">{evaluation.message}</Banner>
        ) : (
          <>
            <div className="mb-4 flex flex-wrap gap-4 text-xs text-ink-600">
              <span>
                train {evaluation.split?.train.n} &middot; validation{" "}
                {evaluation.split?.validation.n} &middot; test {evaluation.split?.test.n}
              </span>
              <span>
                base rate of taking the active option{" "}
                {((evaluation.approach_base_rate ?? 0) * 100).toFixed(0)}%
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[34rem] text-sm">
                <thead>
                  <tr className="border-b border-ink-800 text-xs uppercase tracking-wide text-ink-600">
                    <th className="py-2 text-left font-medium">Approach</th>
                    <th className="py-2 text-right font-medium">Accuracy</th>
                    <th className="py-2 text-right font-medium">Brier</th>
                    <th className="py-2 text-right font-medium">Log loss</th>
                    <th className="py-2 text-right font-medium">AUC</th>
                    <th className="py-2 text-right font-medium">ECE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-ink-800/60">
                  {Object.entries(evaluation.results ?? {}).map(([name, row]) => (
                    <tr
                      key={name}
                      className={
                        name === evaluation.best_by_brier ? "text-accent" : "text-ink-400"
                      }
                    >
                      <td className="py-2 capitalize">
                        {name}
                        {name === evaluation.best_by_brier && (
                          <span className="ml-2 text-[10px] uppercase tracking-wide">
                            best
                          </span>
                        )}
                      </td>
                      <td className="py-2 text-right font-mono">
                        {row.accuracy.toFixed(3)}
                      </td>
                      <td className="py-2 text-right font-mono">{row.brier.toFixed(3)}</td>
                      <td className="py-2 text-right font-mono">
                        {row.log_loss.toFixed(3)}
                      </td>
                      <td className="py-2 text-right font-mono">
                        {row.roc_auc === null ? "n/a" : row.roc_auc.toFixed(3)}
                      </td>
                      <td className="py-2 text-right font-mono">{row.ece.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="mt-4 text-xs leading-relaxed text-ink-600">{evaluation.note}</p>
          </>
        )}
      </Card>
    </div>
  );
}
