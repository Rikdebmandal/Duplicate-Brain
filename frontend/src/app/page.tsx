"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { RadarChart, TimelineChart } from "@/components/charts";
import {
  Banner,
  Card,
  ConfidenceBadge,
  Empty,
  Shell,
  Spinner,
  Stat,
} from "@/components/ui";
import { api } from "@/lib/api";
import type { Analytics, Profile, TimelineEntry } from "@/lib/types";

export default function DashboardPage() {
  return (
    <Shell>
      <Dashboard />
    </Shell>
  );
}

function Dashboard() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.analytics(), api.profile(), api.timeline()])
      .then(([a, p, t]) => {
        setAnalytics(a);
        setProfile(p);
        setTimeline(t.timeline);
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <Banner tone="error">{error}</Banner>;
  if (!analytics || !profile) return <Spinner label="Loading your twin" />;

  const decisions = analytics.counts.decisions;
  const accuracy = analytics.accuracy.accuracy;

  if (decisions === 0) {
    return <GettingStarted />;
  }

  const radar = profile.traits.map((trait) => ({
    label: trait.label.replace(/ (priority|tolerance|orientation|sensitivity|influence|aversion|tendency)$/i, ""),
    value: trait.value,
    confidence: trait.confidence,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-ink-200">Dashboard</h1>
        <p className="mt-1 text-sm text-ink-400">
          Everything below is derived from the {decisions} decisions you have recorded.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Decisions analysed"
          value={String(decisions)}
          hint={
            analytics.model_confidence.ml_ready
              ? "Enough for the statistical layer"
              : `${analytics.model_confidence.decisions_until_ml} more until the statistical layer fits`
          }
        />
        <Stat
          label="Prediction accuracy"
          value={accuracy === null ? "—" : `${(accuracy * 100).toFixed(0)}%`}
          hint={
            analytics.accuracy.checked === 0
              ? "No predictions checked against reality yet"
              : `${analytics.accuracy.correct} of ${analytics.accuracy.checked} checked`
          }
          tone={accuracy === null ? "default" : accuracy >= 0.6 ? "positive" : "caution"}
        />
        <Stat
          label="Mean Brier score"
          value={
            analytics.accuracy.mean_brier === null
              ? "—"
              : analytics.accuracy.mean_brier.toFixed(3)
          }
          hint="Lower is better. Measures honesty, not just hit rate."
        />
        <Stat
          label="Profile confidence"
          value={`${(analytics.model_confidence.mean_trait_confidence * 100).toFixed(0)}%`}
          hint="Average evidence strength across all traits"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <Card
          title="Behavioural profile"
          subtitle="Solid line: estimated tendency. Dotted line: how much evidence stands behind it."
          action={
            <Link href="/profile" className="text-xs text-accent hover:underline">
              Details
            </Link>
          }
        >
          <RadarChart points={radar} />
          <p className="mt-3 text-xs text-ink-600">
            The dashed circle is the midpoint &mdash; where a trait sits when nothing is
            known about it yet.
          </p>
        </Card>

        <div className="space-y-6">
          <Card
            title="Decisions over time"
            subtitle="Dot size is how much the decision mattered. Above the line means the active option was taken."
            action={
              <Link href="/decisions" className="text-xs text-accent hover:underline">
                History
              </Link>
            }
          >
            <TimelineChart entries={timeline} />
          </Card>

          <Card title="Situation profile" subtitle="Which factors your recorded decisions actually involve.">
            <div className="space-y-2">
              {analytics.behavioural_summary.factor_profile
                .filter((f) => f.mean_magnitude > 0.02)
                .slice(0, 6)
                .map((factor) => (
                  <div key={factor.factor} className="flex items-center gap-3">
                    <span className="w-40 shrink-0 truncate text-xs text-ink-400">
                      {factor.label}
                    </span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
                      <div
                        className="h-full rounded-full bg-accent/70"
                        style={{ width: `${factor.mean_magnitude * 100}%` }}
                      />
                    </div>
                    <span className="w-10 text-right font-mono text-xs text-ink-600">
                      {factor.mean_magnitude.toFixed(2)}
                    </span>
                  </div>
                ))}
            </div>
            {analytics.behavioural_summary.approach_rate !== null && (
              <p className="mt-4 text-xs text-ink-600">
                Across your history you took the active option in{" "}
                <span className="font-mono text-ink-400">
                  {(analytics.behavioural_summary.approach_rate * 100).toFixed(0)}%
                </span>{" "}
                of recorded decisions. Any model has to beat that base rate to be worth
                anything.
              </p>
            )}
          </Card>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Recent predictions"
          action={
            <Link href="/predict" className="text-xs text-accent hover:underline">
              New scenario
            </Link>
          }
        >
          {analytics.recent_predictions.length === 0 ? (
            <Empty>
              No predictions yet.{" "}
              <Link href="/predict" className="text-accent hover:underline">
                Describe a situation
              </Link>{" "}
              to make one.
            </Empty>
          ) : (
            <ul className="divide-y divide-ink-800">
              {analytics.recent_predictions.map((prediction) => (
                <li key={prediction.id} className="py-3 first:pt-0 last:pb-0">
                  <Link href={`/predict?id=${prediction.id}`} className="group block">
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="truncate text-sm text-ink-200 group-hover:text-accent">
                        {prediction.predicted_option}
                      </span>
                      <span className="shrink-0 font-mono text-sm text-ink-400">
                        {(prediction.predicted_probability * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-ink-600">
                      <ConfidenceBadge label={prediction.confidence_label} />
                      <span>{new Date(prediction.created_at).toLocaleDateString()}</span>
                      {prediction.has_outcome && (
                        <span
                          className={
                            prediction.was_correct ? "text-positive" : "text-negative"
                          }
                        >
                          {prediction.was_correct ? "matched reality" : "did not match"}
                        </span>
                      )}
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="What the model has learned" subtitle="Counts over your record - observations, not inferences.">
          {profile.memory.patterns.length === 0 ? (
            <Empty>
              No stable patterns yet. Patterns appear once at least three recorded
              decisions share a factor.
            </Empty>
          ) : (
            <ul className="space-y-3">
              {profile.memory.patterns.slice(0, 5).map((pattern) => (
                <li key={pattern.id} className="text-sm leading-relaxed text-ink-400">
                  <span className="mr-2 text-accent">&bull;</span>
                  {pattern.content}
                </li>
              ))}
            </ul>
          )}

          {analytics.active_model ? (
            <div className="mt-5 border-t border-ink-800 pt-4 text-xs text-ink-600">
              Statistical layer: {analytics.active_model.kind}, trained on{" "}
              {analytics.active_model.trained_on_decisions} decisions. Time-ordered
              cross-validation accuracy{" "}
              <span className="font-mono">
                {(analytics.active_model.cv_accuracy * 100).toFixed(0)}%
              </span>
              , Brier{" "}
              <span className="font-mono">{analytics.active_model.cv_brier.toFixed(3)}</span>
              .
            </div>
          ) : (
            <div className="mt-5 border-t border-ink-800 pt-4 text-xs text-ink-600">
              The statistical layer needs{" "}
              {analytics.model_confidence.min_decisions_for_ml} decisions before it will
              fit. Until then predictions come from the trait model and case retrieval.
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function GettingStarted() {
  const steps = [
    {
      title: "Record decisions you have already made",
      body: "The situation, the options you had, which one you took, and — most valuable of all — why. Ten or twelve is enough to start.",
      href: "/decisions",
      cta: "Add a decision",
    },
    {
      title: "Answer the preference questionnaire",
      body: "Thirty short statements. Self-report is weighted below actual behaviour, but it gives the model something to work with on day one.",
      href: "/profile",
      cta: "Open questionnaire",
    },
    {
      title: "Describe a situation you are facing",
      body: "The model estimates what you would probably do, shows the past decisions that informed it, and lets you change the conditions to see what would shift the answer.",
      href: "/predict",
      cta: "Try a scenario",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-ink-200">Start here</h1>
        <p className="mt-1 max-w-2xl text-sm leading-relaxed text-ink-400">
          The twin has nothing to work from yet. It learns only from decisions you record,
          so its estimates stay at the no-information midpoint until you give it something.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {steps.map((step, index) => (
          <Card key={step.title}>
            <div className="mb-3 grid h-7 w-7 place-items-center rounded-md bg-accent/15 font-mono text-xs text-accent">
              {index + 1}
            </div>
            <h3 className="text-sm font-medium text-ink-200">{step.title}</h3>
            <p className="mt-2 text-xs leading-relaxed text-ink-400">{step.body}</p>
            <Link
              href={step.href}
              className="mt-4 inline-block text-xs text-accent hover:underline"
            >
              {step.cta} &rarr;
            </Link>
          </Card>
        ))}
      </div>

      <Banner tone="info">
        A demo account with 37 decisions already recorded ships with the project. Run{" "}
        <code className="font-mono text-xs">python -m seeds.seed_data --reset</code> in the
        backend and sign in as <code className="font-mono text-xs">demo@example.com</code>{" "}
        to see the system with a full history behind it.
      </Banner>
    </div>
  );
}
