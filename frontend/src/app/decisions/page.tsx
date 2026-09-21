"use client";

import { useCallback, useEffect, useState } from "react";

import { TimelineChart } from "@/components/charts";
import {
  Banner,
  Button,
  Card,
  Empty,
  Field,
  Shell,
  Spinner,
  inputClass,
} from "@/components/ui";
import { api } from "@/lib/api";
import type { DecisionRecord, TextAnalysis, TimelineEntry } from "@/lib/types";

export default function DecisionsPage() {
  return (
    <Shell>
      <Decisions />
    </Shell>
  );
}

function Decisions() {
  const [tab, setTab] = useState<"history" | "add" | "import">("history");
  const [items, setItems] = useState<DecisionRecord[]>([]);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [list, tl] = await Promise.all([
        api.listDecisions({ limit: 200 }),
        api.timeline(),
      ]);
      setItems(list.items);
      setTotal(list.total);
      setTimeline(tl.timeline);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load decisions.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-ink-200">Decision history</h1>
          <p className="mt-1 text-sm text-ink-400">
            {total} decisions on record. Everything the model knows comes from here.
          </p>
        </div>
        <div className="flex gap-1 rounded-lg bg-ink-900 p-1">
          {(
            [
              ["history", "History"],
              ["add", "Add decision"],
              ["import", "Import from text"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`rounded-md px-3 py-1.5 text-sm transition ${
                tab === key ? "bg-ink-800 text-ink-200" : "text-ink-400 hover:text-ink-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {error && <Banner tone="error">{error}</Banner>}

      {tab === "add" && <AddDecision onSaved={() => { load(); setTab("history"); }} />}
      {tab === "import" && <ImportText onImported={load} />}

      {tab === "history" && (
        <>
          {timeline.length > 0 && (
            <Card
              title="Timeline"
              subtitle="Above the line: took the active option. Below: chose the safer one. Dot size is importance."
            >
              <TimelineChart entries={timeline} />
            </Card>
          )}

          {loading ? (
            <Spinner />
          ) : items.length === 0 ? (
            <Empty>
              No decisions recorded yet. Use &ldquo;Add decision&rdquo; to record one you
              have already made.
            </Empty>
          ) : (
            <div className="space-y-3">
              {items.map((decision) => (
                <DecisionCard key={decision.id} decision={decision} onDeleted={load} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function DecisionCard({
  decision,
  onDeleted,
}: {
  decision: DecisionRecord;
  onDeleted: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const topFactors = Object.entries(decision.factors)
    .filter(([, value]) => value > 0.15)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  return (
    <Card className="transition hover:border-ink-700">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-xs text-ink-600">
            <span className="rounded bg-ink-800 px-1.5 py-0.5 text-ink-400">
              {decision.category}
            </span>
            <span>{decision.occurred_at?.slice(0, 10)}</span>
            <span>importance {decision.importance}/10</span>
            {decision.source !== "manual" && <span>via {decision.source}</span>}
          </div>

          <p className="mt-2 text-sm leading-relaxed text-ink-200">
            {expanded || decision.situation.length < 180
              ? decision.situation
              : `${decision.situation.slice(0, 180)}...`}
          </p>

          <div className="mt-3 flex flex-wrap gap-1.5">
            {decision.options.map((option) => (
              <span
                key={option.label}
                className={`rounded-md px-2 py-1 text-xs ${
                  option.was_chosen
                    ? "bg-accent/15 text-accent"
                    : "bg-ink-800 text-ink-600 line-through"
                }`}
              >
                {option.label}
              </span>
            ))}
          </div>

          {decision.reason && (
            <p className="mt-3 border-l-2 border-ink-700 pl-3 text-xs italic leading-relaxed text-ink-400">
              &ldquo;{decision.reason}&rdquo;
            </p>
          )}

          {expanded && (
            <div className="mt-4 space-y-3 border-t border-ink-800 pt-3">
              <div>
                <div className="mb-1.5 text-xs uppercase tracking-wide text-ink-600">
                  Detected factors
                </div>
                {topFactors.length === 0 ? (
                  <p className="text-xs text-ink-600">
                    No factors detected above threshold in this text.
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {topFactors.map(([key, value]) => (
                      <span
                        key={key}
                        className="rounded bg-ink-800 px-2 py-1 font-mono text-[11px] text-ink-400"
                      >
                        {key} {value.toFixed(2)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              {decision.outcome && (
                <div>
                  <div className="mb-1 text-xs uppercase tracking-wide text-ink-600">
                    Outcome
                  </div>
                  <p className="text-xs text-ink-400">{decision.outcome}</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex shrink-0 flex-col items-end gap-2">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="text-xs text-accent hover:underline"
          >
            {expanded ? "Less" : "More"}
          </button>
          <button
            onClick={async () => {
              if (!window.confirm("Delete this decision? The profile will be rebuilt without it."))
                return;
              setDeleting(true);
              await api.deleteDecision(decision.id).catch(() => undefined);
              onDeleted();
            }}
            disabled={deleting}
            className="text-xs text-ink-600 hover:text-negative"
          >
            Delete
          </button>
        </div>
      </div>
    </Card>
  );
}

function AddDecision({ onSaved }: { onSaved: () => void }) {
  const [situation, setSituation] = useState("");
  const [options, setOptions] = useState(["", ""]);
  const [chosen, setChosen] = useState(0);
  const [reason, setReason] = useState("");
  const [importance, setImportance] = useState(5);
  const [category, setCategory] = useState("general");
  const [occurredAt, setOccurredAt] = useState("");
  const [outcome, setOutcome] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    const cleaned = options.map((o) => o.trim()).filter(Boolean);
    if (situation.trim().length < 3 || cleaned.length < 2) {
      setError("Describe the situation and give at least two options.");
      return;
    }
    if (!cleaned[chosen]) {
      setError("Mark which option you actually took.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.createDecision({
        situation: situation.trim(),
        options: cleaned,
        decision: cleaned[chosen],
        reason,
        importance,
        category,
        outcome,
        occurred_at: occurredAt ? new Date(occurredAt).toISOString() : null,
      });
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the decision.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      title="Record a decision you have already made"
      subtitle="Past decisions are the training data. The reason matters as much as the choice."
    >
      {error && (
        <div className="mb-4">
          <Banner tone="error">{error}</Banner>
        </div>
      )}
      <div className="space-y-4">
        <Field label="What was the situation?">
          <textarea
            value={situation}
            onChange={(e) => setSituation(e.target.value)}
            rows={3}
            className={inputClass}
            placeholder="A company in another city offered 50% more money, but I would have to move away from my parents."
          />
        </Field>

        <Field label="What were your options?" hint="Select the one you actually took.">
          <div className="space-y-2">
            {options.map((option, index) => (
              <div key={index} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="chosen"
                  checked={chosen === index}
                  onChange={() => setChosen(index)}
                  className="h-3.5 w-3.5 shrink-0 accent-accent"
                  aria-label={`Chose option ${index + 1}`}
                />
                <input
                  value={option}
                  onChange={(e) =>
                    setOptions((prev) => prev.map((v, i) => (i === index ? e.target.value : v)))
                  }
                  className={inputClass}
                  placeholder={index === 0 ? "Accept and relocate" : "Stay in my current job"}
                />
                {options.length > 2 && (
                  <button
                    onClick={() => {
                      setOptions((prev) => prev.filter((_, i) => i !== index));
                      if (chosen >= index && chosen > 0) setChosen(chosen - 1);
                    }}
                    className="rounded-lg border border-ink-700 px-2.5 py-2 text-ink-400 hover:border-negative/50 hover:text-negative"
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

        <Field
          label="Why did you choose that?"
          hint="The most valuable field in the form: it is the only source for value-driven traits."
        >
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            rows={2}
            className={inputClass}
            placeholder="I did not want to move away from family, and the company felt too new."
          />
        </Field>

        <div className="grid gap-4 sm:grid-cols-3">
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
          <Field label="When?">
            <input
              type="date"
              value={occurredAt}
              onChange={(e) => setOccurredAt(e.target.value)}
              className={inputClass}
            />
          </Field>
          <Field label={`Importance ${importance}/10`}>
            <input
              type="range"
              min={1}
              max={10}
              value={importance}
              onChange={(e) => setImportance(Number(e.target.value))}
              className="mt-3"
            />
          </Field>
        </div>

        <Field label="What happened afterwards?" hint="Optional.">
          <input
            value={outcome}
            onChange={(e) => setOutcome(e.target.value)}
            className={inputClass}
            placeholder="Stayed. Got a raise six months later."
          />
        </Field>

        <Button onClick={save} disabled={busy}>
          {busy ? "Saving..." : "Save decision"}
        </Button>
      </div>
    </Card>
  );
}

function ImportText({ onImported }: { onImported: () => void }) {
  const [text, setText] = useState("");
  const [analysis, setAnalysis] = useState<TextAnalysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<number[]>([]);

  async function analyse() {
    setBusy(true);
    setError(null);
    try {
      setAnalysis(await api.analyseText(text));
      setSaved([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyse the text.");
    } finally {
      setBusy(false);
    }
  }

  async function confirm(index: number) {
    if (!analysis) return;
    const candidate = analysis.candidates[index];
    try {
      await api.createDecision({
        situation: candidate.situation,
        options: candidate.options,
        decision: candidate.decision,
        reason: candidate.reason,
        importance: 5,
      });
      setSaved((prev) => [...prev, index]);
      onImported();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save that candidate.");
    }
  }

  return (
    <div className="space-y-6">
      <Card
        title="Import from a journal or notes"
        subtitle="Candidates are proposals. Nothing enters your history until you confirm it."
      >
        {error && (
          <div className="mb-4">
            <Banner tone="error">{error}</Banner>
          </div>
        )}
        <Field label="Paste text" hint="Journal entries, notes, an interview transcript.">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={8}
            className={inputClass}
            placeholder="Last March I decided to turn down the Bengaluru offer instead of relocating, because I wanted to stay close to my parents..."
          />
        </Field>
        <div className="mt-4">
          <Button onClick={analyse} disabled={busy || text.trim().length < 20}>
            {busy ? "Reading..." : "Find decisions in this text"}
          </Button>
        </div>
      </Card>

      {analysis && (
        <>
          <Card
            title={`${analysis.candidates.length} candidate decisions found`}
            subtitle={analysis.note}
          >
            {analysis.candidates.length === 0 ? (
              <Empty>
                No clear decisions found. The extractor looks for phrases like &ldquo;I
                decided&rdquo;, &ldquo;I turned down&rdquo;, &ldquo;I chose&rdquo;.
              </Empty>
            ) : (
              <div className="space-y-4">
                {analysis.candidates.map((candidate, index) => (
                  <div
                    key={index}
                    className="rounded-lg border border-ink-800 bg-ink-950 p-4"
                  >
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="font-mono text-xs text-ink-600">
                        extraction confidence {(candidate.confidence * 100).toFixed(0)}%
                      </span>
                      {saved.includes(index) ? (
                        <span className="text-xs text-positive">Added</span>
                      ) : (
                        <Button variant="ghost" onClick={() => confirm(index)}>
                          Confirm and add
                        </Button>
                      )}
                    </div>
                    <p className="text-xs leading-relaxed text-ink-400">
                      {candidate.situation}
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {candidate.options.map((option) => (
                        <span
                          key={option}
                          className={`rounded px-2 py-0.5 text-xs ${
                            option === candidate.decision
                              ? "bg-accent/15 text-accent"
                              : "bg-ink-800 text-ink-600"
                          }`}
                        >
                          {option}
                        </span>
                      ))}
                    </div>
                    {candidate.reason && (
                      <p className="mt-2 text-xs italic text-ink-600">
                        {candidate.reason}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Card>

          {(analysis.values.length > 0 || analysis.goals.length > 0) && (
            <Card title="Values and goals mentioned" subtitle="Extracted for context, not added to the model automatically.">
              <div className="grid gap-6 sm:grid-cols-2">
                {analysis.values.length > 0 && (
                  <div>
                    <div className="mb-2 text-xs uppercase tracking-wide text-ink-600">
                      Stated values
                    </div>
                    <ul className="space-y-1.5">
                      {analysis.values.map((value, index) => (
                        <li key={index} className="text-xs leading-relaxed text-ink-400">
                          &bull; {value}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {analysis.goals.length > 0 && (
                  <div>
                    <div className="mb-2 text-xs uppercase tracking-wide text-ink-600">
                      Stated goals
                    </div>
                    <ul className="space-y-1.5">
                      {analysis.goals.map((goal, index) => (
                        <li key={index} className="text-xs leading-relaxed text-ink-400">
                          &bull; {goal}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
