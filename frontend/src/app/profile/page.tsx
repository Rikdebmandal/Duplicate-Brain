"use client";

import { useCallback, useEffect, useState } from "react";

import { RadarChart } from "@/components/charts";
import {
  Banner,
  Button,
  Card,
  Empty,
  Shell,
  Spinner,
  inputClass,
} from "@/components/ui";
import { api } from "@/lib/api";
import type { Profile, QuestionnaireItem, Trait } from "@/lib/types";

export default function ProfilePage() {
  return (
    <Shell>
      <ProfileView />
    </Shell>
  );
}

function ProfileView() {
  const [tab, setTab] = useState<"traits" | "memory" | "questionnaire">("traits");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setProfile(await api.profile());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the profile.");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (error) return <Banner tone="error">{error}</Banner>;
  if (!profile) return <Spinner label="Loading the profile" />;

  const radar = profile.traits.map((trait) => ({
    label: trait.label.replace(
      / (priority|tolerance|orientation|sensitivity|influence|aversion|tendency)$/i,
      "",
    ),
    value: trait.value,
    confidence: trait.confidence,
  }));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-ink-200">Behavioural profile</h1>
          <p className="mt-1 text-sm text-ink-400">
            Built from {profile.evidence_summary.decisions} decisions and{" "}
            {profile.evidence_summary.trait_observations} weighted observations.
          </p>
        </div>
        <div className="flex gap-1 rounded-lg bg-ink-900 p-1">
          {(
            [
              ["traits", "Traits"],
              ["memory", "Memory"],
              ["questionnaire", "Questionnaire"],
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

      <Banner tone="warn">{profile.disclaimer}</Banner>

      {tab === "traits" && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,22rem)_1fr]">
          <Card title="Profile shape">
            <RadarChart points={radar} />
            <div className="mt-4 space-y-1.5 text-xs text-ink-600">
              <div className="flex items-center gap-2">
                <span className="h-0.5 w-6 bg-accent" /> estimated value
              </div>
              <div className="flex items-center gap-2">
                <span className="h-0.5 w-6 border-t border-dashed border-ink-400" />{" "}
                confidence in that estimate
              </div>
            </div>
          </Card>

          <Card
            title="Traits"
            subtitle="Every value is an estimate with a credible interval. Click a trait to see the decisions behind it."
            action={
              <Button
                variant="ghost"
                onClick={() => api.rebuildProfile().then(load)}
              >
                Rebuild
              </Button>
            }
          >
            <div className="divide-y divide-ink-800">
              {profile.traits.map((trait) => (
                <TraitRow key={trait.key} trait={trait} onChanged={load} />
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === "memory" && <MemoryView profile={profile} onChanged={load} />}
      {tab === "questionnaire" && <Questionnaire onSaved={load} />}
    </div>
  );
}

function TraitRow({ trait, onChanged }: { trait: Trait; onChanged: () => void }) {
  const [open, setOpen] = useState(false);
  const [evidence, setEvidence] = useState<
    { id: string; source: string; observation: number; weight: number; note: string }[] | null
  >(null);
  const [override, setOverride] = useState<string>(
    trait.user_override === null ? "" : String(trait.user_override),
  );

  const [lo, hi] = trait.credible_interval;
  const unknown = trait.evidence_count === 0;

  async function toggle() {
    const next = !open;
    setOpen(next);
    if (next && !evidence) {
      const result = await api.traitEvidence(trait.key).catch(() => null);
      setEvidence(result?.evidence ?? []);
    }
  }

  return (
    <div className="py-3">
      <button onClick={toggle} className="w-full text-left">
        <div className="flex items-baseline justify-between gap-3">
          <span className="text-sm text-ink-200">{trait.label}</span>
          <span className="flex items-baseline gap-2 font-mono text-xs">
            {trait.status === "user_specified" && (
              <span className="rounded bg-accent/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-accent">
                yours
              </span>
            )}
            <span className={unknown ? "text-ink-600" : "text-ink-200"}>
              {unknown ? "unknown" : trait.value.toFixed(2)}
            </span>
          </span>
        </div>

        <div className="relative mt-2 h-2 rounded-full bg-ink-800">
          {/* Credible interval, so a confident estimate looks different from a guess. */}
          <div
            className="absolute inset-y-0 rounded-full bg-ink-700"
            style={{ left: `${lo * 100}%`, width: `${Math.max(1, (hi - lo) * 100)}%` }}
          />
          <div
            className="absolute top-1/2 h-3 w-1 -translate-y-1/2 rounded-full bg-accent"
            style={{ left: `${trait.value * 100}%` }}
          />
          <div className="absolute left-1/2 top-0 h-full w-px bg-ink-600" />
        </div>

        <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 text-[11px] text-ink-600">
          <span>
            {trait.low_label} &larr;&rarr; {trait.high_label}
          </span>
          <span>
            {unknown
              ? "no evidence yet"
              : `confidence ${trait.confidence.toFixed(2)} from ${trait.evidence_count} observations`}
          </span>
        </div>
      </button>

      {open && (
        <div className="mt-3 space-y-3 rounded-lg border border-ink-800 bg-ink-950 p-3">
          <p className="text-xs leading-relaxed text-ink-400">{trait.description}</p>

          <div>
            <div className="mb-1.5 text-[11px] uppercase tracking-wide text-ink-600">
              Evidence
            </div>
            {evidence === null ? (
              <p className="text-xs text-ink-600">Loading...</p>
            ) : evidence.length === 0 ? (
              <p className="text-xs text-ink-600">
                Nothing has spoken to this trait yet. It sits at the midpoint until
                something does.
              </p>
            ) : (
              <ul className="space-y-1">
                {evidence.slice(0, 6).map((item) => (
                  <li key={item.id} className="flex gap-2 text-[11px] text-ink-400">
                    <span className="shrink-0 font-mono text-ink-600">
                      {item.observation.toFixed(2)}
                    </span>
                    <span className="shrink-0 rounded bg-ink-800 px-1 text-[10px] text-ink-600">
                      {item.source}
                    </span>
                    <span className="leading-snug">{item.note}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="flex flex-wrap items-end gap-2 border-t border-ink-800 pt-3">
            <label className="text-[11px] text-ink-600">
              <span className="mb-1 block">Override this estimate (0&ndash;1)</span>
              <input
                value={override}
                onChange={(e) => setOverride(e.target.value)}
                placeholder={trait.inferred_value.toFixed(2)}
                className={`${inputClass} w-28`}
              />
            </label>
            <Button
              variant="ghost"
              onClick={async () => {
                const value = override.trim() === "" ? null : Number(override);
                if (value !== null && (Number.isNaN(value) || value < 0 || value > 1)) return;
                await api.overrideTrait(trait.key, value);
                onChanged();
              }}
            >
              {override.trim() === "" ? "Clear override" : "Set"}
            </Button>
            <p className="text-[11px] leading-snug text-ink-600">
              If the model has you wrong, say so. Overrides win over inference until you
              clear them.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function MemoryView({ profile, onChanged }: { profile: Profile; onChanged: () => void }) {
  const [key, setKey] = useState("");
  const [content, setContent] = useState("");

  const layers = [
    {
      title: "Facts",
      caption: "Things you told the system directly. Treated as given.",
      items: profile.memory.facts,
      tone: "text-positive",
    },
    {
      title: "Preferences",
      caption: "Repeatedly observed leanings, each carrying a confidence.",
      items: profile.memory.preferences,
      tone: "text-caution",
    },
    {
      title: "Patterns",
      caption: "Counts over your record. Observations, not inferences.",
      items: profile.memory.patterns,
      tone: "text-accent",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-3">
        {layers.map((layer) => (
          <Card key={layer.title} title={layer.title} subtitle={layer.caption}>
            {layer.items.length === 0 ? (
              <Empty>Nothing here yet.</Empty>
            ) : (
              <ul className="space-y-3">
                {layer.items.map((item) => (
                  <li key={item.id} className="text-xs leading-relaxed text-ink-400">
                    <span className={`mr-1.5 ${layer.tone}`}>&bull;</span>
                    {item.content}
                    {item.evidence_count > 0 && (
                      <span className="ml-1.5 font-mono text-[10px] text-ink-600">
                        [{item.evidence_count} obs, conf {item.confidence.toFixed(2)}]
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </Card>
        ))}
      </div>

      <Card
        title="Add a fact"
        subtitle="Stable context the system cannot infer - who depends on you, what you have committed to."
      >
        <div className="flex flex-wrap gap-3">
          <input
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="living_situation"
            className={`${inputClass} sm:w-48`}
          />
          <input
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Lives in the same city as their parents."
            className={`${inputClass} flex-1`}
          />
          <Button
            onClick={async () => {
              if (!key.trim() || !content.trim()) return;
              await api.addFact({ key: key.trim(), content: content.trim() });
              setKey("");
              setContent("");
              onChanged();
            }}
          >
            Add
          </Button>
        </div>
      </Card>
    </div>
  );
}

function Questionnaire({ onSaved }: { onSaved: () => void }) {
  const [items, setItems] = useState<QuestionnaireItem[]>([]);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.questionnaire().then((result) => {
      setItems(result.items);
      setNote(result.note);
      setAnswers(
        Object.fromEntries(
          result.items
            .filter((item) => item.current_answer !== null)
            .map((item) => [item.key, item.current_answer as number]),
        ),
      );
    });
  }, []);

  if (items.length === 0) return <Spinner />;

  const grouped = items.reduce<Record<string, QuestionnaireItem[]>>((acc, item) => {
    (acc[item.construct] ??= []).push(item);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <Banner tone="info">{note}</Banner>

      {Object.entries(grouped).map(([construct, group]) => (
        <Card key={construct} title={construct}>
          <div className="space-y-5">
            {group.map((item) => (
              <div key={item.key}>
                <p className="mb-2 text-sm leading-relaxed text-ink-200">{item.prompt}</p>
                <div className="flex flex-wrap gap-1.5">
                  {item.scale.map((label, index) => {
                    const value = index + 1;
                    const active = answers[item.key] === value;
                    return (
                      <button
                        key={value}
                        onClick={() =>
                          setAnswers((prev) => ({ ...prev, [item.key]: value }))
                        }
                        className={`rounded-lg border px-2.5 py-1.5 text-xs transition ${
                          active
                            ? "border-accent bg-accent/15 text-accent"
                            : "border-ink-700 text-ink-400 hover:border-ink-600"
                        }`}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </Card>
      ))}

      <div className="flex items-center gap-4">
        <Button
          disabled={busy || Object.keys(answers).length === 0}
          onClick={async () => {
            setBusy(true);
            await api.submitQuestionnaire(
              Object.entries(answers).map(([item_key, value]) => ({ item_key, value })),
            );
            setBusy(false);
            setSaved(true);
            onSaved();
          }}
        >
          {busy ? "Saving..." : `Save ${Object.keys(answers).length} answers`}
        </Button>
        {saved && <span className="text-sm text-positive">Saved. Profile rebuilt.</span>}
      </div>
    </div>
  );
}
