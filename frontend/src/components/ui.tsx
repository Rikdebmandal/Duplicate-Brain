"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ReactNode, useEffect, useState } from "react";

import { api, getToken, setToken } from "@/lib/api";
import type { User } from "@/lib/types";

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

export function Card({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-ink-800 bg-ink-900/60 p-5 shadow-sm ${className}`}
    >
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink-200">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs text-ink-400">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "positive" | "negative" | "caution";
}) {
  const toneClass = {
    default: "text-ink-200",
    positive: "text-positive",
    negative: "text-negative",
    caution: "text-caution",
  }[tone];
  return (
    <div className="rounded-xl border border-ink-800 bg-ink-900/60 p-4">
      <div className="text-xs uppercase tracking-wide text-ink-400">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-400">{hint}</div>}
    </div>
  );
}

export function Button({
  children,
  onClick,
  type = "button",
  variant = "primary",
  disabled = false,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  type?: "button" | "submit";
  variant?: "primary" | "ghost" | "danger";
  disabled?: boolean;
  className?: string;
}) {
  const styles = {
    primary: "bg-accent text-white hover:bg-accent-soft disabled:bg-ink-700",
    ghost: "border border-ink-700 text-ink-200 hover:border-ink-600 hover:bg-ink-800",
    danger: "border border-negative/50 text-negative hover:bg-negative/10",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-60 ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-ink-400">
        {label}
      </span>
      {children}
      {hint && <span className="mt-1 block text-xs text-ink-400">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-lg border border-ink-700 bg-ink-950 px-3 py-2 text-sm text-ink-200 " +
  "placeholder:text-ink-600 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent";

export function Banner({
  tone = "info",
  children,
}: {
  tone?: "info" | "warn" | "error" | "success";
  children: ReactNode;
}) {
  const styles = {
    info: "border-accent-dim bg-accent/10 text-ink-200",
    warn: "border-caution/40 bg-caution/10 text-caution",
    error: "border-negative/40 bg-negative/10 text-negative",
    success: "border-positive/40 bg-positive/10 text-positive",
  }[tone];
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${styles}`}>{children}</div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 text-sm text-ink-400">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-ink-600 border-t-accent" />
      {label}...
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-ink-700 px-4 py-8 text-center text-sm text-ink-400">
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// domain-specific display
// ---------------------------------------------------------------------------

/**
 * Confidence in the *estimate*, not the probability. Rendered as a distinct
 * chip so it can never be confused with the predicted probability itself.
 */
export function ConfidenceBadge({
  label,
  score,
}: {
  label: string;
  score?: number;
}) {
  const styles: Record<string, string> = {
    high: "border-positive/40 bg-positive/10 text-positive",
    medium: "border-caution/40 bg-caution/10 text-caution",
    low: "border-ink-600 bg-ink-800 text-ink-400",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${
        styles[label] ?? styles.low
      }`}
      title="How much evidence stands behind this estimate - not how likely the outcome is."
    >
      {label} confidence
      {score !== undefined && (
        <span className="font-mono opacity-70">{score.toFixed(2)}</span>
      )}
    </span>
  );
}

/** A labelled probability bar. */
export function ProbabilityBar({
  label,
  value,
  highlight = false,
  compare,
}: {
  label: string;
  value: number;
  highlight?: boolean;
  compare?: number;
}) {
  const delta = compare === undefined ? null : value - compare;
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className={highlight ? "font-medium text-ink-200" : "text-ink-400"}>
          {label}
        </span>
        <span className="flex items-baseline gap-2 font-mono tabular-nums">
          {delta !== null && Math.abs(delta) >= 0.005 && (
            <span className={delta > 0 ? "text-xs text-positive" : "text-xs text-negative"}>
              {delta > 0 ? "+" : ""}
              {(delta * 100).toFixed(0)}
            </span>
          )}
          <span className={highlight ? "text-ink-200" : "text-ink-400"}>
            {(value * 100).toFixed(0)}%
          </span>
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-ink-800">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            highlight ? "bg-accent" : "bg-ink-600"
          }`}
          style={{ width: `${Math.max(1, value * 100)}%` }}
        />
        {compare !== undefined && (
          <div
            className="relative -top-2 h-2 w-px bg-ink-200/70"
            style={{ marginLeft: `${Math.max(0, compare * 100)}%` }}
            title={`was ${(compare * 100).toFixed(0)}%`}
          />
        )}
      </div>
    </div>
  );
}

/** A factor's signed contribution, rendered from the centre outwards. */
export function ContributionBar({
  label,
  value,
  magnitude,
  symbol,
  max,
  evidence,
}: {
  label: string;
  value: number;
  magnitude: number;
  symbol: string;
  max: number;
  evidence?: string[];
}) {
  const scale = max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
  const positive = value > 0;
  return (
    <div className="grid grid-cols-[minmax(0,11rem)_1fr_auto] items-center gap-3 py-1.5">
      <div className="min-w-0">
        <div className="truncate text-sm text-ink-200" title={label}>
          {label}
        </div>
        {evidence && evidence.length > 0 && (
          <div className="truncate text-[11px] text-ink-600" title={evidence.join(" | ")}>
            &ldquo;{evidence[0]}&rdquo;
          </div>
        )}
      </div>
      <div className="relative h-3">
        <div className="absolute inset-y-0 left-1/2 w-px bg-ink-700" />
        <div
          className={`absolute top-0.5 h-2 rounded-sm ${
            positive ? "bg-positive/80" : "bg-negative/80"
          }`}
          style={{
            width: `${scale * 50}%`,
            left: positive ? "50%" : `${50 - scale * 50}%`,
          }}
        />
      </div>
      <div className="w-24 text-right">
        <span
          className={`font-mono text-xs ${positive ? "text-positive" : "text-negative"}`}
        >
          {symbol}
        </span>
        <span className="ml-2 font-mono text-[11px] text-ink-600">
          {magnitude.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

/** Standing reminder that everything here is an estimate. */
export function EthicsNote({ children }: { children?: ReactNode }) {
  return (
    <p className="text-xs leading-relaxed text-ink-600">
      {children ??
        "This is a behavioural prediction model, not a simulation of a person's mind. Every trait is an estimate with a stated confidence, and every prediction is a probability rather than a statement of what someone will do. It must not be used for medical, legal, employment or lending decisions."}
    </p>
  );
}

// ---------------------------------------------------------------------------
// shell
// ---------------------------------------------------------------------------

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/decisions", label: "Decision history" },
  { href: "/predict", label: "New scenario" },
  { href: "/profile", label: "Behavioural profile" },
  { href: "/performance", label: "Model performance" },
];

export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    api
      .me()
      .then(setUser)
      .catch(() => router.replace("/login"))
      .finally(() => setChecked(true));
  }, [router]);

  if (!checked) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner label="Signing in" />
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-ink-800 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-3 px-6 py-3">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="grid h-7 w-7 place-items-center rounded-md bg-accent/15 text-accent">
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M12 3v18M4 8l8-5 8 5M4 16l8 5 8-5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span className="text-sm font-semibold text-ink-200">Cognitive Twin</span>
          </Link>

          <nav className="flex flex-1 flex-wrap gap-1">
            {NAV.map((item) => {
              const active =
                item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`rounded-lg px-3 py-1.5 text-sm transition ${
                    active
                      ? "bg-ink-800 text-ink-200"
                      : "text-ink-400 hover:bg-ink-900 hover:text-ink-200"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-3 text-xs text-ink-400">
            <span className="hidden sm:inline">{user?.email}</span>
            <button
              onClick={() => {
                setToken(null);
                router.replace("/login");
              }}
              className="rounded-lg border border-ink-700 px-2.5 py-1.5 hover:bg-ink-800 hover:text-ink-200"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">{children}</main>

      <footer className="mx-auto max-w-7xl px-6 pb-10">
        <EthicsNote />
      </footer>
    </div>
  );
}
