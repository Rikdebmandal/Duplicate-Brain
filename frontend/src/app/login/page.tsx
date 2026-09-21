"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

import { Banner, Button, Field, inputClass } from "@/components/ui";
import { ApiError, api, getToken, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [allowLlm, setAllowLlm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (getToken()) router.replace("/");
  }, [router]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const result =
        mode === "login"
          ? await api.login({ email, password })
          : await api.register({
              email,
              password,
              display_name: displayName,
              allow_llm_processing: allowLlm,
            });
      setToken(result.access_token);
      router.replace("/");
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Something went wrong. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6 py-12">
      <div className="mb-8">
        <div className="mb-4 flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-accent/15 text-accent">
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M12 3v18M4 8l8-5 8 5M4 16l8 5 8-5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <h1 className="text-lg font-semibold text-ink-200">
            Personal Cognitive Digital Twin
          </h1>
        </div>
        <p className="text-sm leading-relaxed text-ink-400">
          Record the decisions you have actually made, and the model estimates how you
          would probably respond to something new &mdash; with the evidence behind every
          estimate.
        </p>
      </div>

      <form onSubmit={submit} className="space-y-4 rounded-xl border border-ink-800 bg-ink-900/60 p-6">
        <div className="flex gap-1 rounded-lg bg-ink-950 p-1">
          {(["login", "register"] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => {
                setMode(option);
                setError(null);
              }}
              className={`flex-1 rounded-md px-3 py-1.5 text-sm transition ${
                mode === option ? "bg-ink-800 text-ink-200" : "text-ink-400 hover:text-ink-200"
              }`}
            >
              {option === "login" ? "Sign in" : "Create account"}
            </button>
          ))}
        </div>

        {error && <Banner tone="error">{error}</Banner>}

        <Field label="Email">
          <input
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={inputClass}
            placeholder="you@example.com"
          />
        </Field>

        <Field
          label="Password"
          hint={mode === "register" ? "At least 8 characters." : undefined}
        >
          <input
            type="password"
            required
            minLength={8}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={inputClass}
          />
        </Field>

        {mode === "register" && (
          <>
            <Field label="Display name">
              <input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className={inputClass}
                placeholder="Optional"
              />
            </Field>

            <label className="flex cursor-pointer items-start gap-3 rounded-lg border border-ink-800 bg-ink-950 p-3">
              <input
                type="checkbox"
                checked={allowLlm}
                onChange={(e) => setAllowLlm(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-ink-600 bg-ink-900 accent-accent"
              />
              <span className="text-xs leading-relaxed text-ink-400">
                <span className="font-medium text-ink-200">
                  Allow the optional LLM reasoning layer.
                </span>{" "}
                Scenario text and retrieved past decisions are sent to the language model
                to generate narrative explanations. Everything works without this; leave
                it off and the other three layers carry the prediction. You can change it
                later.
              </span>
            </label>
          </>
        )}

        <Button type="submit" disabled={busy} className="w-full">
          {busy
            ? "Working..."
            : mode === "login"
              ? "Sign in"
              : "Create account"}
        </Button>
      </form>

      <p className="mt-6 text-xs leading-relaxed text-ink-600">
        Your decision history is private to your account. It is a record of how you think,
        so it is treated accordingly: you can export everything, and deleting your account
        erases it rather than flagging it as deleted.
      </p>
    </div>
  );
}
