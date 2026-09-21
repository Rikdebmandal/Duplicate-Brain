"use client";

import type {
  Analytics,
  AuthResponse,
  CounterfactualResult,
  DecisionRecord,
  Evaluation,
  FactorMeta,
  Landscape,
  OutcomeResult,
  Performance,
  Prediction,
  Profile,
  QuestionnaireItem,
  TextAnalysis,
  TimelineEntry,
  TraitMeta,
  User,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "https://duplicate-brain.onrender.com/api/v1";
const TOKEN_KEY = "cognitive-twin-token";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string = "error",
    readonly details: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private browsing: the session simply will not persist */
  }
}

async function request<T>(
  path: string,
  options: RequestInit & { auth?: boolean } = {},
): Promise<T> {
  const { auth = true, headers, ...rest } = options;
  const finalHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    ...((headers as Record<string, string>) ?? {}),
  };

  if (auth) {
    const token = getToken();
    if (token) finalHeaders.Authorization = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, { ...rest, headers: finalHeaders });
  } catch {
    throw new ApiError(
      "Could not reach the API. Is the backend running?",
      0,
      "network_error",
    );
  }

  if (response.status === 204) return undefined as T;

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const error = body?.error ?? {};
    // A 401 means the stored token is stale; clearing it sends the user back
    // to sign-in rather than leaving them on a page that silently fails.
    if (response.status === 401) setToken(null);
    throw new ApiError(
      error.message ?? `Request failed (${response.status})`,
      response.status,
      error.code ?? "error",
      error.details ?? null,
    );
  }

  return body as T;
}

export const api = {
  // -- auth ----------------------------------------------------------------
  register: (payload: {
    email: string;
    password: string;
    display_name?: string;
    allow_llm_processing?: boolean;
  }) =>
    request<AuthResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify(payload),
      auth: false,
    }),

  login: (payload: { email: string; password: string }) =>
    request<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
      auth: false,
    }),

  me: () => request<User>("/auth/me"),

  updateMe: (payload: Partial<Pick<User, "display_name" | "allow_llm_processing">>) =>
    request<User>("/auth/me", { method: "PATCH", body: JSON.stringify(payload) }),

  deleteAccount: () =>
    request<{ deleted: boolean; records_removed: Record<string, number> }>(
      "/auth/me",
      { method: "DELETE" },
    ),

  exportData: () => request<Record<string, unknown>>("/auth/export"),

  // -- decisions -----------------------------------------------------------
  createDecision: (payload: Record<string, unknown>) =>
    request<{ decision: DecisionRecord; profile: Record<string, unknown> }>(
      "/decisions",
      { method: "POST", body: JSON.stringify(payload) },
    ),

  listDecisions: (params: { limit?: number; offset?: number; search?: string } = {}) => {
    const query = new URLSearchParams();
    if (params.limit) query.set("limit", String(params.limit));
    if (params.offset) query.set("offset", String(params.offset));
    if (params.search) query.set("search", params.search);
    const suffix = query.toString() ? `?${query}` : "";
    return request<{
      items: DecisionRecord[];
      total: number;
      limit: number;
      offset: number;
    }>(`/decisions${suffix}`);
  },

  deleteDecision: (id: string) =>
    request<{ deleted: boolean }>(`/decisions/${id}`, { method: "DELETE" }),

  timeline: () => request<{ timeline: TimelineEntry[] }>("/decisions/timeline"),

  // -- profile -------------------------------------------------------------
  profile: () => request<Profile>("/profile"),

  traitEvidence: (trait: string) =>
    request<{
      trait: string;
      value: number;
      confidence: number;
      evidence_count: number;
      evidence: {
        id: string;
        decision_id: string | null;
        source: string;
        observation: number;
        weight: number;
        note: string;
      }[];
      note: string;
    }>(`/profile/traits/${trait}/evidence`),

  overrideTrait: (trait: string, value: number | null) =>
    request<{ trait: string; user_override: number | null; status: string }>(
      `/profile/traits/${trait}/override`,
      { method: "PUT", body: JSON.stringify({ value }) },
    ),

  rebuildProfile: () =>
    request<Record<string, unknown>>("/profile/rebuild", { method: "POST" }),

  questionnaire: () =>
    request<{ items: QuestionnaireItem[]; answered: number; total: number; note: string }>(
      "/profile/questionnaire",
    ),

  submitQuestionnaire: (answers: { item_key: string; value: number }[]) =>
    request<{ accepted: number; rejected: unknown[] }>("/profile/questionnaire", {
      method: "POST",
      body: JSON.stringify({ answers }),
    }),

  addFact: (payload: { key: string; content: string; importance?: number }) =>
    request<Record<string, unknown>>("/profile/memory/facts", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  schema: () =>
    request<{ factors: FactorMeta[]; traits: TraitMeta[] }>("/profile/schema"),

  // -- prediction ----------------------------------------------------------
  predict: (payload: {
    scenario: string;
    options: string[];
    category?: string;
    use_llm?: boolean;
    factor_overrides?: Record<string, number>;
  }) =>
    request<Prediction>("/scenario/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getPrediction: (id: string) => request<Prediction>(`/predictions/${id}`),

  listPredictions: (limit = 25) =>
    request<{ items: Record<string, unknown>[] }>(`/predictions?limit=${limit}`),

  counterfactual: (id: string, overrides: Record<string, number>, persist = false) =>
    request<CounterfactualResult>(`/predictions/${id}/counterfactual`, {
      method: "POST",
      body: JSON.stringify({ overrides, persist }),
    }),

  landscape: (id: string, steps = 5, factors?: string[]) =>
    request<Landscape>(`/predictions/${id}/landscape`, {
      method: "POST",
      body: JSON.stringify({ steps, factors: factors ?? null }),
    }),

  recordOutcome: (
    id: string,
    payload: {
      actual_option: string;
      note?: string;
      reason?: string;
      importance?: number;
      promote_to_history?: boolean;
    },
  ) =>
    request<OutcomeResult>(`/predictions/${id}/feedback`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  submitFeedback: (payload: {
    kind: string;
    target_key?: string;
    value?: number | null;
    comment?: string;
    prediction_id?: string;
  }) =>
    request<Record<string, unknown>>("/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // -- analytics -----------------------------------------------------------
  analytics: () => request<Analytics>("/analytics"),

  performance: () => request<Performance>("/analytics/performance"),

  evaluate: () =>
    request<Evaluation>("/analytics/evaluate", {
      method: "POST",
      body: JSON.stringify({}),
    }),

  trainModel: () =>
    request<Record<string, unknown>>("/models/train", { method: "POST" }),

  analyseText: (text: string) =>
    request<TextAnalysis>("/text/analyze", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
};
