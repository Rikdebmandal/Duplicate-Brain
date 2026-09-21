// Shapes returned by the backend. Kept deliberately close to the API payloads
// so a change on either side shows up as a type error rather than at runtime.

export interface User {
  id: string;
  email: string;
  display_name: string;
  allow_llm_processing: boolean;
  data_retention_days: number;
  created_at: string;
  profile_refreshed_at: string | null;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
  user: User;
}

export interface FactorMeta {
  key: string;
  label: string;
  description: string;
  sign_under_approach: number;
}

export interface TraitMeta {
  key: string;
  label: string;
  description: string;
  low_label: string;
  high_label: string;
  kind: "tolerance" | "priority";
}

export interface Trait {
  key: string;
  label: string;
  description: string;
  low_label: string;
  high_label: string;
  kind: "tolerance" | "priority";
  value: number;
  inferred_value: number;
  user_override: number | null;
  confidence: number;
  evidence_count: number;
  credible_interval: [number, number];
  supporting_decisions: string[];
  /** Always shown in the UI so an estimate is never rendered as a fact. */
  status: "inferred" | "user_specified";
}

export interface MemoryItem {
  id: string;
  kind: "fact" | "preference" | "pattern";
  key: string;
  content: string;
  importance: number;
  confidence: number;
  evidence_count: number;
  detail: Record<string, unknown>;
  last_updated: string | null;
}

export interface Profile {
  traits: Trait[];
  model_params: Record<string, number>;
  evidence_summary: {
    decisions: number;
    questionnaire_items: number;
    trait_observations: number;
    ml_ready: boolean;
    min_decisions_for_ml: number;
  };
  memory: {
    facts: MemoryItem[];
    preferences: MemoryItem[];
    patterns: MemoryItem[];
  };
  disclaimer: string;
}

export interface DecisionOptionRecord {
  label: string;
  was_chosen: boolean;
  stance: Record<string, unknown>;
}

export interface DecisionRecord {
  id: string;
  situation: string;
  category: string;
  options: DecisionOptionRecord[];
  decision: string;
  reason: string;
  importance: number;
  outcome: string;
  satisfaction: number | null;
  occurred_at: string | null;
  source: string;
  factors: Record<string, number>;
  coverage: number;
}

export interface TimelineEntry {
  id: string;
  occurred_at: string;
  scenario: string;
  category: string;
  chosen_option: string;
  importance: number;
  approach: number;
  satisfaction: number | null;
  dominant_factor: string | null;
}

export interface Contribution {
  factor: string;
  label: string;
  magnitude: number;
  exposure: number;
  valence: number;
  value: number;
}

export interface FactorExplanation {
  factor: string;
  label: string;
  description: string;
  magnitude: number;
  contribution: number;
  direction: "supports" | "opposes";
  symbol: string;
  valence: number;
  evidence_phrases: string[];
}

export interface SimilarDecision {
  decision_id: string;
  similarity: number;
  text_similarity: number;
  factor_similarity: number;
  scenario_text: string;
  chosen_option: string;
  reason: string;
  occurred_at: string;
  importance: number;
  approach: number;
  outcome: string;
  satisfaction: number | null;
  factors: Record<string, number>;
}

export interface LayerOutput {
  name: "profile" | "statistical" | "retrieval" | "llm";
  available: boolean;
  weight: number;
  probabilities: number[];
  detail: Record<string, unknown> & { reason?: string; description?: string };
}

export interface PredictionOption {
  label: string;
  probability: number;
  utility: number;
  stance: Record<string, unknown>;
  contributions: Contribution[];
}

export interface Explanation {
  summary: string;
  llm_reasoning: string;
  llm_caveats: string[];
  important_factors: FactorExplanation[];
  supporting: FactorExplanation[];
  opposing: FactorExplanation[];
  reasoning_trace: { step: number; title: string; detail: string }[];
  observed: string[];
  inferred: { trait: string; label: string; value: number; confidence: number; statement: string }[];
  ml_attribution: {
    feature: string;
    contribution: number;
    value: number;
    coefficient: number | null;
    scope: "local" | "global";
  }[];
  factor_source: string;
  emotions_detected: Record<string, number>;
  limitations: string[];
}

export interface Prediction {
  id: string;
  created_at: string;
  scenario: string;
  category: string;
  options: PredictionOption[];
  predicted_option: string;
  predicted_probability: number;
  confidence_label: "low" | "medium" | "high";
  confidence_score: number;
  confidence_inputs: Record<string, number>;
  factors: Record<string, number>;
  factor_source: string;
  layers: LayerOutput[];
  similar_decisions: SimilarDecision[];
  explanation: Explanation;
  evidence_count: number;
  model_version: string;
  is_counterfactual: boolean;
  overrides: Record<string, number>;
  disclaimer: string;
}

export interface CounterfactualResult {
  original_prediction_id: string;
  changed_factors: { factor: string; label: string; from: number; to: number }[];
  original: { probabilities: Record<string, number>; predicted_option: string };
  modified: { probabilities: Record<string, number>; predicted_option: string };
  flipped: boolean;
  deltas: Record<string, number>;
  prediction: Prediction;
}

export interface LandscapeSweep {
  factor: string;
  label: string;
  current_value: number;
  points: {
    value: number;
    probabilities: Record<string, number>;
    predicted_option: string;
  }[];
  flip_point: { between: [number, number]; from: string; to: string } | null;
  sensitivity: number;
}

export interface Landscape {
  prediction_id: string;
  options: string[];
  current_prediction: string;
  sweeps: LandscapeSweep[];
  note: string;
}

export interface Analytics {
  counts: {
    decisions: number;
    predictions: number;
    outcomes_recorded: number;
    questionnaire_answered: number;
  };
  accuracy: {
    checked: number;
    correct: number;
    accuracy: number | null;
    mean_brier: number | null;
  };
  model_confidence: {
    mean_trait_confidence: number;
    ml_ready: boolean;
    min_decisions_for_ml: number;
    decisions_until_ml: number;
  };
  active_model: {
    kind: string;
    version: string;
    trained_at: string;
    cv_accuracy: number;
    cv_brier: number;
    trained_on_decisions: number;
  } | null;
  behavioural_summary: {
    approach_rate: number | null;
    factor_profile: { factor: string; label: string; mean_magnitude: number }[];
    categories: { category: string; count: number }[];
  };
  recent_predictions: {
    id: string;
    created_at: string;
    predicted_option: string;
    predicted_probability: number;
    confidence_label: string;
    has_outcome: boolean;
    was_correct: boolean | null;
  }[];
  traits: { key: string; value: number; confidence: number; evidence_count: number }[];
}

export interface Performance {
  n: number;
  message?: string;
  metrics: {
    n: number;
    accuracy: number;
    baseline_accuracy: number;
    brier: number;
    log_loss: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number | null;
    ece: number;
    mean_confidence: number;
    calibration_bins: {
      lower: number;
      upper: number;
      count: number;
      mean_confidence: number;
      observed_accuracy: number;
      gap: number;
    }[];
    by_category: Record<string, { n: number; accuracy: number; brier: number }>;
  } | null;
  timeline: {
    recorded_at: string;
    predicted_option: string;
    actual_option: string;
    was_correct: boolean;
    predicted_probability: number;
    probability_of_actual: number;
    brier_score: number;
    confidence_label: string;
    running_accuracy: number;
  }[];
  by_confidence: Record<
    string,
    { n: number; correct: number; accuracy: number; mean_probability: number }
  >;
  calibration_temperature?: number;
  interpretation?: string;
}

export interface EvaluationMetrics {
  n: number;
  accuracy: number;
  brier: number;
  log_loss: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  ece: number;
}

export interface Evaluation {
  status: "ok" | "insufficient_data";
  message?: string;
  n_decisions: number;
  split?: Record<string, { n: number; range: [string, string] | null }>;
  approach_base_rate?: number;
  fitted_params?: Record<string, number>;
  results?: Record<string, EvaluationMetrics>;
  best_by_brier?: string;
  note?: string;
}

export interface QuestionnaireItem {
  key: string;
  construct: string;
  prompt: string;
  trait: string;
  reverse_scored: boolean;
  scale: string[];
  current_answer: number | null;
}

export interface TextAnalysis {
  candidates: {
    situation: string;
    options: string[];
    decision: string;
    reason: string;
    confidence: number;
    factors: Record<string, number>;
    emotions: Record<string, number>;
    source_excerpt: string;
    needs_review: boolean;
  }[];
  values: string[];
  goals: string[];
  emotions: Record<string, number>;
  factors: Record<string, number>;
  recurring_terms: { term: string; count: number }[];
  word_count: number;
  note: string;
}

export interface OutcomeResult {
  outcome_id: string;
  was_correct: boolean;
  probability_of_actual: number;
  brier_score: number;
  log_loss: number;
  promoted_decision_id: string | null;
  retrained: boolean;
  calibration_temperature: number;
  message: string;
}
