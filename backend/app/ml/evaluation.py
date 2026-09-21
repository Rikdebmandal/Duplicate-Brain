"""Offline evaluation harness: baseline vs ML vs retrieval vs LLM vs hybrid.

Two properties make this harness honest, and both are easy to get wrong:

**Time-based splitting.** Decisions are ordered by when they happened and cut
into train / validation / test blocks. The test block is always in the future
relative to the training block, because the question the system claims to
answer is "given the past, what happens next" - not "given a random 75% of a
life, interpolate the rest".

**No profile leakage.** Traits and fitted parameters are recomputed *from the
training block alone*. Reusing the stored profile would leak the test
decisions into the person model, since the stored profile was built from every
decision on record. That single mistake would inflate every number here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.logging_conf import get_logger
from app.ml.metrics import EvaluationResult, evaluate
from app.ml.trainer import _fit_estimator, build_dataset
from app.services.decision_engine.features import build_features
from app.services.decision_engine.taxonomy import TRAIT_KEYS
from app.services.decision_engine.utility import UtilityModel, softmax
from app.services.embeddings.embedder import cosine, get_embedder
from app.services.embeddings.store import factor_similarity
from app.services.prediction import fusion
from app.services.prediction.retrieval import RETRIEVAL_SHARPNESS, baseline_distribution
from app.services.profile.inference import accumulate, observations_from_decision
from app.services.profile.service import ProfileService, TrainingCase

log = get_logger(__name__)

#: Minimum decisions before an evaluation is meaningful at all.
MIN_CASES = 10


@dataclass
class Split:
    train: list[TrainingCase]
    validation: list[TrainingCase]
    test: list[TrainingCase]

    def to_dict(self) -> dict:
        def bounds(cases: Sequence[TrainingCase]) -> list[str] | None:
            if not cases:
                return None
            return [cases[0].occurred_at.isoformat(), cases[-1].occurred_at.isoformat()]

        return {
            "train": {"n": len(self.train), "range": bounds(self.train)},
            "validation": {"n": len(self.validation), "range": bounds(self.validation)},
            "test": {"n": len(self.test), "range": bounds(self.test)},
        }


def time_split(
    cases: Sequence[TrainingCase],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> Split:
    ordered = sorted(cases, key=lambda c: c.occurred_at)
    n = len(ordered)
    train_end = max(1, int(n * train_fraction))
    validation_end = max(train_end, int(n * (train_fraction + validation_fraction)))
    return Split(
        train=ordered[:train_end],
        validation=ordered[train_end:validation_end],
        test=ordered[validation_end:],
    )


def traits_from_cases(cases: Sequence[TrainingCase]) -> dict[str, float]:
    """Rebuild the person model from a subset - the anti-leakage step."""
    observations = []
    for case in cases:
        observations.extend(
            observations_from_decision(
                decision_id=case.decision_id,
                factors=case.factors,
                chosen_stance=case.chosen_stance,
                importance=case.importance,
                occurred_at=case.occurred_at,
                reason=case.reason,
            )
        )
    posteriors = accumulate(observations)
    return {key: posteriors[key].value for key in TRAIT_KEYS}


class InMemoryRetriever:
    """Nearest-neighbour search restricted to an explicit case list.

    The production store queries every decision the user has. During evaluation
    that would hand the model its own answers, so retrieval is rebuilt here over
    the training block only.
    """

    def __init__(self, cases: Sequence[TrainingCase]) -> None:
        self.embedder = get_embedder()
        self.cases = list(cases)
        self.vectors = [self.embedder.embed(c.scenario_text) for c in self.cases]

    def predict(self, case: TrainingCase, k: int = 5) -> tuple[list[float], float]:
        if not self.cases:
            n = len(case.stances)
            return [1.0 / n] * n, 0.0

        query = self.embedder.embed(case.scenario_text)
        scored: list[tuple[float, TrainingCase]] = []
        for vector, other in zip(self.vectors, self.cases, strict=False):
            text_sim = max(0.0, cosine(query, vector))
            fact_sim = max(0.0, factor_similarity(case.factors, other.factors))
            scored.append((0.55 * text_sim + 0.45 * fact_sim, other))
        scored.sort(key=lambda item: -item[0])
        top = [item for item in scored[:k] if item[0] >= 0.12]

        n_options = len(case.stances)
        if not top:
            return [1.0 / n_options] * n_options, 0.0

        weights = [0.0] * n_options
        for similarity, other in top:
            importance = 0.6 + (other.importance or 5) / 10.0
            for index, stance in enumerate(case.stances):
                match = 1.0 - abs(stance.approach - other.chosen_stance.approach)
                weights[index] += similarity * importance * match

        total = sum(weights)
        if total <= 0:
            return [1.0 / n_options] * n_options, 0.0
        shares = [w / total for w in weights]
        support = min(1.0, top[0][0] * min(1.0, len(top) / 3.0) * 1.4)
        return softmax([s * RETRIEVAL_SHARPNESS for s in shares], 1.0), support


def _profile_probabilities(model: UtilityModel, case: TrainingCase) -> list[float]:
    utilities = [u.utility for u in model.score_options(case.factors, case.stances)]
    return softmax(utilities, model.params.temperature)


def run_evaluation(
    cases: Sequence[TrainingCase],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    llm_predictor=None,
) -> dict:
    """Compare every approach on a held-out future block of decisions."""
    ordered = sorted(cases, key=lambda c: c.occurred_at)
    if len(ordered) < MIN_CASES:
        return {
            "status": "insufficient_data",
            "message": (
                f"Evaluation needs at least {MIN_CASES} decisions; "
                f"{len(ordered)} are on record."
            ),
            "n_decisions": len(ordered),
        }

    split = time_split(ordered, train_fraction, validation_fraction)
    if not split.test:
        return {
            "status": "insufficient_data",
            "message": "The time split left no decisions in the test block.",
            "n_decisions": len(ordered),
        }

    fit_cases = split.train + split.validation
    traits = traits_from_cases(split.train)
    params, _ = ProfileService.fit_params(traits, fit_cases)
    model = UtilityModel(traits, params)
    approach_base_rate = (
        sum(c.chosen_stance.approach for c in fit_cases) / len(fit_cases)
        if fit_cases else 0.5
    )

    # -- fit the statistical layer on the training block only -----------------
    estimator = None
    X_train, y_train, _ = build_dataset(fit_cases, model)
    if len(X_train) and len(set(y_train.tolist())) >= 2:
        try:
            estimator = _fit_estimator("logistic", X_train, y_train)
        except Exception as exc:  # pragma: no cover - degenerate data
            log.warning("evaluation training failed: %s", exc)

    retriever = InMemoryRetriever(fit_cases)

    predictions: dict[str, list[tuple[list[float], int]]] = {
        "baseline": [], "profile": [], "ml": [], "retrieval": [], "hybrid": [],
    }
    if llm_predictor is not None:
        predictions["llm"] = []
    categories: list[str] = []

    for case in split.test:
        categories.append(case.category)
        n_options = len(case.stances)

        base = baseline_distribution(case.stances, approach_base_rate)
        predictions["baseline"].append((base, case.chosen_index))

        profile_probs = _profile_probabilities(model, case)
        predictions["profile"].append((profile_probs, case.chosen_index))

        if estimator is not None:
            rows = np.array(
                [r.values for r in build_features(case.factors, case.stances, model)],
                dtype=float,
            )
            raw = np.clip(estimator.predict_proba(rows)[:, 1], 1e-6, 1 - 1e-6)
            ml_probs = list(raw / raw.sum())
        else:
            ml_probs = [1.0 / n_options] * n_options
        predictions["ml"].append((ml_probs, case.chosen_index))

        retrieval_probs, support = retriever.predict(case)
        predictions["retrieval"].append((retrieval_probs, case.chosen_index))

        layers = [
            fusion.LayerOutput("profile", profile_probs, 1.0),
            fusion.LayerOutput("statistical", ml_probs, 1.35 if estimator is not None else 0.0,
                               available=estimator is not None),
            fusion.LayerOutput("retrieval", retrieval_probs, 1.10 * support, available=support > 0),
        ]

        if llm_predictor is not None:
            llm_probs = llm_predictor(case, traits, retriever)
            if llm_probs is None:
                llm_probs = [1.0 / n_options] * n_options
                predictions["llm"].append((llm_probs, case.chosen_index))
            else:
                predictions["llm"].append((llm_probs, case.chosen_index))
                layers.append(fusion.LayerOutput("llm", llm_probs, 0.85))

        pooled = fusion.log_linear_pool(layers)
        predictions["hybrid"].append((pooled, case.chosen_index))

    results: dict[str, dict] = {}
    for name, cases_for_name in predictions.items():
        result: EvaluationResult = evaluate(
            cases_for_name,
            categories=categories,
            baseline_cases=predictions["baseline"],
        )
        results[name] = result.to_dict()

    ranked = sorted(results.items(), key=lambda item: item[1]["brier"])
    best_name = ranked[0][0]

    return {
        "status": "ok",
        "n_decisions": len(ordered),
        "split": split.to_dict(),
        "approach_base_rate": round(approach_base_rate, 4),
        "fitted_params": params.to_dict(),
        "traits_used": {k: round(v, 4) for k, v in traits.items()},
        "results": results,
        "best_by_brier": best_name,
        "note": (
            "Models are ranked by Brier score rather than accuracy: a well-calibrated "
            "model that admits uncertainty is more useful here than a confident one "
            "that is wrong at the moments that matter. Traits and parameters are "
            "refitted from the training block only, so the test block is genuinely held out."
        ),
    }
