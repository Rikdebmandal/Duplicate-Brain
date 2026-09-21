"""The learning loop.

    prediction -> actual decision -> error -> updated model -> better prediction

Recording an outcome does four things:

1. scores the prediction (Brier, log loss, hit/miss);
2. **promotes the outcome into the decision history**, so the situation the
   person just resolved becomes training evidence like any other decision -
   this is the step that actually makes the twin improve rather than merely
   keep score;
3. refits the post-fusion calibration temperature, which is what corrects
   systematic over- or under-confidence;
4. rebuilds the profile and retrains the statistical layer once enough new
   evidence has accumulated.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.errors import ConflictError, NotFoundError
from app.logging_conf import get_logger
from app.ml import trainer
from app.ml.metrics import evaluate
from app.models import (
    Decision,
    DecisionFactor,
    DecisionOption,
    Feedback,
    Prediction,
    PredictionOutcome,
    Scenario,
    User,
)
from app.services.decision_engine.parser import resolve_stances
from app.services.decision_engine.taxonomy import TRAIT_KEYS
from app.services.embeddings.store import VectorStore
from app.services.prediction.fusion import apply_calibration
from app.services.profile.service import ProfileService, load_training_cases

log = get_logger(__name__)

#: Calibration temperature grid. >1 flattens an over-confident model.
_CALIBRATION_GRID = [0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0, 2.5, 3.0]
#: Below this many scored predictions, calibration is left at 1.0 - fitting it
#: on three outcomes would be noise dressed up as learning.
MIN_OUTCOMES_FOR_CALIBRATION = 8


@dataclass
class OutcomeResult:
    outcome_id: str
    was_correct: bool
    probability_of_actual: float
    brier_score: float
    log_loss: float
    promoted_decision_id: str | None
    retrained: bool
    calibration_temperature: float
    message: str

    def to_dict(self) -> dict:
        return {
            "outcome_id": self.outcome_id,
            "was_correct": self.was_correct,
            "probability_of_actual": round(self.probability_of_actual, 4),
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss, 4),
            "promoted_decision_id": self.promoted_decision_id,
            "retrained": self.retrained,
            "calibration_temperature": round(self.calibration_temperature, 4),
            "message": self.message,
        }


class FeedbackService:
    """Records reality and folds it back into the model."""

    def __init__(self, db: Session, user: User) -> None:
        self.db = db
        self.user = user
        self.profiles = ProfileService(db)
        self.store = VectorStore(db)

    # -- recording outcomes --------------------------------------------------
    def record_outcome(
        self,
        prediction_id: str,
        actual_option: str,
        *,
        note: str = "",
        importance: int = 5,
        reason: str = "",
        promote_to_history: bool = True,
    ) -> OutcomeResult:
        prediction = self.db.get(Prediction, prediction_id)
        if prediction is None or prediction.user_id != self.user.id:
            raise NotFoundError("Prediction not found.")
        if prediction.outcome is not None:
            raise ConflictError("An outcome has already been recorded for this prediction.")

        options = sorted(prediction.options, key=lambda o: o.position)
        labels = [o.label for o in options]
        probabilities = [float(o.probability) for o in options]

        actual_index = _match_option(actual_option, labels)
        if actual_index is None:
            raise NotFoundError(
                f"'{actual_option}' is not one of the options this prediction covered: {labels}"
            )
        resolved_label = labels[actual_index]

        p_actual = probabilities[actual_index]
        brier = sum(
            (p - (1.0 if i == actual_index else 0.0)) ** 2
            for i, p in enumerate(probabilities)
        )
        nll = -math.log(max(1e-9, p_actual))
        predicted_index = max(range(len(probabilities)), key=lambda i: probabilities[i])

        outcome = PredictionOutcome(
            user_id=self.user.id,
            prediction_id=prediction.id,
            actual_option=resolved_label,
            was_correct=predicted_index == actual_index,
            probability_of_actual=p_actual,
            brier_score=brier,
            log_loss=nll,
            note=note,
            recorded_at=datetime.now(timezone.utc),
        )
        self.db.add(outcome)
        self.db.flush()

        promoted_id = None
        if promote_to_history:
            promoted_id = self._promote_to_decision(
                prediction, resolved_label, labels, reason or note, importance
            )
            outcome.promoted_decision_id = promoted_id

        calibration_temperature = self.fit_calibration()
        retrained = self._maybe_retrain()

        self.db.flush()
        log.info(
            "outcome recorded user=%s prediction=%s correct=%s p_actual=%.3f",
            self.user.id, prediction.id, outcome.was_correct, p_actual,
        )

        return OutcomeResult(
            outcome_id=outcome.id,
            was_correct=outcome.was_correct,
            probability_of_actual=p_actual,
            brier_score=brier,
            log_loss=nll,
            promoted_decision_id=promoted_id,
            retrained=retrained,
            calibration_temperature=calibration_temperature,
            message=(
                "Prediction matched the actual decision."
                if outcome.was_correct
                else "Prediction did not match. The decision has been added to the "
                     "history and the model updated."
            ),
        )

    def _promote_to_decision(
        self,
        prediction: Prediction,
        actual_option: str,
        labels: Sequence[str],
        reason: str,
        importance: int,
    ) -> str:
        """Turn a resolved prediction into a first-class historical decision."""
        source_scenario = self.db.get(Scenario, prediction.scenario_id)
        scenario = Scenario(
            user_id=self.user.id,
            text=source_scenario.text if source_scenario else prediction.predicted_option,
            context=dict(source_scenario.context or {}) if source_scenario else {},
            category=source_scenario.category if source_scenario else "general",
            factors=dict(prediction.factors or {}),
            factor_source=source_scenario.factor_source if source_scenario else "lexical",
            coverage=source_scenario.coverage if source_scenario else 0.0,
            factor_evidence=dict(source_scenario.factor_evidence or {}) if source_scenario else {},
            emotions=dict(source_scenario.emotions or {}) if source_scenario else {},
            is_hypothetical=False,
        )
        self.db.add(scenario)
        self.db.flush()

        for key, magnitude in (prediction.factors or {}).items():
            self.db.add(
                DecisionFactor(
                    scenario_id=scenario.id,
                    factor_key=key,
                    magnitude=float(magnitude),
                    source="prediction_outcome",
                )
            )

        decision = Decision(
            user_id=self.user.id,
            scenario_id=scenario.id,
            chosen_option=actual_option,
            reason=reason,
            importance=max(1, min(10, importance)),
            occurred_at=datetime.now(timezone.utc),
            source="prediction_outcome",
        )
        self.db.add(decision)
        self.db.flush()

        stances = resolve_stances(list(labels))
        for index, (label, stance) in enumerate(zip(labels, stances, strict=False)):
            self.db.add(
                DecisionOption(
                    decision_id=decision.id,
                    label=label,
                    position=index,
                    was_chosen=(label == actual_option),
                    stance=stance.to_dict(),
                )
            )
        self.db.flush()
        self.store.index_decision(decision, scenario)
        return decision.id

    # -- calibration -----------------------------------------------------------
    def scored_predictions(self) -> list[tuple[list[float], int]]:
        """Every prediction that has a recorded outcome, as (probs, actual)."""
        rows = self.db.execute(
            select(Prediction, PredictionOutcome)
            .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
            .where(Prediction.user_id == self.user.id)
            .order_by(PredictionOutcome.recorded_at.asc())
        ).all()

        cases: list[tuple[list[float], int]] = []
        for prediction, outcome in rows:
            options = sorted(prediction.options, key=lambda o: o.position)
            if not options:
                continue
            labels = [o.label for o in options]
            index = _match_option(outcome.actual_option, labels)
            if index is None:
                continue
            cases.append(([float(o.probability) for o in options], index))
        return cases

    def fit_calibration(self) -> float:
        """Fit the post-fusion temperature that minimises log loss on outcomes."""
        cases = self.scored_predictions()
        params = self.profiles.model_params(self.user.id)
        if len(cases) < MIN_OUTCOMES_FOR_CALIBRATION:
            return params.calibration_temperature

        best_t, best_loss = 1.0, float("inf")
        for temperature in _CALIBRATION_GRID:
            loss = 0.0
            for probabilities, actual_index in cases:
                adjusted = apply_calibration(probabilities, temperature)
                loss -= math.log(max(1e-9, adjusted[actual_index]))
            loss /= len(cases)
            # Mild pull towards 1.0 so a short streak cannot swing it hard.
            loss += 0.08 * ((temperature - 1.0) ** 2)
            if loss < best_loss:
                best_loss, best_t = loss, temperature

        stored = dict(self.user.model_params or {})
        stored["calibration_temperature"] = best_t
        self.user.model_params = stored
        log.info(
            "calibration temperature for user=%s set to %.2f on %d outcomes",
            self.user.id, best_t, len(cases),
        )
        return best_t

    def _maybe_retrain(self) -> bool:
        """Rebuild the profile and refit the model on a cadence, not every time."""
        total = self.db.execute(
            select(func.count(PredictionOutcome.id)).where(
                PredictionOutcome.user_id == self.user.id
            )
        ).scalar_one()
        if total % max(1, settings.retrain_every_n_outcomes) != 0:
            return False

        self.profiles.rebuild(self.user.id)
        cases = load_training_cases(self.db, self.user.id)
        model = self.profiles.utility_model(self.user.id)
        trainer.train_for_user(self.db, self.user.id, cases, model)
        return True

    # -- corrections -------------------------------------------------------------
    def submit_feedback(
        self,
        *,
        kind: str,
        target_key: str = "",
        value: float | None = None,
        comment: str = "",
        prediction_id: str | None = None,
        detail: dict | None = None,
    ) -> Feedback:
        """Record a correction from the person and apply it where it belongs."""
        if kind == "trait_correction":
            if target_key not in TRAIT_KEYS:
                raise NotFoundError(f"'{target_key}' is not a known trait.")
            if value is None or not 0.0 <= float(value) <= 1.0:
                raise ConflictError("A trait correction needs a value between 0 and 1.")

        record = Feedback(
            user_id=self.user.id,
            prediction_id=prediction_id,
            kind=kind,
            target_key=target_key,
            value=value,
            comment=comment,
            detail=detail or {},
            applied=False,
        )
        self.db.add(record)
        self.db.flush()

        if kind == "trait_correction":
            # Corrections are treated as strong evidence rather than a hard
            # override, so later behaviour can still move the trait. A hard
            # override is available separately on the profile endpoint.
            self.profiles.rebuild(self.user.id)
            record.applied = True

        self.db.flush()
        return record

    def override_trait(self, trait_key: str, value: float | None) -> dict:
        """Pin a trait to a user-specified value, or release it back to inference."""
        if trait_key not in TRAIT_KEYS:
            raise NotFoundError(f"'{trait_key}' is not a known trait.")
        rows = {r.trait_key: r for r in self.profiles.trait_rows(self.user.id)}
        row = rows.get(trait_key)
        if row is None:
            self.profiles.rebuild(self.user.id)
            rows = {r.trait_key: r for r in self.profiles.trait_rows(self.user.id)}
            row = rows.get(trait_key)
        if row is None:  # pragma: no cover - rebuild always creates every trait
            raise NotFoundError(f"Trait '{trait_key}' is not on record.")

        row.user_override = None if value is None else float(min(1.0, max(0.0, value)))
        self.db.flush()
        return {
            "trait": trait_key,
            "user_override": row.user_override,
            "inferred_value": round(row.value, 4),
            "status": "user_specified" if row.user_override is not None else "inferred",
        }

    # -- reporting -----------------------------------------------------------------
    def performance(self) -> dict:
        """Accuracy and calibration over the person's own prediction history."""
        rows = self.db.execute(
            select(Prediction, PredictionOutcome)
            .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
            .where(Prediction.user_id == self.user.id)
            .order_by(PredictionOutcome.recorded_at.asc())
        ).all()

        if not rows:
            return {
                "n": 0,
                "message": "No predictions have been checked against reality yet.",
                "metrics": None,
                "timeline": [],
                "by_confidence": {},
            }

        cases: list[tuple[list[float], int]] = []
        categories: list[str] = []
        timeline: list[dict] = []
        by_confidence: dict[str, dict[str, float]] = {}

        running_hits = 0
        for index, (prediction, outcome) in enumerate(rows, start=1):
            options = sorted(prediction.options, key=lambda o: o.position)
            labels = [o.label for o in options]
            actual_index = _match_option(outcome.actual_option, labels)
            if actual_index is None:
                continue
            probabilities = [float(o.probability) for o in options]
            cases.append((probabilities, actual_index))

            scenario = self.db.get(Scenario, prediction.scenario_id)
            categories.append(scenario.category if scenario else "general")

            running_hits += int(outcome.was_correct)
            timeline.append(
                {
                    "recorded_at": outcome.recorded_at.isoformat(),
                    "predicted_option": prediction.predicted_option,
                    "actual_option": outcome.actual_option,
                    "was_correct": outcome.was_correct,
                    "predicted_probability": round(prediction.predicted_probability, 4),
                    "probability_of_actual": round(outcome.probability_of_actual, 4),
                    "brier_score": round(outcome.brier_score, 4),
                    "confidence_label": prediction.confidence_label,
                    "running_accuracy": round(running_hits / index, 4),
                }
            )

            bucket = by_confidence.setdefault(
                prediction.confidence_label, {"n": 0, "correct": 0, "mean_probability": 0.0}
            )
            bucket["n"] += 1
            bucket["correct"] += int(outcome.was_correct)
            bucket["mean_probability"] += prediction.predicted_probability

        for bucket in by_confidence.values():
            n = max(1, bucket["n"])
            bucket["accuracy"] = round(bucket["correct"] / n, 4)
            bucket["mean_probability"] = round(bucket["mean_probability"] / n, 4)

        base_rate = _approach_base_rate(self.db, self.user.id)
        baseline_cases = [
            (_baseline_probabilities(len(probs), base_rate), actual)
            for probs, actual in cases
        ]
        result = evaluate(cases, categories=categories, baseline_cases=baseline_cases)

        return {
            "n": result.n,
            "metrics": result.to_dict(),
            "timeline": timeline,
            "by_confidence": by_confidence,
            "calibration_temperature": self.profiles.model_params(
                self.user.id
            ).calibration_temperature,
            "interpretation": _interpret(result),
        }


def _match_option(value: str, labels: Sequence[str]) -> int | None:
    """Match an option by exact text first, then case-insensitively."""
    for index, label in enumerate(labels):
        if label == value:
            return index
    lowered = (value or "").strip().lower()
    for index, label in enumerate(labels):
        if label.strip().lower() == lowered:
            return index
    return None


def _approach_base_rate(db: Session, user_id: str) -> float:
    cases = load_training_cases(db, user_id)
    if not cases:
        return 0.5
    return sum(c.chosen_stance.approach for c in cases) / len(cases)


def _baseline_probabilities(n_options: int, base_rate: float) -> list[float]:
    """A crude "always predict the usual thing" reference distribution."""
    if n_options <= 0:
        return []
    weights = [base_rate] + [(1.0 - base_rate) / max(1, n_options - 1)] * (n_options - 1)
    total = sum(weights)
    return [w / total for w in weights]


def _interpret(result) -> str:
    if result.n < 5:
        return (
            "Too few checked predictions to judge performance. Record more actual "
            "decisions before reading anything into these numbers."
        )
    parts = []
    if result.accuracy > result.baseline_accuracy + 0.05:
        parts.append(
            f"The model beats the always-predict-the-usual baseline "
            f"({result.accuracy:.0%} vs {result.baseline_accuracy:.0%})."
        )
    elif result.accuracy < result.baseline_accuracy - 0.05:
        parts.append(
            f"The model is currently worse than the baseline "
            f"({result.accuracy:.0%} vs {result.baseline_accuracy:.0%}), which usually "
            f"means the recorded decisions are too varied for the patterns found so far."
        )
    else:
        parts.append("The model is roughly level with a simple base-rate baseline.")

    if result.ece > 0.15:
        direction = "over-confident" if result.mean_confidence > result.accuracy else "under-confident"
        parts.append(
            f"Probabilities are poorly calibrated (ECE {result.ece:.2f}) and currently "
            f"{direction}; the calibration temperature will correct for this as more "
            f"outcomes are recorded."
        )
    else:
        parts.append(f"Probabilities are reasonably well calibrated (ECE {result.ece:.2f}).")
    return " ".join(parts)
