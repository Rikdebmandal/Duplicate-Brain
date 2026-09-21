"""Building, persisting and reporting the person model."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.logging_conf import get_logger
from app.models import (
    BehavioralTrait,
    Decision,
    Feedback,
    MemoryItem,
    QuestionnaireResponse,
    Scenario,
    TraitEvidence,
    User,
)
from app.services.decision_engine.parser import OptionStance, resolve_stances
from app.services.decision_engine.taxonomy import (
    FACTOR_BY_KEY,
    FACTOR_KEYS,
    TRAIT_BY_KEY,
    TRAIT_KEYS,
)
from app.services.decision_engine.utility import ModelParams, UtilityModel, softmax
from app.services.profile import questionnaire as qn
from app.services.profile.inference import (
    Observation,
    TraitPosterior,
    accumulate,
    observations_from_decision,
    recency_weight,
)

log = get_logger(__name__)

#: Temperature grid for the calibration search. Low = a highly consistent
#: decider; high = one whose choices vary given the same trade-off.
_TEMPERATURE_GRID = [0.20, 0.30, 0.40, 0.55, 0.70, 0.90, 1.20, 1.60, 2.20]
#: Status-quo bias grid, in utility units applied to the passive branch. The
#: range is wide enough that a fitted value landing on an endpoint means the
#: person really is that lopsided, rather than the grid having run out.
_BIAS_GRID = [-1.5, -1.0, -0.7, -0.45, -0.25, -0.1, 0.0,
              0.1, 0.25, 0.45, 0.7, 1.0, 1.5]


@dataclass
class TrainingCase:
    """One historical decision reduced to what the models need."""

    decision_id: str
    factors: dict[str, float]
    stances: list[OptionStance]
    chosen_index: int
    occurred_at: datetime
    importance: int
    category: str
    scenario_text: str
    reason: str

    @property
    def chosen_stance(self) -> OptionStance:
        return self.stances[self.chosen_index]


def load_training_cases(db: Session, user_id: str) -> list[TrainingCase]:
    """Every historical decision, in chronological order."""
    rows = db.execute(
        select(Decision, Scenario)
        .join(Scenario, Decision.scenario_id == Scenario.id)
        .where(Decision.user_id == user_id)
        .order_by(Decision.occurred_at.asc())
    ).all()

    cases: list[TrainingCase] = []
    for decision, scenario in rows:
        options = sorted(decision.options, key=lambda o: o.position)
        if len(options) < 2:
            continue
        stances: list[OptionStance] = []
        chosen_index = -1
        # Stances are cached at write time; recompute the whole set together if
        # any are missing, since a stance is only meaningful against the others.
        fallback = (
            resolve_stances([o.label for o in options])
            if any(not o.stance for o in options) else None
        )
        for idx, option in enumerate(options):
            stance_data = option.stance or {}
            if stance_data:
                stances.append(
                    OptionStance(
                        label=option.label,
                        approach=float(stance_data.get("approach", 0.5)),
                        defer=float(stance_data.get("defer", 0.0)),
                        compromise=float(stance_data.get("compromise", 0.0)),
                        confidence=float(stance_data.get("confidence", 0.0)),
                        matched_cues=list(stance_data.get("matched_cues", [])),
                    )
                )
            else:
                stances.append(fallback[idx])
            if option.was_chosen:
                chosen_index = idx
        if chosen_index < 0:
            continue
        cases.append(
            TrainingCase(
                decision_id=decision.id,
                factors=dict(scenario.factors or {}),
                stances=stances,
                chosen_index=chosen_index,
                occurred_at=decision.occurred_at,
                importance=decision.importance,
                category=scenario.category or "general",
                scenario_text=scenario.text,
                reason=decision.reason or "",
            )
        )
    return cases


class ProfileService:
    """Owns the person model: traits, their evidence, and fitted parameters."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # -- reading -----------------------------------------------------------
    def trait_rows(self, user_id: str) -> list[BehavioralTrait]:
        return list(
            self.db.execute(
                select(BehavioralTrait).where(BehavioralTrait.user_id == user_id)
            ).scalars().all()
        )

    def traits_dict(self, user_id: str) -> dict[str, float]:
        """Trait values for scoring, with any user override applied."""
        values = {key: 0.5 for key in TRAIT_KEYS}
        for row in self.trait_rows(user_id):
            values[row.trait_key] = (
                row.user_override if row.user_override is not None else row.value
            )
        return values

    def model_params(self, user_id: str) -> ModelParams:
        user = self.db.get(User, user_id)
        return ModelParams.from_dict((user.model_params if user else None) or {})

    def utility_model(self, user_id: str) -> UtilityModel:
        return UtilityModel(self.traits_dict(user_id), self.model_params(user_id))

    # -- evidence gathering -------------------------------------------------
    def _questionnaire_observations(self, user_id: str) -> list[Observation]:
        rows = self.db.execute(
            select(QuestionnaireResponse).where(QuestionnaireResponse.user_id == user_id)
        ).scalars().all()
        observations: list[Observation] = []
        for row in rows:
            item = qn.ITEMS_BY_KEY.get(row.item_key)
            if item is None:
                continue
            observations.append(
                Observation(
                    trait=item.trait,
                    value=float(row.value),
                    weight=1.0,
                    source="questionnaire",
                    note=f'self-report: "{item.prompt}"',
                    detail={"item": row.item_key, "raw_value": row.raw_value},
                )
            )
        return observations

    def _feedback_observations(self, user_id: str) -> list[Observation]:
        """Explicit corrections count heavily - the person is the ground truth."""
        rows = self.db.execute(
            select(Feedback).where(
                Feedback.user_id == user_id,
                Feedback.kind == "trait_correction",
            )
        ).scalars().all()
        observations: list[Observation] = []
        for row in rows:
            if row.target_key not in TRAIT_KEYS or row.value is None:
                continue
            observations.append(
                Observation(
                    trait=row.target_key,
                    value=float(min(1.0, max(0.0, row.value))),
                    weight=3.0 * recency_weight(row.created_at),
                    source="feedback",
                    note=row.comment or "corrected by the person directly",
                    detail={"feedback_id": row.id},
                )
            )
        return observations

    def collect_observations(
        self, user_id: str, cases: Sequence[TrainingCase] | None = None
    ) -> list[Observation]:
        cases = list(cases) if cases is not None else load_training_cases(self.db, user_id)
        observations: list[Observation] = []
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
        observations.extend(self._questionnaire_observations(user_id))
        observations.extend(self._feedback_observations(user_id))
        return observations

    # -- parameter fitting ---------------------------------------------------
    @staticmethod
    def fit_params(
        traits: dict[str, float],
        cases: Sequence[TrainingCase],
    ) -> tuple[ModelParams, float | None]:
        """Fit decision temperature and status-quo bias by grid search.

        Only these two parameters are fitted, and ``utility_scale`` is pinned at
        1.0: scale and temperature are the same degree of freedom (only their
        ratio affects the softmax), so fitting both would be unidentifiable.

        The objective is mean negative log-likelihood of the options actually
        chosen, with a mild pull towards the defaults so that a handful of
        decisions cannot produce an extreme, over-confident model.
        """
        default = ModelParams()
        if len(cases) < 4:
            # NaN would serialise to invalid JSON; "not fitted" is None.
            return default, None

        best_params, best_loss = default, float("inf")
        for temperature in _TEMPERATURE_GRID:
            for bias in _BIAS_GRID:
                params = ModelParams(temperature=temperature, status_quo_bias=bias)
                model = UtilityModel(traits, params)
                loss = 0.0
                for case in cases:
                    utilities = [
                        u.utility for u in model.score_options(case.factors, case.stances)
                    ]
                    probs = softmax(utilities, temperature)
                    p = max(1e-9, probs[case.chosen_index])
                    loss -= math.log(p)
                loss /= len(cases)
                # Regularisation: keep the fit near sane defaults on small data.
                loss += 0.05 * ((temperature - 0.55) ** 2) + 0.15 * (bias ** 2)
                if loss < best_loss:
                    best_loss, best_params = loss, params

        return best_params, round(best_loss, 5)

    # -- rebuilding ----------------------------------------------------------
    def rebuild(self, user_id: str) -> dict:
        """Recompute the whole person model from scratch.

        Rebuilding rather than incrementally updating is deliberate: every
        trait must always be reproducible from the evidence currently on
        record, so that deleting a decision actually removes its influence.
        """
        cases = load_training_cases(self.db, user_id)
        observations = self.collect_observations(user_id, cases)
        posteriors = accumulate(observations)

        overrides = {
            row.trait_key: row.user_override
            for row in self.trait_rows(user_id)
            if row.user_override is not None
        }

        self._persist_traits(user_id, posteriors, overrides)
        self._persist_evidence(user_id, observations)

        traits = {
            key: (overrides.get(key) if overrides.get(key) is not None else post.value)
            for key, post in posteriors.items()
        }
        params, loss = self.fit_params(traits, cases)

        user = self.db.get(User, user_id)
        if user is not None:
            user.model_params = params.to_dict()
            user.profile_refreshed_at = datetime.now(timezone.utc)

        patterns = self._synthesise_memory(user_id, cases, posteriors)
        self.db.flush()

        log.info(
            "rebuilt profile for user=%s decisions=%d observations=%d nll=%s",
            user_id, len(cases), len(observations), loss,
        )
        return {
            "decisions": len(cases),
            "observations": len(observations),
            "fitted_params": params.to_dict(),
            "fit_loss": loss,
            "patterns": patterns,
        }

    def _persist_traits(
        self,
        user_id: str,
        posteriors: dict[str, TraitPosterior],
        overrides: dict[str, float | None],
    ) -> None:
        existing = {row.trait_key: row for row in self.trait_rows(user_id)}
        for key, posterior in posteriors.items():
            row = existing.get(key)
            if row is None:
                row = BehavioralTrait(user_id=user_id, trait_key=key)
                self.db.add(row)
            row.value = round(posterior.value, 6)
            row.confidence = round(posterior.confidence, 6)
            row.evidence_count = posterior.evidence_count
            row.alpha = round(posterior.alpha, 6)
            row.beta = round(posterior.beta, 6)
            row.supporting_decisions = posterior.supporting_decisions
            if key in overrides:
                row.user_override = overrides[key]

    def _persist_evidence(self, user_id: str, observations: Sequence[Observation]) -> None:
        """Replace the evidence log so it always mirrors the current traits."""
        self.db.execute(delete(TraitEvidence).where(TraitEvidence.user_id == user_id))
        for obs in observations:
            self.db.add(
                TraitEvidence(
                    user_id=user_id,
                    trait_key=obs.trait,
                    decision_id=obs.decision_id,
                    source=obs.source,
                    observation=obs.value,
                    weight=obs.weight,
                    note=obs.note,
                    detail=obs.detail,
                )
            )

    # -- memory --------------------------------------------------------------
    def _synthesise_memory(
        self,
        user_id: str,
        cases: Sequence[TrainingCase],
        posteriors: dict[str, TraitPosterior],
    ) -> list[dict]:
        """Derive the preference and pattern memory layers.

        Facts are never synthesised - they only ever come from the person. That
        separation is what lets the UI show "you told us this" and "we inferred
        this" as different kinds of claim.
        """
        self.db.execute(
            delete(MemoryItem).where(
                MemoryItem.user_id == user_id,
                MemoryItem.kind.in_(["preference", "pattern"]),
            )
        )
        now = datetime.now(timezone.utc)
        summaries: list[dict] = []

        # Preferences: traits that are both confidently estimated and non-neutral.
        for key, posterior in posteriors.items():
            if posterior.confidence < 0.25 or abs(posterior.value - 0.5) < 0.1:
                continue
            trait = TRAIT_BY_KEY[key]
            leaning = trait.high_label if posterior.value > 0.5 else trait.low_label
            self.db.add(
                MemoryItem(
                    user_id=user_id,
                    kind="preference",
                    key=key,
                    content=f"Estimated to lean {leaning} on {trait.label.lower()}.",
                    importance=round(abs(posterior.value - 0.5) * 2, 4),
                    confidence=round(posterior.confidence, 4),
                    evidence_count=posterior.evidence_count,
                    detail={
                        "value": round(posterior.value, 4),
                        "credible_interval": list(posterior.credible_interval()),
                        "consistency": round(posterior.consistency, 4),
                    },
                    last_seen_at=now,
                )
            )

        # Patterns: literal counts over the person's own history. These are
        # observations, not inferences, and are worded as such.
        for factor_key in FACTOR_KEYS:
            factor = FACTOR_BY_KEY[factor_key]
            if factor.sign_under_approach == 0:
                continue
            relevant = [c for c in cases if c.factors.get(factor_key, 0.0) >= 0.45]
            if len(relevant) < 3:
                continue
            took = sum(1 for c in relevant if c.chosen_stance.approach >= 0.5)
            total = len(relevant)
            rate = took / total
            if 0.34 < rate < 0.66:
                continue
            direction = "took the active option" if rate >= 0.5 else "chose the safer option"
            count = took if rate >= 0.5 else total - took
            content = (
                f"In {count} of {total} recorded decisions where "
                f"{factor.label.lower()} was substantial, {direction}."
            )
            self.db.add(
                MemoryItem(
                    user_id=user_id,
                    kind="pattern",
                    key=f"pattern:{factor_key}",
                    content=content,
                    importance=round(min(1.0, total / 10.0), 4),
                    confidence=round(abs(rate - 0.5) * 2, 4),
                    evidence_count=total,
                    detail={
                        "factor": factor_key,
                        "approach_rate": round(rate, 4),
                        "n": total,
                        "decision_ids": [c.decision_id for c in relevant][:10],
                    },
                    last_seen_at=now,
                )
            )
            summaries.append({"factor": factor_key, "content": content, "n": total})

        return summaries

    # -- reporting ------------------------------------------------------------
    def profile_payload(self, user_id: str) -> dict:
        """The full behavioural profile, labelled observed vs inferred."""
        rows = {row.trait_key: row for row in self.trait_rows(user_id)}
        cases = load_training_cases(self.db, user_id)
        params = self.model_params(user_id)

        traits: list[dict] = []
        for key in TRAIT_KEYS:
            trait = TRAIT_BY_KEY[key]
            row = rows.get(key)
            value = row.value if row else 0.5
            effective = row.user_override if (row and row.user_override is not None) else value
            posterior = TraitPosterior(
                trait=key,
                alpha=row.alpha if row else 1.0,
                beta=row.beta if row else 1.0,
                evidence_count=row.evidence_count if row else 0,
            )
            lo, hi = posterior.credible_interval()
            traits.append(
                {
                    "key": key,
                    "label": trait.label,
                    "description": trait.description,
                    "low_label": trait.low_label,
                    "high_label": trait.high_label,
                    "kind": trait.kind,
                    "value": round(float(effective), 4),
                    "inferred_value": round(float(value), 4),
                    "user_override": row.user_override if row else None,
                    "confidence": round(float(row.confidence) if row else 0.0, 4),
                    "evidence_count": int(row.evidence_count) if row else 0,
                    "credible_interval": [lo, hi],
                    "supporting_decisions": list(row.supporting_decisions) if row else [],
                    # Every trait is an estimate. The API says so on every row so
                    # that no client can render one as a fact by accident.
                    "status": "inferred" if (not row or row.user_override is None)
                              else "user_specified",
                }
            )

        memory = self.db.execute(
            select(MemoryItem).where(MemoryItem.user_id == user_id)
        ).scalars().all()

        return {
            "traits": traits,
            "model_params": params.to_dict(),
            "evidence_summary": {
                "decisions": len(cases),
                "questionnaire_items": self.db.execute(
                    select(QuestionnaireResponse).where(
                        QuestionnaireResponse.user_id == user_id
                    )
                ).scalars().all().__len__(),
                "trait_observations": self.db.execute(
                    select(TraitEvidence).where(TraitEvidence.user_id == user_id)
                ).scalars().all().__len__(),
                "ml_ready": len(cases) >= settings.min_decisions_for_ml,
                "min_decisions_for_ml": settings.min_decisions_for_ml,
            },
            "memory": {
                "facts": [_memory_payload(m) for m in memory if m.kind == "fact"],
                "preferences": [_memory_payload(m) for m in memory if m.kind == "preference"],
                "patterns": [_memory_payload(m) for m in memory if m.kind == "pattern"],
            },
            "disclaimer": (
                "These values are statistical estimates of decision-making tendencies "
                "inferred from the decisions on record. They are not psychological "
                "measurements, diagnoses, or statements of fact about this person."
            ),
        }

    def trait_evidence(self, user_id: str, trait_key: str, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            select(TraitEvidence)
            .where(TraitEvidence.user_id == user_id, TraitEvidence.trait_key == trait_key)
            .order_by(TraitEvidence.weight.desc())
            .limit(limit)
        ).scalars().all()
        return [
            {
                "id": row.id,
                "decision_id": row.decision_id,
                "source": row.source,
                "observation": round(row.observation, 4),
                "weight": round(row.weight, 4),
                "note": row.note,
                "detail": row.detail,
            }
            for row in rows
        ]


def _memory_payload(item: MemoryItem) -> dict:
    return {
        "id": item.id,
        "kind": item.kind,
        "key": item.key,
        "content": item.content,
        "importance": round(item.importance, 4),
        "confidence": round(item.confidence, 4),
        "evidence_count": item.evidence_count,
        "detail": item.detail,
        "last_updated": (item.last_seen_at or item.updated_at).isoformat()
        if (item.last_seen_at or item.updated_at) else None,
    }
