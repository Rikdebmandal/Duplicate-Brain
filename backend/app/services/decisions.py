"""Creating, reading and deleting historical decisions.

Writing a decision is the only place the factor extraction runs against ground
truth, so it does the full job once and caches everything: the parsed factor
vector on the scenario, the per-factor rows for analytics, the option stances,
and the embeddings. Prediction then never re-parses history.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.errors import AppError, NotFoundError
from app.logging_conf import get_logger
from app.models import (
    Decision,
    DecisionFactor,
    DecisionOption,
    Scenario,
    User,
)
from app.services.decision_engine.parser import extract_emotions, parse_scenario
from app.services.decision_engine.taxonomy import FACTOR_KEYS
from app.services.embeddings.store import VectorStore

log = get_logger(__name__)


@dataclass
class DecisionInput:
    situation: str
    options: list[str]
    decision: str
    reason: str = ""
    context: dict | None = None
    category: str = "general"
    emotional_state: dict | None = None
    importance: int = 5
    occurred_at: datetime | None = None
    outcome: str = ""
    satisfaction: float | None = None
    expected_reward: float = 0.0
    perceived_risk: float = 0.0
    factor_overrides: dict[str, float] | None = None
    source: str = "manual"


class DecisionService:
    def __init__(self, db: Session, user: User) -> None:
        self.db = db
        self.user = user
        self.store = VectorStore(db)

    # -- writing ------------------------------------------------------------
    def create(self, payload: DecisionInput) -> Decision:
        options = [o.strip() for o in payload.options if o and o.strip()]
        if len(options) < 2:
            raise AppError("A decision needs at least two options to be informative.")

        chosen = payload.decision.strip()
        matched = _match(chosen, options)
        if matched is None:
            # Accepting the choice as an extra option would silently corrupt the
            # option set, so this is a hard error with a helpful message.
            raise AppError(
                f"The chosen option '{chosen}' is not in the option list {options}."
            )

        parsed = parse_scenario(
            payload.situation,
            options,
            context_text=" ".join(
                str(v) for v in (payload.context or {}).values()
                if isinstance(v, str | int | float)
            ) + " " + (payload.reason or ""),
            factor_overrides=payload.factor_overrides,
        )

        emotions = dict(payload.emotional_state or {})
        if not emotions:
            emotions = extract_emotions(f"{payload.situation} {payload.reason}")

        scenario = Scenario(
            user_id=self.user.id,
            text=payload.situation,
            context=payload.context or {},
            category=(payload.category or "general").lower()[:64],
            factors=parsed.factors,
            factor_source="user" if payload.factor_overrides else "lexical",
            coverage=parsed.coverage,
            factor_evidence={"hits": [h.to_dict() for h in parsed.hits[:60]]},
            emotions=emotions,
            is_hypothetical=False,
        )
        self.db.add(scenario)
        self.db.flush()

        for key in FACTOR_KEYS:
            magnitude = float(parsed.factors.get(key, 0.0))
            if magnitude <= 0.0:
                continue
            self.db.add(
                DecisionFactor(
                    scenario_id=scenario.id,
                    factor_key=key,
                    magnitude=magnitude,
                    source=scenario.factor_source,
                    evidence={
                        "phrases": [
                            h.context for h in parsed.hits
                            if h.factor == key and not h.negated
                        ][:4]
                    },
                )
            )

        decision = Decision(
            user_id=self.user.id,
            scenario_id=scenario.id,
            chosen_option=options[matched],
            reason=payload.reason or "",
            emotional_state=emotions,
            importance=max(1, min(10, int(payload.importance or 5))),
            expected_reward=float(payload.expected_reward or 0.0),
            perceived_risk=float(payload.perceived_risk or 0.0),
            outcome=payload.outcome or "",
            satisfaction=payload.satisfaction,
            occurred_at=payload.occurred_at or datetime.now(timezone.utc),
            source=payload.source,
        )
        self.db.add(decision)
        self.db.flush()

        for index, (label, stance) in enumerate(zip(options, parsed.options, strict=False)):
            self.db.add(
                DecisionOption(
                    decision_id=decision.id,
                    label=label,
                    position=index,
                    was_chosen=(index == matched),
                    stance=stance.to_dict(),
                )
            )
        self.db.flush()
        self.store.index_decision(decision, scenario)
        self.db.flush()
        return decision

    def delete(self, decision_id: str) -> None:
        decision = self.get(decision_id)
        scenario = self.db.get(Scenario, decision.scenario_id)
        self.db.delete(decision)
        if scenario is not None:
            # The scenario exists for this decision only; removing both is what
            # makes a deletion actually erase the evidence.
            self.db.delete(scenario)
        self.db.flush()

    # -- reading ---------------------------------------------------------------
    def get(self, decision_id: str) -> Decision:
        decision = self.db.get(Decision, decision_id)
        if decision is None or decision.user_id != self.user.id:
            raise NotFoundError("Decision not found.")
        return decision

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        search: str | None = None,
    ) -> tuple[list[dict], int]:
        query = (
            select(Decision, Scenario)
            .join(Scenario, Decision.scenario_id == Scenario.id)
            .where(Decision.user_id == self.user.id)
        )
        count_query = (
            select(func.count(Decision.id))
            .select_from(Decision)
            .join(Scenario, Decision.scenario_id == Scenario.id)
            .where(Decision.user_id == self.user.id)
        )
        if category:
            query = query.where(Scenario.category == category.lower())
            count_query = count_query.where(Scenario.category == category.lower())
        if search:
            pattern = f"%{search.lower()}%"
            condition = func.lower(Scenario.text).like(pattern)
            query = query.where(condition)
            count_query = count_query.where(condition)

        total = self.db.execute(count_query).scalar_one()
        rows = self.db.execute(
            query.order_by(Decision.occurred_at.desc()).limit(limit).offset(offset)
        ).all()
        return [serialise(d, s) for d, s in rows], int(total)

    def timeline(self) -> list[dict]:
        rows = self.db.execute(
            select(Decision, Scenario)
            .join(Scenario, Decision.scenario_id == Scenario.id)
            .where(Decision.user_id == self.user.id)
            .order_by(Decision.occurred_at.asc())
        ).all()
        return [
            {
                "id": decision.id,
                "occurred_at": decision.occurred_at.isoformat(),
                "scenario": scenario.text[:160],
                "category": scenario.category,
                "chosen_option": decision.chosen_option,
                "importance": decision.importance,
                "approach": float((decision.chosen.stance or {}).get("approach", 0.5))
                if decision.chosen else 0.5,
                "satisfaction": decision.satisfaction,
                "dominant_factor": _dominant_factor(scenario.factors or {}),
            }
            for decision, scenario in rows
        ]


def _match(value: str, options: Sequence[str]) -> int | None:
    for index, option in enumerate(options):
        if option == value:
            return index
    lowered = value.strip().lower()
    for index, option in enumerate(options):
        if option.strip().lower() == lowered:
            return index
    return None


def _dominant_factor(factors: dict[str, float]) -> str | None:
    if not factors:
        return None
    key, value = max(factors.items(), key=lambda kv: kv[1])
    return key if value >= 0.2 else None


def serialise(decision: Decision, scenario: Scenario) -> dict:
    chosen = decision.chosen
    return {
        "id": decision.id,
        "scenario_id": scenario.id,
        "situation": scenario.text,
        "context": scenario.context,
        "category": scenario.category,
        "options": [
            {
                "label": option.label,
                "was_chosen": option.was_chosen,
                "stance": option.stance,
            }
            for option in sorted(decision.options, key=lambda o: o.position)
        ],
        "decision": decision.chosen_option,
        "reason": decision.reason,
        "emotional_state": decision.emotional_state,
        "importance": decision.importance,
        "expected_reward": decision.expected_reward,
        "perceived_risk": decision.perceived_risk,
        "outcome": decision.outcome,
        "satisfaction": decision.satisfaction,
        "occurred_at": decision.occurred_at.isoformat() if decision.occurred_at else None,
        "source": decision.source,
        "factors": {k: round(v, 4) for k, v in (scenario.factors or {}).items()},
        "factor_source": scenario.factor_source,
        "coverage": round(scenario.coverage, 4),
        "stance": (chosen.stance if chosen else {}),
        "created_at": decision.created_at.isoformat() if decision.created_at else None,
    }
