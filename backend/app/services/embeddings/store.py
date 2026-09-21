"""Vector storage and similar-decision retrieval.

Retrieval is the second layer of the hybrid architecture and the source of the
"Similar historical decisions" panel. Similarity is deliberately *not* pure
text cosine:

    similarity = w_text * cosine(text) + w_factor * cosine(factors)

Text similarity finds situations phrased alike; factor similarity finds
situations that pose the same *trade-off* even when the words differ ("quit a
stable job for a startup" vs "leave a fixed deposit for equity"). The second is
what actually transfers behaviourally, so it carries substantial weight.

On Postgres the text half is served by pgvector's ``<=>`` operator with an
IVFFlat index; on SQLite the same ranking is computed with NumPy. Both paths
return identical results, so the test suite exercises the real logic.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.config import settings
from app.database.types import PGVECTOR_AVAILABLE
from app.logging_conf import get_logger
from app.models import Decision, EmbeddingRecord, Scenario
from app.services.decision_engine.taxonomy import FACTOR_KEYS
from app.services.embeddings.embedder import cosine, get_embedder

log = get_logger(__name__)

TEXT_WEIGHT = 0.55
FACTOR_WEIGHT = 0.45


def factor_vector(factors: dict[str, float]) -> list[float]:
    return [float(factors.get(k, 0.0)) for k in FACTOR_KEYS]


def factor_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine over factor space, with an all-zero guard.

    Two scenarios with no detected factors are *not* similar - they are both
    unparsed. Returning 0 keeps unparsed noise out of the evidence panel.
    """
    va, vb = factor_vector(a), factor_vector(b)
    if sum(va) <= 0 or sum(vb) <= 0:
        return 0.0
    return cosine(va, vb)


@dataclass
class Neighbour:
    """One retrieved historical decision."""

    decision_id: str
    scenario_id: str
    similarity: float
    text_similarity: float
    factor_similarity: float
    scenario_text: str
    chosen_option: str
    reason: str
    occurred_at: str
    importance: int
    factors: dict[str, float]
    approach: float
    outcome: str = ""
    satisfaction: float | None = None

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "scenario_id": self.scenario_id,
            "similarity": round(self.similarity, 4),
            "text_similarity": round(self.text_similarity, 4),
            "factor_similarity": round(self.factor_similarity, 4),
            "scenario_text": self.scenario_text,
            "chosen_option": self.chosen_option,
            "reason": self.reason,
            "occurred_at": self.occurred_at,
            "importance": self.importance,
            "approach": round(self.approach, 3),
            "outcome": self.outcome,
            "satisfaction": self.satisfaction,
            "factors": {k: round(v, 3) for k, v in self.factors.items()},
        }


class VectorStore:
    """Persists embeddings and answers nearest-neighbour queries."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.embedder = get_embedder()

    # -- writing ---------------------------------------------------------
    def upsert(
        self,
        *,
        user_id: str,
        owner_type: str,
        owner_id: str,
        text: str,
        kind: str = "situation",
    ) -> EmbeddingRecord:
        vector = self.embedder.embed(text)
        existing = self.db.execute(
            select(EmbeddingRecord).where(
                EmbeddingRecord.owner_type == owner_type,
                EmbeddingRecord.owner_id == owner_id,
                EmbeddingRecord.kind == kind,
            )
        ).scalar_one_or_none()

        if existing:
            existing.vector = vector
            existing.dim = len(vector)
            existing.backend = self.embedder.name
            existing.source_text = text[:4000]
            return existing

        record = EmbeddingRecord(
            user_id=user_id,
            owner_type=owner_type,
            owner_id=owner_id,
            kind=kind,
            vector=vector,
            dim=len(vector),
            backend=self.embedder.name,
            source_text=text[:4000],
        )
        self.db.add(record)
        return record

    def index_scenario(self, scenario: Scenario) -> None:
        self.upsert(
            user_id=scenario.user_id,
            owner_type="scenario",
            owner_id=scenario.id,
            text=scenario.text,
            kind="situation",
        )

    def index_decision(self, decision: Decision, scenario: Scenario) -> None:
        """Index a decision twice: the situation, and the person's reasoning.

        They retrieve different things - "what happened" and "why I chose that"
        - and the reasoning index is what surfaces a stated principle applied
        in an otherwise unrelated context.
        """
        self.index_scenario(scenario)
        reasoning = (decision.reason or "").strip()
        if reasoning:
            self.upsert(
                user_id=decision.user_id,
                owner_type="decision",
                owner_id=decision.id,
                text=f"{decision.chosen_option}. {reasoning}",
                kind="reasoning",
            )

    # -- reading ---------------------------------------------------------
    def _text_similarities(
        self, user_id: str, query_vector: Sequence[float], scenario_ids: Iterable[str]
    ) -> dict[str, float]:
        ids = list(scenario_ids)
        if not ids:
            return {}

        if settings.is_postgres and PGVECTOR_AVAILABLE:
            literal = "[" + ",".join(f"{v:.7f}" for v in query_vector) + "]"
            rows = self.db.execute(
                sql_text(
                    """
                    SELECT owner_id, 1 - (vector <=> CAST(:q AS vector)) AS similarity
                    FROM embeddings
                    WHERE user_id = :uid
                      AND owner_type = 'scenario'
                      AND kind = 'situation'
                      AND owner_id = ANY(:ids)
                    """
                ),
                {"q": literal, "uid": user_id, "ids": ids},
            ).all()
            return {str(r[0]): float(r[1]) for r in rows}

        records = self.db.execute(
            select(EmbeddingRecord).where(
                EmbeddingRecord.user_id == user_id,
                EmbeddingRecord.owner_type == "scenario",
                EmbeddingRecord.kind == "situation",
                EmbeddingRecord.owner_id.in_(ids),
            )
        ).scalars().all()
        if not records:
            return {}

        matrix = np.array([r.vector for r in records], dtype=float)
        query = np.array(list(query_vector), dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        qnorm = float(np.linalg.norm(query))
        if qnorm <= 0:
            return {r.owner_id: 0.0 for r in records}
        safe = np.where(norms > 0, norms, 1.0)
        sims = (matrix @ query) / (safe * qnorm)
        sims = np.where(norms > 0, sims, 0.0)
        return {r.owner_id: float(s) for r, s in zip(records, sims, strict=False)}

    def similar_decisions(
        self,
        *,
        user_id: str,
        query_text: str,
        query_factors: dict[str, float],
        k: int = 5,
        exclude_decision_ids: Sequence[str] = (),
        min_similarity: float = 0.05,
    ) -> list[Neighbour]:
        """Rank the person's own history against a new situation."""
        rows = self.db.execute(
            select(Decision, Scenario)
            .join(Scenario, Decision.scenario_id == Scenario.id)
            .where(Decision.user_id == user_id)
        ).all()
        rows = [(d, s) for d, s in rows if d.id not in set(exclude_decision_ids)]
        if not rows:
            return []

        query_vector = self.embedder.embed(query_text)
        text_sims = self._text_similarities(user_id, query_vector, {s.id for _, s in rows})

        neighbours: list[Neighbour] = []
        for decision, scenario in rows:
            t_sim = max(0.0, text_sims.get(scenario.id, 0.0))
            f_sim = max(0.0, factor_similarity(query_factors, scenario.factors or {}))
            blended = TEXT_WEIGHT * t_sim + FACTOR_WEIGHT * f_sim
            if blended < min_similarity:
                continue
            chosen = decision.chosen
            neighbours.append(
                Neighbour(
                    decision_id=decision.id,
                    scenario_id=scenario.id,
                    similarity=blended,
                    text_similarity=t_sim,
                    factor_similarity=f_sim,
                    scenario_text=scenario.text,
                    chosen_option=decision.chosen_option,
                    reason=decision.reason or "",
                    occurred_at=decision.occurred_at.isoformat() if decision.occurred_at else "",
                    importance=decision.importance,
                    factors=scenario.factors or {},
                    approach=float((chosen.stance or {}).get("approach", 0.5)) if chosen else 0.5,
                    outcome=decision.outcome or "",
                    satisfaction=decision.satisfaction,
                )
            )

        neighbours.sort(key=lambda n: -n.similarity)
        return neighbours[:k]

    def rebuild_index(self, user_id: str) -> int:
        """Re-embed everything for one user, e.g. after switching backends."""
        count = 0
        rows = self.db.execute(
            select(Decision, Scenario)
            .join(Scenario, Decision.scenario_id == Scenario.id)
            .where(Decision.user_id == user_id)
        ).all()
        for decision, scenario in rows:
            self.index_decision(decision, scenario)
            count += 1
        self.db.flush()
        log.info("rebuilt %d embeddings for user %s", count, user_id)
        return count
