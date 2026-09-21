"""Turning retrieved neighbours into a probability distribution.

This is the case-based reasoning layer: "in the four most similar situations
you have recorded, you chose the cautious option three times". It knows nothing
about traits and nothing about the utility model, which is exactly why it is
worth pooling with them - it can be right when the structured model's factor
reading is wrong.

Two neighbours can only vote for a new option if there is a sensible way to map
their old choice onto it. Mapping is done on **stance** (did they take the
active branch or not) rather than on option text, so "quit and join the
startup" can inform "accept the offer in Pune" even though no words overlap.
Where option text *does* match closely, that adds a direct bonus.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.services.decision_engine.parser import OptionStance
from app.services.decision_engine.utility import softmax
from app.services.embeddings.embedder import _features  # noqa: F401  (shared tokenisation)
from app.services.embeddings.store import Neighbour

#: How sharply neighbour votes are turned into probabilities. Higher = more
#: willing to let a strong majority dominate.
RETRIEVAL_SHARPNESS = 2.4
#: Similarity below this is treated as no evidence at all.
MIN_USEFUL_SIMILARITY = 0.12


def _jaccard(a: str, b: str) -> float:
    tokens_a = {t for t in _features(a) if not t.startswith("#")}
    tokens_b = {t for t in _features(b) if not t.startswith("#")}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


@dataclass
class RetrievalVote:
    """One neighbour's contribution to one option."""

    decision_id: str
    option: str
    weight: float
    stance_match: float
    text_match: float
    similarity: float

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "option": self.option,
            "weight": round(self.weight, 4),
            "stance_match": round(self.stance_match, 4),
            "text_match": round(self.text_match, 4),
            "similarity": round(self.similarity, 4),
        }


@dataclass
class RetrievalResult:
    probabilities: list[float]
    votes: list[RetrievalVote]
    support: float          # 0-1: how much usable evidence there was
    n_neighbours: int

    def to_dict(self) -> dict:
        return {
            "probabilities": [round(p, 4) for p in self.probabilities],
            "support": round(self.support, 4),
            "n_neighbours": self.n_neighbours,
            "votes": [v.to_dict() for v in self.votes],
        }


def predict_from_neighbours(
    neighbours: Sequence[Neighbour],
    stances: Sequence[OptionStance],
) -> RetrievalResult:
    """Score options by how the person behaved in comparable situations."""
    n_options = len(stances)
    if n_options == 0:
        return RetrievalResult([], [], 0.0, 0)

    uniform = [1.0 / n_options] * n_options
    usable = [n for n in neighbours if n.similarity >= MIN_USEFUL_SIMILARITY]
    if not usable:
        return RetrievalResult(uniform, [], 0.0, len(neighbours))

    scores = [0.0] * n_options
    votes: list[RetrievalVote] = []

    for neighbour in usable:
        # Importance-weighted: a decision the person called a 9 is a stronger
        # precedent than one they called a 3.
        importance = 0.6 + (neighbour.importance or 5) / 10.0
        base = neighbour.similarity * importance

        for index, stance in enumerate(stances):
            stance_match = 1.0 - abs(stance.approach - neighbour.approach)
            text_match = _jaccard(stance.label, neighbour.chosen_option)
            affinity = 0.7 * stance_match + 0.3 * text_match
            contribution = base * affinity
            scores[index] += contribution
            if affinity > 0.5:
                votes.append(
                    RetrievalVote(
                        decision_id=neighbour.decision_id,
                        option=stance.label,
                        weight=contribution,
                        stance_match=stance_match,
                        text_match=text_match,
                        similarity=neighbour.similarity,
                    )
                )

    total = sum(scores)
    if total <= 0:
        return RetrievalResult(uniform, votes, 0.0, len(neighbours))

    shares = [s / total for s in scores]
    # Sharpen away from uniform in proportion to how much evidence there is.
    probabilities = softmax([s * RETRIEVAL_SHARPNESS for s in shares], 1.0)

    top_similarity = max(n.similarity for n in usable)
    support = min(1.0, top_similarity * min(1.0, len(usable) / 3.0) * 1.4)

    return RetrievalResult(
        probabilities=probabilities,
        votes=sorted(votes, key=lambda v: -v.weight)[:12],
        support=float(support),
        n_neighbours=len(usable),
    )


def baseline_distribution(
    stances: Sequence[OptionStance], approach_base_rate: float
) -> list[float]:
    """The "most common decision" baseline the evaluation harness compares to.

    Rather than a fixed label (meaningless when option sets differ), it uses the
    person's historical rate of taking the active branch and assigns each option
    the probability implied by its own stance.
    """
    if not stances:
        return []
    raw = [
        approach_base_rate * s.approach + (1.0 - approach_base_rate) * (1.0 - s.approach)
        for s in stances
    ]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(stances)] * len(stances)
    return [r / total for r in raw]


def probabilities_by_label(
    labels: Sequence[str], probabilities: Sequence[float]
) -> dict[str, float]:
    return {label: float(p) for label, p in zip(labels, probabilities, strict=False)}
