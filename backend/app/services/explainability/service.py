"""The explanation layer.

Everything this module emits must be traceable to something the system actually
holds: a factor magnitude with the words that produced it, a trait with its
evidence count, a past decision with its date. The one rule enforced throughout
is the distinction the brief insists on:

* **Observed** - "you declined 7 of 9 recorded opportunities that required
  relocation". A count over records.
* **Inferred** - "family proximity is estimated to weigh heavily (0.84,
  confidence 0.71)". A model estimate, always shown with its uncertainty.

The two are returned in separate keys and worded differently, so a client
cannot render an inference as a fact by accident.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from app.services.decision_engine.parser import FactorHit
from app.services.decision_engine.taxonomy import FACTOR_BY_KEY, TRAIT_BY_KEY
from app.services.decision_engine.utility import Contribution
from app.services.embeddings.store import Neighbour

#: Rendering thresholds for the +++ / --- strength bars in the UI.
_STRENGTH_BANDS = [(0.45, 3), (0.22, 2), (0.07, 1)]


def strength_symbol(value: float) -> str:
    """Render a signed contribution as +++ / ++ / + / - / -- / ---."""
    magnitude = abs(value)
    level = 0
    for threshold, count in _STRENGTH_BANDS:
        if magnitude >= threshold:
            level = count
            break
    if level == 0:
        return "~"
    return ("+" if value > 0 else "-") * level


@dataclass
class FactorExplanation:
    factor: str
    label: str
    description: str
    magnitude: float
    contribution: float
    direction: str
    symbol: str
    valence: float
    evidence_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor": self.factor,
            "label": self.label,
            "description": self.description,
            "magnitude": round(self.magnitude, 4),
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
            "symbol": self.symbol,
            "valence": round(self.valence, 4),
            "evidence_phrases": self.evidence_phrases[:4],
        }


def factor_explanations(
    contributions: Sequence[Contribution],
    hits: Sequence[FactorHit],
    limit: int = 8,
) -> list[FactorExplanation]:
    """Rank the factors driving the predicted option, with their textual evidence."""
    by_factor: dict[str, list[str]] = {}
    for hit in hits:
        if hit.negated:
            continue
        by_factor.setdefault(hit.factor, []).append(hit.context)

    explanations = [
        FactorExplanation(
            factor=c.factor,
            label=c.label,
            description=FACTOR_BY_KEY[c.factor].description if c.factor in FACTOR_BY_KEY else "",
            magnitude=c.magnitude,
            contribution=c.value,
            direction="supports" if c.value > 0 else "opposes",
            symbol=strength_symbol(c.value),
            valence=c.valence,
            evidence_phrases=by_factor.get(c.factor, []),
        )
        for c in contributions
        if abs(c.value) > 1e-4
    ]
    explanations.sort(key=lambda e: -abs(e.contribution))
    return explanations[:limit]


def ml_attribution(
    trained_model,
    feature_rows: Sequence[Sequence[float]],
    predicted_index: int,
    limit: int = 8,
) -> list[dict]:
    """Local feature attribution for the statistical layer.

    For the logistic model this is exact: in a conditional logit only
    *differences between options* can change the choice, so the local
    contribution of feature *i* to the predicted option is
    ``coef_i * (x_i - mean_i across the options)``. That is a genuine
    decomposition of the score gap rather than a global importance ranking.

    For gradient boosting there is no such closed form, so the tree importances
    are reported instead, and labelled as global rather than local.
    """
    if trained_model is None or not feature_rows:
        return []

    matrix = np.array(feature_rows, dtype=float)
    names = trained_model.feature_names
    row = matrix[predicted_index]
    centred = row - matrix.mean(axis=0)

    if trained_model.kind == "logistic":
        coefficients = np.array(
            [trained_model.importances.get(name, 0.0) for name in names], dtype=float
        )
        contributions = coefficients * centred
        order = np.argsort(-np.abs(contributions))[:limit]
        return [
            {
                "feature": names[i],
                "contribution": round(float(contributions[i]), 5),
                "value": round(float(row[i]), 4),
                "coefficient": round(float(coefficients[i]), 5),
                "scope": "local",
            }
            for i in order
            if abs(contributions[i]) > 1e-6
        ]

    importances = np.array(
        [trained_model.importances.get(name, 0.0) for name in names], dtype=float
    )
    order = np.argsort(-importances)[:limit]
    return [
        {
            "feature": names[i],
            "contribution": round(float(importances[i]), 5),
            "value": round(float(row[i]), 4),
            "coefficient": None,
            "scope": "global",
        }
        for i in order
        if importances[i] > 1e-6
    ]


def observed_statements(
    neighbours: Sequence[Neighbour],
    patterns: Sequence[dict],
    decision_count: int,
) -> list[str]:
    """Statements that are literally counts over the record. No inference."""
    statements: list[str] = [
        f"{decision_count} decisions are on record for this person."
    ]
    for pattern in patterns[:4]:
        statements.append(pattern.get("content", ""))

    if neighbours:
        aligned = [n for n in neighbours if n.similarity >= 0.25]
        if aligned:
            took = sum(1 for n in aligned if n.approach >= 0.5)
            statements.append(
                f"Among the {len(aligned)} most similar recorded situations, "
                f"the active option was taken {took} time(s) and declined "
                f"{len(aligned) - took} time(s)."
            )
    return [s for s in statements if s]


def inferred_statements(
    traits: dict[str, float],
    confidences: dict[str, float],
    contributions: Sequence[Contribution],
    limit: int = 5,
) -> list[dict]:
    """Trait-level claims, each carrying its own confidence and hedge."""
    trait_for_factor = {
        "financial_reward": "financial_priority",
        "risk": "risk_tolerance",
        "uncertainty": "uncertainty_tolerance",
        "stability": "loss_aversion",
        "family_impact": "family_priority",
        "career_growth": "career_priority",
        "social_consequence": "social_influence",
        "long_term_benefit": "long_term_orientation",
        "moral_weight": "moral_sensitivity",
        "effort_cost": "planning_tendency",
        "autonomy": "social_influence",
    }

    seen: set[str] = set()
    statements: list[dict] = []
    for contribution in sorted(contributions, key=lambda c: -abs(c.value)):
        trait_key = trait_for_factor.get(contribution.factor)
        if not trait_key or trait_key in seen:
            continue
        trait = TRAIT_BY_KEY.get(trait_key)
        if trait is None:
            continue
        seen.add(trait_key)

        value = traits.get(trait_key, 0.5)
        confidence = confidences.get(trait_key, 0.0)
        leaning = trait.high_label if value >= 0.5 else trait.low_label
        hedge = (
            "the evidence supports" if confidence >= 0.6
            else "the evidence weakly suggests" if confidence >= 0.3
            else "there is not yet enough evidence to say much, but the current estimate is"
        )
        statements.append(
            {
                "trait": trait_key,
                "label": trait.label,
                "value": round(float(value), 4),
                "confidence": round(float(confidence), 4),
                "statement": (
                    f"On {trait.label.lower()}, {hedge} a tendency towards "
                    f"{leaning} (estimate {value:.2f}, confidence {confidence:.2f})."
                ),
            }
        )
        if len(statements) >= limit:
            break
    return statements


def build_reasoning_trace(
    *,
    predicted_label: str,
    probability: float,
    factors: list[FactorExplanation],
    neighbours: Sequence[Neighbour],
    layer_summary: Sequence[dict],
    confidence_label: str,
) -> list[dict]:
    """A numbered, auditable account of how the number was produced."""
    trace: list[dict] = []

    detected = [f for f in factors if f.magnitude >= 0.15]
    trace.append(
        {
            "step": 1,
            "title": "Read the situation",
            "detail": (
                "Detected " + ", ".join(
                    f"{f.label.lower()} ({f.magnitude:.2f})" for f in detected[:5]
                ) + "."
            ) if detected else (
                "No decision factors were confidently detected in the text, so the "
                "prediction rests almost entirely on the behavioural profile."
            ),
        }
    )

    supporting = [f for f in factors if f.contribution > 0][:3]
    opposing = [f for f in factors if f.contribution < 0][:3]
    trace.append(
        {
            "step": 2,
            "title": "Weigh the trade-off against the profile",
            "detail": (
                "Pushing towards this option: "
                + (", ".join(f"{f.label.lower()} {f.symbol}" for f in supporting) or "nothing notable")
                + ". Pushing against: "
                + (", ".join(f"{f.label.lower()} {f.symbol}" for f in opposing) or "nothing notable")
                + "."
            ),
        }
    )

    if neighbours:
        trace.append(
            {
                "step": 3,
                "title": "Compare with what actually happened before",
                "detail": "; ".join(
                    f"{n.similarity:.0%} similar: chose \"{n.chosen_option}\""
                    for n in neighbours[:3]
                ) + ".",
            }
        )
    else:
        trace.append(
            {
                "step": 3,
                "title": "Compare with what actually happened before",
                "detail": "No comparable situation is on record, so case-based "
                          "evidence contributed nothing to this estimate.",
            }
        )

    trace.append(
        {
            "step": 4,
            "title": "Pool the layers",
            "detail": "; ".join(
                f"{layer['name']} {layer['weight']:.0%} weight"
                for layer in layer_summary if layer.get("available")
            ) + ".",
        }
    )
    trace.append(
        {
            "step": 5,
            "title": "Result",
            "detail": (
                f"\"{predicted_label}\" at {probability:.0%}, with {confidence_label} "
                f"confidence in that estimate."
            ),
        }
    )
    return trace


def natural_language_summary(
    *,
    predicted_label: str,
    probability: float,
    confidence_label: str,
    factors: Sequence[FactorExplanation],
    neighbours: Sequence[Neighbour],
    decision_count: int,
) -> str:
    """A grounded, hedged summary that works with no LLM available."""
    supporting = [f for f in factors if f.contribution > 0][:2]
    opposing = [f for f in factors if f.contribution < 0][:2]

    parts = [
        f'Based on {decision_count} recorded decision(s), the model estimates '
        f'"{predicted_label}" as the most likely choice at {probability:.0%} '
        f'({confidence_label} confidence in the estimate).'
    ]
    if supporting:
        parts.append(
            "The strongest pull towards it comes from "
            + " and ".join(f.label.lower() for f in supporting) + "."
        )
    if opposing:
        parts.append(
            "Working against it: "
            + " and ".join(f.label.lower() for f in opposing) + "."
        )
    if neighbours:
        best = neighbours[0]
        parts.append(
            f'The closest comparable situation on record ({best.similarity:.0%} similar) '
            f'ended with "{best.chosen_option}"'
            + (f', reasoned as "{best.reason}".' if best.reason else ".")
        )
    if decision_count < 10:
        parts.append(
            "With this little history the estimate is provisional and should be "
            "read as a starting hypothesis rather than a prediction."
        )
    return " ".join(parts)


LIMITATIONS = [
    "This is a behavioural prediction model, not a simulation of a person's mind.",
    "Estimates are probabilities derived from recorded decisions, not statements of fact.",
    "Factors are extracted from text and may miss context that was never written down.",
    "Accuracy degrades on decision types that are absent from the recorded history.",
    "It must not be used to make medical, legal, employment, lending or other "
    "high-stakes determinations about a person.",
]
