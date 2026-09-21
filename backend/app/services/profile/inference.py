"""Trait inference: turning observed choices into a person model with error bars.

The rule this module exists to enforce is the one from the brief: **observed
and inferred are different things**. A decision record is an observation. A
trait is an estimate derived from many observations, and it is never allowed to
appear without the evidence that produced it and the uncertainty around it.

Mechanics
---------
Each trait is a Beta posterior. Every relevant historical decision contributes
a weighted pseudo-observation ``x`` in [0, 1]:

    "This situation put a large amount of *risk* on the table (magnitude 0.8),
     and the person took the active option (approach 1.0). That is evidence
     for risk tolerance: x = 1.0, weight = 0.8 * importance * recency."

Weights fold in three things:

* **factor magnitude** - a scenario with no risk in it says nothing about risk
  tolerance, so it contributes nothing rather than contributing noise;
* **importance** - a decision the person rated 9/10 counts for more than one
  they rated 2/10;
* **recency** - evidence decays with a multi-year half-life, so a profile can
  track genuine preference change instead of averaging over a decade.

The posterior mean is the reported value and the posterior standard deviation
becomes the confidence, which means a trait backed by two decisions is
*reported as* barely known rather than reported as a number.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.services.decision_engine.parser import OptionStance, tokenize
from app.services.decision_engine.taxonomy import TRAIT_KEYS

# Beta(1, 1): a flat prior. Its standard deviation is the yardstick against
# which confidence is measured, so a trait with no evidence has confidence 0.
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
PRIOR_STD = math.sqrt(
    (PRIOR_ALPHA * PRIOR_BETA)
    / (((PRIOR_ALPHA + PRIOR_BETA) ** 2) * (PRIOR_ALPHA + PRIOR_BETA + 1.0))
)

#: Evidence half-life in years. Long enough that a stable person keeps a stable
#: profile; short enough that a real change of priorities shows up.
EVIDENCE_HALF_LIFE_YEARS = 3.0

#: Below this factor magnitude the situation did not really pose the trade-off.
MIN_FACTOR_MAGNITUDE = 0.15

#: Relative reliability by evidence source. Behaviour outranks self-report.
SOURCE_RELIABILITY = {
    "decision": 1.0,
    "questionnaire": 0.55,
    "text": 0.35,
    "feedback": 1.6,
}


@dataclass(frozen=True)
class EvidenceRule:
    """Maps a factor present in a situation onto a trait observation."""

    trait: str
    factor: str
    #: +1: taking the active option is evidence *for* the trait.
    #: -1: taking the active option is evidence *against* it.
    direction: int
    scale: float = 1.0
    #: Optional second factor that must also be present for the rule to fire.
    gated_on: str | None = None
    template: str = ""


EVIDENCE_RULES: list[EvidenceRule] = [
    EvidenceRule("risk_tolerance", "risk", 1, 1.0,
                 template="risk was on the table and the person {verb} the active option"),
    EvidenceRule("uncertainty_tolerance", "uncertainty", 1, 1.0,
                 template="the outcome was unclear and the person {verb} the active option"),
    EvidenceRule("financial_priority", "financial_reward", 1, 1.0,
                 template="money was at stake and the person {verb} the option offering it"),
    EvidenceRule("family_priority", "family_impact", -1, 1.0,
                 template="family was affected and the person {verb} the disruptive option"),
    EvidenceRule("career_priority", "career_growth", 1, 1.0,
                 template="career growth was available and the person {verb} it"),
    EvidenceRule("social_influence", "social_consequence", -1, 1.0,
                 template="social consequences were in play and the person {verb} them"),
    EvidenceRule("long_term_orientation", "long_term_benefit", 1, 1.0,
                 template="the payoff was delayed and the person {verb} it"),
    EvidenceRule("moral_sensitivity", "moral_weight", -1, 1.0,
                 template="a value conflict was present and the person {verb} it"),
    EvidenceRule("loss_aversion", "stability", -1, 1.0,
                 template="settled ground was at stake and the person {verb} giving it up"),
    # Reward sensitivity is only informative when the reward had to be weighed
    # against something: gate it on risk being present.
    EvidenceRule("reward_sensitivity", "financial_reward", 1, 0.9, gated_on="risk",
                 template="a reward had to be weighed against risk and the person {verb} it"),
    EvidenceRule("planning_tendency", "effort_cost", 1, 0.5,
                 template="the option demanded sustained effort and the person {verb} it"),
]

#: Reason-text cues that speak directly to deliberation style.
_PLANNING_CUES = [
    "researched", "research", "compared", "spreadsheet", "calculated", "planned",
    "thought it through", "weighed", "analysed", "analyzed", "list of pros",
    "took my time", "slept on it", "consulted",
]
_IMPULSE_CUES = [
    "gut", "instinct", "spur of the moment", "impulse", "without thinking",
    "just felt", "on a whim", "immediately said", "did not think",
]


@dataclass
class Observation:
    """One weighted pseudo-observation for one trait."""

    trait: str
    value: float
    weight: float
    source: str
    decision_id: str | None = None
    note: str = ""
    detail: dict = field(default_factory=dict)


def recency_weight(occurred_at: datetime | None, now: datetime | None = None) -> float:
    """Exponential decay with :data:`EVIDENCE_HALF_LIFE_YEARS`."""
    if occurred_at is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    age_years = max(0.0, (now - occurred_at).total_seconds() / (365.25 * 24 * 3600))
    return float(0.5 ** (age_years / EVIDENCE_HALF_LIFE_YEARS))


def importance_weight(importance: int) -> float:
    """Map a 1-10 importance rating onto a [0.6, 1.6] multiplier."""
    return 0.6 + max(1, min(10, int(importance or 5))) / 10.0


def observations_from_decision(
    *,
    decision_id: str,
    factors: dict[str, float],
    chosen_stance: OptionStance,
    importance: int = 5,
    occurred_at: datetime | None = None,
    reason: str = "",
    now: datetime | None = None,
) -> list[Observation]:
    """Extract every trait observation supported by one historical decision."""
    observations: list[Observation] = []
    decay = recency_weight(occurred_at, now)
    imp = importance_weight(importance)
    approach = float(chosen_stance.approach)
    verb_taken = "took" if approach >= 0.5 else "declined"

    for rule in EVIDENCE_RULES:
        magnitude = float(factors.get(rule.factor, 0.0))
        if magnitude < MIN_FACTOR_MAGNITUDE:
            continue
        gate = 1.0
        if rule.gated_on:
            gate = float(factors.get(rule.gated_on, 0.0))
            if gate < MIN_FACTOR_MAGNITUDE:
                continue

        value = approach if rule.direction > 0 else 1.0 - approach
        weight = magnitude * gate * imp * decay * rule.scale
        # A stance we could not classify tells us little about direction.
        weight *= 0.35 + 0.65 * max(chosen_stance.confidence, 0.2)
        if weight <= 0.01:
            continue

        observations.append(
            Observation(
                trait=rule.trait,
                value=round(value, 4),
                weight=round(weight, 4),
                source="decision",
                decision_id=decision_id,
                note=rule.template.format(verb=verb_taken),
                detail={
                    "factor": rule.factor,
                    "factor_magnitude": round(magnitude, 3),
                    "approach": round(approach, 3),
                    "importance": importance,
                    "recency_weight": round(decay, 3),
                },
            )
        )

    # Deferring under uncertainty is direct evidence about deliberation style.
    if chosen_stance.defer > 0.2 or factors.get("uncertainty", 0.0) >= MIN_FACTOR_MAGNITUDE:
        weight = max(0.15, factors.get("uncertainty", 0.0)) * imp * decay * 0.7
        observations.append(
            Observation(
                trait="planning_tendency",
                value=round(min(1.0, 0.5 + 0.5 * chosen_stance.defer), 4),
                weight=round(weight, 4),
                source="decision",
                decision_id=decision_id,
                note="chose to wait and gather information" if chosen_stance.defer > 0.2
                     else "decided without deferring despite an unclear picture",
                detail={"defer": chosen_stance.defer},
            )
        )

    # The person's own account of how they decided.
    observations.extend(_observations_from_reason(decision_id, reason, imp * decay))
    return observations


def _observations_from_reason(
    decision_id: str, reason: str, base_weight: float
) -> list[Observation]:
    if not reason:
        return []
    tokens = set(tokenize(reason))
    joined = " ".join(tokenize(reason))
    planning_hits = sum(
        1 for cue in _PLANNING_CUES if (cue in joined if " " in cue else cue in tokens)
    )
    impulse_hits = sum(
        1 for cue in _IMPULSE_CUES if (cue in joined if " " in cue else cue in tokens)
    )
    if not planning_hits and not impulse_hits:
        return []

    total = planning_hits + impulse_hits
    value = planning_hits / total
    return [
        Observation(
            trait="planning_tendency",
            value=round(value, 4),
            weight=round(min(1.0, 0.35 * total) * base_weight, 4),
            source="text",
            decision_id=decision_id,
            note="the stated reasoning describes "
                 + ("deliberation" if value >= 0.5 else "acting on instinct"),
            detail={"planning_cues": planning_hits, "impulse_cues": impulse_hits},
        )
    ]


@dataclass
class TraitPosterior:
    """A trait estimate with everything needed to justify it."""

    trait: str
    alpha: float
    beta: float
    evidence_count: int
    supporting_decisions: list[str] = field(default_factory=list)
    sources: dict[str, int] = field(default_factory=dict)
    consistency: float = 0.0

    @property
    def value(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    @property
    def std(self) -> float:
        total = self.alpha + self.beta
        return float(math.sqrt((self.alpha * self.beta) / ((total ** 2) * (total + 1.0))))

    @property
    def confidence(self) -> float:
        """0 with no evidence, approaching 1 as the posterior tightens."""
        return float(max(0.0, min(1.0, 1.0 - self.std / PRIOR_STD)))

    def credible_interval(self, width: float = 0.9) -> tuple[float, float]:
        """Normal approximation to the Beta credible interval, clipped to [0, 1]."""
        z = 1.6449 if width >= 0.9 else 1.0
        lo = max(0.0, self.value - z * self.std)
        hi = min(1.0, self.value + z * self.std)
        return round(lo, 4), round(hi, 4)

    def to_dict(self) -> dict:
        lo, hi = self.credible_interval()
        return {
            "trait": self.trait,
            "value": round(self.value, 4),
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "credible_interval": [lo, hi],
            "consistency": round(self.consistency, 4),
            "supporting_decisions": self.supporting_decisions,
            "sources": self.sources,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
        }


def accumulate(observations: Sequence[Observation]) -> dict[str, TraitPosterior]:
    """Fold weighted observations into one Beta posterior per trait."""
    posteriors = {
        key: TraitPosterior(trait=key, alpha=PRIOR_ALPHA, beta=PRIOR_BETA, evidence_count=0)
        for key in TRAIT_KEYS
    }
    grouped: dict[str, list[Observation]] = {key: [] for key in TRAIT_KEYS}

    for obs in observations:
        if obs.trait not in posteriors:
            continue
        reliability = SOURCE_RELIABILITY.get(obs.source, 0.5)
        weight = max(0.0, obs.weight) * reliability
        if weight <= 0:
            continue
        posterior = posteriors[obs.trait]
        posterior.alpha += weight * obs.value
        posterior.beta += weight * (1.0 - obs.value)
        posterior.evidence_count += 1
        posterior.sources[obs.source] = posterior.sources.get(obs.source, 0) + 1
        grouped[obs.trait].append(obs)

    for key, posterior in posteriors.items():
        items = grouped[key]
        if items:
            total_weight = sum(o.weight for o in items) or 1.0
            mean = sum(o.value * o.weight for o in items) / total_weight
            dispersion = sum(abs(o.value - mean) * o.weight for o in items) / total_weight
            # 0.5 is the maximum possible mean absolute deviation on [0, 1].
            posterior.consistency = float(max(0.0, 1.0 - dispersion / 0.5))
            ranked = sorted(
                (o for o in items if o.decision_id),
                key=lambda o: -o.weight,
            )
            seen: list[str] = []
            for obs in ranked:
                if obs.decision_id not in seen:
                    seen.append(obs.decision_id)
                if len(seen) >= 6:
                    break
            posterior.supporting_decisions = seen

    return posteriors
