"""The profile-driven utility model.

This is the interpretable core of the twin. Given

* a factor reading of the situation (magnitudes in [0, 1]), and
* the person's learned traits (also in [0, 1]),

it produces a utility for every option that decomposes exactly into one
signed contribution per factor. Nothing here is a black box: the number the
UI shows next to "Family proximity: - - -" is literally the term computed in
:meth:`UtilityModel.score_option`.

Two structural ideas are worth calling out because they are what make the
model behave like a person rather than a linear scorer:

**Reward-modulated risk tolerance.** People are not uniformly risk-averse;
they become less deterred by risk when the upside is large. A big reward
therefore *damps* aversion towards zero - scaled by the person's own reward
sensitivity - rather than adding appetite on top, so a merely risk-neutral
person never comes out risk-seeking just because the money is good.

**Loss asymmetry.** Negative contributions are amplified by loss aversion, so
the same objective trade-off is evaluated differently by two people who agree
about the facts.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.services.decision_engine.parser import OptionStance
from app.services.decision_engine.taxonomy import (
    FACTOR_BY_KEY,
    FACTOR_KEYS,
    TRAIT_KEYS,
    neutral_traits,
)

# Structural constants. These are properties of the *model form*, not of the
# person; per-person adaptation happens through traits and ModelParams.
REWARD_RISK_COUPLING = 0.60     # how much a big upside relaxes risk aversion
REWARD_RISK_THRESHOLD = 0.55    # reward magnitude at which that kicks in
DEFER_INFO_GAIN = 0.55          # value of waiting when the situation is unclear
DEFER_TIME_PENALTY = 0.90       # cost of waiting when the clock is running
COMPROMISE_CONFLICT_GAIN = 0.50  # value of a middle path in a torn decision
DEFER_SHIELD = 0.60             # how much deferring mutes the live trade-off
COMPROMISE_SHIELD = 0.35


@dataclass
class ModelParams:
    """Per-person parameters fitted by the calibration layer, not by hand."""

    #: Softmax temperature. Low = highly consistent, high = noisy/variable.
    temperature: float = 0.55
    #: Learned preference for the passive branch, from the person's own base
    #: rate of taking action. Positive = tends to stay put.
    status_quo_bias: float = 0.0
    #: Global scale on factor contributions; fitted so probabilities are
    #: calibrated rather than over-confident.
    utility_scale: float = 1.0
    #: Post-fusion sharpening/softening, fitted from recorded outcomes. >1
    #: flattens an over-confident twin, <1 sharpens an under-confident one.
    calibration_temperature: float = 1.0

    def to_dict(self) -> dict:
        return {
            "temperature": round(self.temperature, 4),
            "status_quo_bias": round(self.status_quo_bias, 4),
            "utility_scale": round(self.utility_scale, 4),
            "calibration_temperature": round(self.calibration_temperature, 4),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> ModelParams:
        data = data or {}
        return cls(
            temperature=float(data.get("temperature", 0.55)),
            status_quo_bias=float(data.get("status_quo_bias", 0.0)),
            utility_scale=float(data.get("utility_scale", 1.0)),
            calibration_temperature=float(data.get("calibration_temperature", 1.0)),
        )


@dataclass
class Contribution:
    """One factor's signed push on one option."""

    factor: str
    label: str
    magnitude: float      # the situation: how much of this factor is present
    exposure: float       # signed: does this option take it on or forgo it
    valence: float        # the person: how attractive that exposure is
    value: float          # exposure * valence, after asymmetry scaling

    def to_dict(self) -> dict:
        return {
            "factor": self.factor,
            "label": self.label,
            "magnitude": round(self.magnitude, 4),
            "exposure": round(self.exposure, 4),
            "valence": round(self.valence, 4),
            "value": round(self.value, 4),
        }


@dataclass
class OptionUtility:
    """A scored option with its full decomposition."""

    label: str
    utility: float
    contributions: list[Contribution] = field(default_factory=list)
    stance: str = "approach"
    approach: float = 0.5
    extras: dict[str, float] = field(default_factory=dict)

    def top_contributions(self, n: int = 5) -> list[Contribution]:
        return sorted(self.contributions, key=lambda c: -abs(c.value))[:n]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "utility": round(self.utility, 4),
            "stance": self.stance,
            "approach": round(self.approach, 3),
            "contributions": [c.to_dict() for c in self.contributions],
            "extras": {k: round(v, 4) for k, v in self.extras.items()},
        }


class UtilityModel:
    """Scores options for one person against one situation."""

    def __init__(
        self,
        traits: dict[str, float] | None = None,
        params: ModelParams | None = None,
    ) -> None:
        base = neutral_traits()
        if traits:
            for key, value in traits.items():
                if key in base and value is not None:
                    base[key] = float(min(1.0, max(0.0, value)))
        self.traits = base
        self.params = params or ModelParams()

    # -- trait-derived valences ------------------------------------------
    def reward_damping(self, factors: dict[str, float]) -> float:
        """How much a large upside blunts this person's aversion to risk, 0-1."""
        reward = factors.get("financial_reward", 0.0)
        excess = max(0.0, reward - REWARD_RISK_THRESHOLD) / (1.0 - REWARD_RISK_THRESHOLD)
        return float(REWARD_RISK_COUPLING * self.traits["reward_sensitivity"] * excess)

    def risk_valence(self, factors: dict[str, float]) -> float:
        """Signed attractiveness of risk, after the upside has had its say.

        A large reward *dampens aversion*; it does not manufacture appetite.
        Scaling the aversion towards zero rather than adding a lift matters:
        adding one would turn a person with merely neutral risk tolerance into
        an actively risk-seeking one whenever the money was good, which is both
        wrong and the kind of error that is hard to notice in a single
        prediction and obvious in a counterfactual sweep.
        """
        base = 2.0 * (self.traits["risk_tolerance"] - 0.5)
        damping = self.reward_damping(factors)
        if base < 0:
            return base * (1.0 - damping)
        return base * (1.0 + 0.5 * damping)

    def effective_risk_tolerance(self, factors: dict[str, float]) -> float:
        """The risk valence expressed back on the 0-1 trait scale, for display."""
        return float(min(1.0, max(0.0, 0.5 + self.risk_valence(factors) / 2.0)))

    def factor_valences(self, factors: dict[str, float]) -> dict[str, float]:
        """How attractive exposure to each factor is, for this person, here.

        Returns values in roughly [-1, 1]. Positive means "more of this pulls
        me towards the option that provides it".
        """
        t = self.traits
        time_discount = 1.0 - 0.5 * factors.get("time_pressure", 0.0)

        return {
            "financial_reward": 0.5 * t["financial_priority"] + 0.5 * t["reward_sensitivity"],
            "risk": self.risk_valence(factors),
            "uncertainty": 2.0 * (t["uncertainty_tolerance"] - 0.5),
            "career_growth": t["career_priority"],
            "autonomy": 0.35 + 0.5 * (1.0 - t["social_influence"]),
            "effort_cost": -(1.0 - 0.5 * t["planning_tendency"]),
            "long_term_benefit": t["long_term_orientation"] * time_discount,
            # Protected goods (sign_under_approach = -1): valence is how much
            # the person values *keeping* them.
            "stability": 0.4 + 0.6 * (1.0 - t["risk_tolerance"]),
            "family_impact": t["family_priority"],
            "social_consequence": t["social_influence"],
            "moral_weight": t["moral_sensitivity"],
            # Modulator only.
            "time_pressure": 0.0,
        }

    def loss_multiplier(self) -> float:
        """Prospect-theory style asymmetry, in [0.5, 1.5]."""
        return 0.5 + self.traits["loss_aversion"]

    # -- scoring -----------------------------------------------------------
    def score_option(
        self,
        factors: dict[str, float],
        stance: OptionStance,
    ) -> OptionUtility:
        valences = self.factor_valences(factors)
        loss_mult = self.loss_multiplier()

        # How live the trade-off is under this option: deferring or
        # compromising mutes the consequences rather than facing them.
        shield = (1.0 - DEFER_SHIELD * stance.defer) * (1.0 - COMPROMISE_SHIELD * stance.compromise)
        direction = 2.0 * stance.approach - 1.0

        contributions: list[Contribution] = []
        positive_mass = 0.0
        negative_mass = 0.0

        for key in FACTOR_KEYS:
            factor = FACTOR_BY_KEY[key]
            if factor.sign_under_approach == 0:
                continue
            magnitude = float(factors.get(key, 0.0))
            if magnitude <= 0.0:
                continue
            exposure = direction * factor.sign_under_approach * magnitude
            valence = valences[key]
            value = exposure * valence * shield
            if value < 0:
                value *= loss_mult
            value *= self.params.utility_scale
            contributions.append(
                Contribution(key, factor.label, magnitude, exposure, valence, value)
            )
            if value > 0:
                positive_mass += value
            else:
                negative_mass -= value

        utility = sum(c.value for c in contributions)

        extras: dict[str, float] = {}

        # Deferral: worth something when the picture is genuinely unclear and
        # the person is a planner; costly when the clock is running.
        if stance.defer > 0:
            info_gain = (
                DEFER_INFO_GAIN
                * stance.defer
                * factors.get("uncertainty", 0.0)
                * self.traits["planning_tendency"]
            )
            time_cost = DEFER_TIME_PENALTY * stance.defer * factors.get("time_pressure", 0.0)
            extras["deferral_value"] = info_gain - time_cost
            utility += info_gain - time_cost

        # Compromise: worth something exactly when the decision is torn.
        if stance.compromise > 0:
            conflict = min(positive_mass, negative_mass)
            bonus = COMPROMISE_CONFLICT_GAIN * stance.compromise * conflict
            extras["compromise_value"] = bonus
            utility += bonus

        # Learned status-quo bias, applied to the passive branch.
        if self.params.status_quo_bias:
            bias = self.params.status_quo_bias * (1.0 - stance.approach)
            extras["status_quo_bias"] = bias
            utility += bias

        return OptionUtility(
            label=stance.label,
            utility=float(utility),
            contributions=contributions,
            stance=stance.stance,
            approach=stance.approach,
            extras=extras,
        )

    def score_options(
        self,
        factors: dict[str, float],
        stances: Sequence[OptionStance],
    ) -> list[OptionUtility]:
        return [self.score_option(factors, s) for s in stances]

    def probabilities(
        self,
        factors: dict[str, float],
        stances: Sequence[OptionStance],
    ) -> tuple[list[float], list[OptionUtility]]:
        """Softmax over option utilities."""
        scored = self.score_options(factors, stances)
        return softmax([s.utility for s in scored], self.params.temperature), scored


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax with a temperature."""
    if not values:
        return []
    temp = max(1e-3, float(temperature))
    scaled = [v / temp for v in values]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    if total <= 0:
        uniform = 1.0 / len(values)
        return [uniform] * len(values)
    return [e / total for e in exps]


def trait_vector(traits: dict[str, float]) -> list[float]:
    """Traits as a fixed-order vector, for the ML feature builder."""
    return [float(traits.get(k, 0.5)) for k in TRAIT_KEYS]
