"""Feature construction for the statistical layer.

The learning problem is framed as a **conditional logit**: one row per
(scenario, option) pair, labelled 1 for the option actually chosen and 0 for
the rest. At prediction time the classifier scores every option and the scores
are normalised across the option set. This framing handles decisions with two
options and decisions with five without changing the model, and it means the
training signal is "why this option beat the others", not "accept vs reject".

The feature block deliberately *includes* the interpretable utility model's own
per-factor contributions. The statistical layer therefore starts from the
structured model and learns the residual - where this particular person
departs from what their traits alone predict. With very little data it
reproduces the utility model; with more data it can overrule it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.services.decision_engine.parser import OptionStance
from app.services.decision_engine.taxonomy import (
    FACTOR_BY_KEY,
    FACTOR_KEYS,
    TRAIT_KEYS,
)
from app.services.decision_engine.utility import UtilityModel

FEATURE_NAMES: list[str] = (
    [f"factor:{k}" for k in FACTOR_KEYS]
    + ["stance:approach", "stance:defer", "stance:compromise", "stance:n_options"]
    + [f"exposure:{k}" for k in FACTOR_KEYS]
    + [f"trait:{k}" for k in TRAIT_KEYS]
    + [f"contribution:{k}" for k in FACTOR_KEYS]
    + ["util:total", "util:positive_mass", "util:negative_mass", "util:conflict"]
)

FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class FeatureRow:
    values: list[float]
    option_label: str

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values, strict=False))


def build_features(
    factors: dict[str, float],
    stances: Sequence[OptionStance],
    model: UtilityModel,
) -> list[FeatureRow]:
    """One feature row per option, in the order the options were given."""
    n_options = max(1, len(stances))
    scored = model.score_options(factors, stances)
    rows: list[FeatureRow] = []

    for stance, utility in zip(stances, scored, strict=False):
        direction = 2.0 * stance.approach - 1.0
        values: list[float] = []

        # 1. the situation
        values.extend(float(factors.get(k, 0.0)) for k in FACTOR_KEYS)

        # 2. the option's stance
        values.extend([
            float(stance.approach),
            float(stance.defer),
            float(stance.compromise),
            float(n_options) / 5.0,
        ])

        # 3. signed exposure: the interaction that actually carries the signal
        for key in FACTOR_KEYS:
            factor = FACTOR_BY_KEY[key]
            values.append(direction * factor.sign_under_approach * float(factors.get(key, 0.0)))

        # 4. the person
        values.extend(float(model.traits.get(k, 0.5)) for k in TRAIT_KEYS)

        # 5. the structured model's own decomposition
        contribution_map = {c.factor: c.value for c in utility.contributions}
        values.extend(float(contribution_map.get(k, 0.0)) for k in FACTOR_KEYS)

        # 6. summary statistics of the trade-off
        positive_mass = sum(c.value for c in utility.contributions if c.value > 0)
        negative_mass = -sum(c.value for c in utility.contributions if c.value < 0)
        values.extend([
            float(utility.utility),
            float(positive_mass),
            float(negative_mass),
            float(min(positive_mass, negative_mass)),
        ])

        rows.append(FeatureRow(values=values, option_label=stance.label))

    return rows
