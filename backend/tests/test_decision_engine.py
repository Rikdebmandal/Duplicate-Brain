"""Unit tests for parsing, stance resolution and the utility model."""
from __future__ import annotations

import pytest

from app.services.decision_engine.parser import (
    classify_option,
    extract_emotions,
    extract_factors,
    parse_scenario,
    resolve_stances,
)
from app.services.decision_engine.taxonomy import FACTOR_KEYS, TRAIT_KEYS, neutral_traits
from app.services.decision_engine.utility import ModelParams, UtilityModel, softmax


class TestFactorExtraction:
    def test_detects_the_obvious_factors(self):
        result = extract_factors(
            "A job paying double my salary, but I must move to another city away "
            "from my parents and the company is only one year old."
        )
        assert result.factors["financial_reward"] > 0.5
        assert result.factors["family_impact"] > 0.4
        assert result.factors["uncertainty"] > 0.2
        assert result.coverage > 0

    def test_magnitudes_stay_in_range(self):
        result = extract_factors("risk risk risk risky gamble danger lose " * 20)
        for key in FACTOR_KEYS:
            assert 0.0 <= result.factors[key] <= 1.0

    def test_negation_suppresses_a_factor(self):
        with_risk = extract_factors("This carries real risk of failure.")
        without = extract_factors("There is no risk of failure here.")
        assert without.factors["risk"] < with_risk.factors["risk"]

    def test_intensifier_raises_magnitude(self):
        plain = extract_factors("The job is risky.")
        strong = extract_factors("The job is extremely risky.")
        assert strong.factors["risk"] > plain.factors["risk"]

    def test_numeric_multiplier_amplifies_reward(self):
        modest = extract_factors("The salary is a bit higher.")
        large = extract_factors("The salary is 3x higher.")
        assert large.factors["financial_reward"] > modest.factors["financial_reward"]

    def test_empty_text_is_all_zero(self):
        result = extract_factors("")
        assert all(value == 0.0 for value in result.factors.values())
        assert result.coverage == 0.0

    def test_every_hit_carries_its_context(self):
        result = extract_factors("I would have to relocate away from my family.")
        hits = result.evidence_for("family_impact")
        assert hits
        assert all(hit.context for hit in hits)


class TestStanceResolution:
    def test_accept_versus_reject(self):
        approach, avoid = resolve_stances(["Accept the offer", "Reject the offer"])
        assert approach.approach > 0.8
        assert avoid.approach < 0.2
        assert approach.confidence > 0.4

    def test_both_options_worded_as_take(self):
        """The case independent classification gets exactly backwards.

        "Take the stock options" and "Take the cash bonus" both begin with an
        approach verb; only the semantic axis separates them.
        """
        equity, cash = resolve_stances(["Take the stock options", "Take the cash bonus"])
        assert equity.approach > cash.approach

    def test_loan_reads_as_more_active_than_paying_from_savings(self):
        pay, loan = resolve_stances(
            ["Pay it from savings immediately", "Take a low-interest loan"]
        )
        assert loan.approach > pay.approach

    def test_options_with_no_cues_still_separate(self):
        stances = resolve_stances(["Put it in the index fund", "Keep it in the fixed deposit"])
        assert stances[0].approach > stances[1].approach
        assert stances[0].confidence > 0.25

    def test_lateral_choice_is_reported_as_undecidable(self):
        """A choice that is not approach-vs-avoid must not be forced onto the axis."""
        stances = resolve_stances(["The blue one", "The green one"])
        assert all(s.approach == 0.5 for s in stances)
        assert all(s.confidence < 0.2 for s in stances)

    def test_deferral_is_detected_and_pulled_off_the_extremes(self):
        commit, wait = resolve_stances(
            ["Commit today for the discount", "Let it go and check properly first"]
        )
        assert wait.defer > 0
        assert wait.approach < commit.approach

    def test_three_options_are_spread_across_the_axis(self):
        stances = resolve_stances(
            ["Accept the offer", "Negotiate a counter-offer", "Reject the offer"]
        )
        assert stances[0].approach > stances[2].approach
        assert stances[1].compromise > 0

    def test_single_option_helper_uses_alternatives(self):
        alone = classify_option("Take the cash bonus")
        against = classify_option("Take the cash bonus", ["Take the stock options"])
        assert against.approach < alone.approach or against.confidence >= alone.confidence


class TestEmotions:
    def test_detects_stated_emotion(self):
        emotions = extract_emotions("I felt anxious and torn about the whole thing.")
        assert "anxious" in emotions
        assert "uncertain" in emotions

    def test_neutral_text_yields_nothing(self):
        assert extract_emotions("The meeting is on Tuesday.") == {}


class TestUtilityModel:
    def make_model(self, **traits) -> UtilityModel:
        values = neutral_traits()
        values.update(traits)
        return UtilityModel(values, ModelParams(temperature=0.55))

    def test_family_priority_pushes_away_from_the_disruptive_option(self):
        parsed = parse_scenario(
            "A job in another city away from my parents, paying more.",
            ["Accept and relocate", "Stay here"],
        )
        family_first = self.make_model(family_priority=0.95)
        indifferent = self.make_model(family_priority=0.05)

        p_family, _ = family_first.probabilities(parsed.factors, parsed.options)
        p_indifferent, _ = indifferent.probabilities(parsed.factors, parsed.options)
        assert p_family[0] < p_indifferent[0]

    def test_contributions_sum_to_the_utility(self):
        parsed = parse_scenario(
            "A risky but very well paid role that means leaving my family.",
            ["Accept", "Decline"],
        )
        model = self.make_model()
        scored = model.score_options(parsed.factors, parsed.options)
        for option in scored:
            total = sum(c.value for c in option.contributions) + sum(option.extras.values())
            assert option.utility == pytest.approx(total, abs=1e-9)

    def test_reward_damps_aversion_but_never_creates_appetite(self):
        model = self.make_model(risk_tolerance=0.5, reward_sensitivity=1.0)
        no_reward = model.risk_valence({"financial_reward": 0.0})
        big_reward = model.risk_valence({"financial_reward": 1.0})
        assert no_reward == pytest.approx(0.0, abs=1e-9)
        assert big_reward == pytest.approx(0.0, abs=1e-9)

    def test_reward_reduces_aversion_for_a_risk_averse_person(self):
        model = self.make_model(risk_tolerance=0.2, reward_sensitivity=0.9)
        modest = model.risk_valence({"financial_reward": 0.2})
        huge = model.risk_valence({"financial_reward": 1.0})
        assert modest < 0
        assert huge > modest       # aversion has softened
        assert huge <= 0           # but has not become appetite

    def test_risk_never_makes_the_risky_option_more_attractive_when_averse(self):
        """Regression: the sign of the risk term must not invert under high reward."""
        model = self.make_model(risk_tolerance=0.3, reward_sensitivity=1.0)
        stances = resolve_stances(["Accept the offer", "Reject the offer"])
        base = {k: 0.0 for k in FACTOR_KEYS}
        base["financial_reward"] = 1.0

        low_risk = dict(base, risk=0.0)
        high_risk = dict(base, risk=1.0)
        p_low, _ = model.probabilities(low_risk, stances)
        p_high, _ = model.probabilities(high_risk, stances)
        assert p_high[0] <= p_low[0] + 1e-9

    def test_loss_aversion_amplifies_the_downside(self):
        parsed = parse_scenario(
            "A job that pays more but means giving up a stable secure position.",
            ["Accept", "Stay"],
        )
        averse = self.make_model(loss_aversion=1.0)
        neutral = self.make_model(loss_aversion=0.0)
        p_averse, _ = averse.probabilities(parsed.factors, parsed.options)
        p_neutral, _ = neutral.probabilities(parsed.factors, parsed.options)
        assert p_averse[0] < p_neutral[0]

    def test_time_pressure_discounts_long_term_benefit(self):
        model = self.make_model(long_term_orientation=0.9)
        relaxed = model.factor_valences({"time_pressure": 0.0})
        rushed = model.factor_valences({"time_pressure": 1.0})
        assert rushed["long_term_benefit"] < relaxed["long_term_benefit"]

    def test_status_quo_bias_shifts_towards_the_passive_option(self):
        parsed = parse_scenario("A new opportunity.", ["Take it", "Stay as I am"])
        traits = neutral_traits()
        biased = UtilityModel(traits, ModelParams(temperature=0.55, status_quo_bias=1.0))
        unbiased = UtilityModel(traits, ModelParams(temperature=0.55, status_quo_bias=0.0))
        p_biased, _ = biased.probabilities(parsed.factors, parsed.options)
        p_unbiased, _ = unbiased.probabilities(parsed.factors, parsed.options)
        assert p_biased[1] > p_unbiased[1]

    def test_traits_are_clamped_to_the_unit_interval(self):
        model = UtilityModel({"risk_tolerance": 5.0, "family_priority": -3.0})
        assert model.traits["risk_tolerance"] == 1.0
        assert model.traits["family_priority"] == 0.0

    def test_unknown_trait_keys_are_ignored(self):
        model = UtilityModel({"not_a_trait": 0.9})
        assert set(model.traits) == set(TRAIT_KEYS)


class TestSoftmax:
    def test_sums_to_one(self):
        assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_lower_temperature_sharpens(self):
        sharp = softmax([1.0, 0.0], 0.2)
        soft = softmax([1.0, 0.0], 2.0)
        assert sharp[0] > soft[0]

    def test_handles_large_values_without_overflow(self):
        result = softmax([1000.0, 999.0], 0.01)
        assert sum(result) == pytest.approx(1.0)

    def test_empty_input(self):
        assert softmax([]) == []


class TestParseScenario:
    def test_option_text_contributes_to_the_factor_reading(self):
        parsed = parse_scenario(
            "I have a decision to make about work.",
            ["Accept the role and relocate away from my family", "Stay put"],
        )
        assert parsed.factors["family_impact"] > 0

    def test_overrides_replace_extracted_values(self):
        parsed = parse_scenario(
            "A very risky venture.", ["Accept", "Decline"], factor_overrides={"risk": 0.1}
        )
        assert parsed.factors["risk"] == 0.1

    def test_overrides_are_clamped(self):
        parsed = parse_scenario(
            "A venture.", ["Accept", "Decline"], factor_overrides={"risk": 9.0}
        )
        assert parsed.factors["risk"] == 1.0
