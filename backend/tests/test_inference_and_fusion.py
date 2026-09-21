"""Trait inference, layer fusion, calibration and evaluation metrics."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.ml.metrics import calibration_bins, evaluate, multiclass_brier
from app.services.decision_engine.parser import resolve_stances
from app.services.decision_engine.taxonomy import FACTOR_KEYS, TRAIT_KEYS
from app.services.prediction import fusion
from app.services.prediction.retrieval import baseline_distribution
from app.services.profile.inference import (
    Observation,
    accumulate,
    importance_weight,
    observations_from_decision,
    recency_weight,
)


def blank(**overrides) -> dict:
    factors = {key: 0.0 for key in FACTOR_KEYS}
    factors.update(overrides)
    return factors


class TestTraitInference:
    def test_no_evidence_means_no_confidence(self):
        posteriors = accumulate([])
        for key in TRAIT_KEYS:
            assert posteriors[key].value == pytest.approx(0.5)
            assert posteriors[key].confidence == pytest.approx(0.0, abs=1e-9)
            assert posteriors[key].evidence_count == 0

    def test_consistent_evidence_moves_the_value_and_raises_confidence(self):
        observations = [
            Observation(trait="risk_tolerance", value=1.0, weight=1.0, source="decision")
            for _ in range(15)
        ]
        posterior = accumulate(observations)["risk_tolerance"]
        assert posterior.value > 0.85
        assert posterior.confidence > 0.6
        assert posterior.evidence_count == 15

    def test_contradictory_evidence_stays_near_the_middle(self):
        observations = (
            [Observation("risk_tolerance", 1.0, 1.0, "decision") for _ in range(8)]
            + [Observation("risk_tolerance", 0.0, 1.0, "decision") for _ in range(8)]
        )
        posterior = accumulate(observations)["risk_tolerance"]
        assert posterior.value == pytest.approx(0.5, abs=0.05)
        assert posterior.consistency < 0.2

    def test_behaviour_outweighs_self_report(self):
        behavioural = accumulate(
            [Observation("risk_tolerance", 1.0, 1.0, "decision")]
        )["risk_tolerance"]
        self_reported = accumulate(
            [Observation("risk_tolerance", 1.0, 1.0, "questionnaire")]
        )["risk_tolerance"]
        assert behavioural.value > self_reported.value

    def test_credible_interval_narrows_with_evidence(self):
        thin = accumulate(
            [Observation("family_priority", 1.0, 1.0, "decision")]
        )["family_priority"]
        thick = accumulate(
            [Observation("family_priority", 1.0, 1.0, "decision") for _ in range(30)]
        )["family_priority"]
        thin_lo, thin_hi = thin.credible_interval()
        thick_lo, thick_hi = thick.credible_interval()
        assert (thick_hi - thick_lo) < (thin_hi - thin_lo)

    def test_recency_decay(self):
        now = datetime.now(timezone.utc)
        assert recency_weight(now, now) == pytest.approx(1.0)
        three_years = recency_weight(now - timedelta(days=365 * 3), now)
        assert three_years == pytest.approx(0.5, abs=0.02)
        assert recency_weight(now - timedelta(days=365 * 9), now) < 0.15

    def test_importance_scales_weight(self):
        assert importance_weight(10) > importance_weight(1)

    def test_decision_evidence_reflects_the_choice_direction(self):
        approach, avoid = resolve_stances(["Accept the risky offer", "Decline it"])
        took_risk = observations_from_decision(
            decision_id="d1", factors=blank(risk=0.9), chosen_stance=approach, importance=8
        )
        declined = observations_from_decision(
            decision_id="d2", factors=blank(risk=0.9), chosen_stance=avoid, importance=8
        )
        risk_took = [o for o in took_risk if o.trait == "risk_tolerance"][0]
        risk_declined = [o for o in declined if o.trait == "risk_tolerance"][0]
        assert risk_took.value > 0.8
        assert risk_declined.value < 0.2

    def test_absent_factors_produce_no_evidence(self):
        approach, _ = resolve_stances(["Accept", "Decline"])
        observations = observations_from_decision(
            decision_id="d1", factors=blank(risk=0.0), chosen_stance=approach
        )
        assert not [o for o in observations if o.trait == "risk_tolerance"]

    def test_family_evidence_is_inverted(self):
        """Declining a family-disrupting option is evidence of *high* family priority."""
        _, avoid = resolve_stances(["Accept and relocate", "Stay near my family"])
        observations = observations_from_decision(
            decision_id="d1", factors=blank(family_impact=0.9), chosen_stance=avoid
        )
        family = [o for o in observations if o.trait == "family_priority"][0]
        assert family.value > 0.8

    def test_stated_reasoning_informs_planning_tendency(self):
        approach, _ = resolve_stances(["Accept", "Decline"])
        observations = observations_from_decision(
            decision_id="d1",
            factors=blank(risk=0.5),
            chosen_stance=approach,
            reason="I researched it thoroughly and compared every option before deciding.",
        )
        planning = [o for o in observations if o.trait == "planning_tendency" and o.source == "text"]
        assert planning and planning[0].value > 0.5


class TestFusion:
    def test_pooling_respects_weights(self):
        layers = [
            fusion.LayerOutput("a", [0.9, 0.1], 3.0),
            fusion.LayerOutput("b", [0.3, 0.7], 1.0),
        ]
        pooled = fusion.log_linear_pool(layers)
        assert sum(pooled) == pytest.approx(1.0)
        assert pooled[0] > 0.5

    def test_unavailable_layers_are_ignored(self):
        layers = [
            fusion.LayerOutput("a", [0.8, 0.2], 1.0),
            fusion.LayerOutput("b", [], 0.0, available=False),
        ]
        assert fusion.log_linear_pool(layers) == pytest.approx([0.8, 0.2], abs=1e-6)

    def test_geometric_pooling_is_less_extreme_than_the_confident_layer(self):
        """A single certain layer must not drag the pool to certainty."""
        layers = [
            fusion.LayerOutput("confident", [0.99, 0.01], 1.0),
            fusion.LayerOutput("unsure", [0.5, 0.5], 1.0),
        ]
        pooled = fusion.log_linear_pool(layers)
        assert pooled[0] < 0.99

    def test_agreement_is_high_when_layers_concur(self):
        layers = [
            fusion.LayerOutput("a", [0.8, 0.2], 1.0),
            fusion.LayerOutput("b", [0.79, 0.21], 1.0),
        ]
        assert fusion.layer_agreement(layers) > 0.9

    def test_agreement_is_low_when_layers_conflict(self):
        layers = [
            fusion.LayerOutput("a", [0.95, 0.05], 1.0),
            fusion.LayerOutput("b", [0.05, 0.95], 1.0),
        ]
        assert fusion.layer_agreement(layers) < 0.3

    def test_calibration_temperature_flattens_and_sharpens(self):
        base = [0.9, 0.1]
        assert fusion.apply_calibration(base, 2.0)[0] < 0.9
        assert fusion.apply_calibration(base, 0.5)[0] > 0.9
        assert fusion.apply_calibration(base, 1.0) == base

    def test_confidence_ignores_the_predicted_probability(self):
        """Confidence measures the evidence, not the sharpness of the answer."""
        thin, thin_label, _ = fusion.confidence_from(
            agreement=0.9, decision_count=1, factor_coverage=0.9,
            trait_confidence=0.1, retrieval_support=0.0,
        )
        thick, thick_label, _ = fusion.confidence_from(
            agreement=0.9, decision_count=60, factor_coverage=0.9,
            trait_confidence=0.9, retrieval_support=0.9,
        )
        assert thin < thick
        assert thin_label == "low"
        assert thick_label == "high"

    def test_fuse_returns_none_without_any_layer(self):
        assert fusion.fuse(
            [], decision_count=0, factor_coverage=0.0,
            trait_confidence=0.0, retrieval_support=0.0,
        ) is None


class TestBaseline:
    def test_baseline_follows_the_base_rate(self):
        stances = resolve_stances(["Accept the offer", "Reject the offer"])
        mostly_declines = baseline_distribution(stances, approach_base_rate=0.2)
        assert mostly_declines[1] > mostly_declines[0]


class TestMetrics:
    def test_multiclass_brier(self):
        assert multiclass_brier([1.0, 0.0], 0) == pytest.approx(0.0)
        assert multiclass_brier([0.0, 1.0], 0) == pytest.approx(2.0)
        assert multiclass_brier([0.5, 0.5], 0) == pytest.approx(0.5)

    def test_evaluate_on_a_perfect_predictor(self):
        cases = [([0.99, 0.01], 0), ([0.02, 0.98], 1), ([0.95, 0.05], 0)]
        result = evaluate(cases)
        assert result.accuracy == pytest.approx(1.0)
        assert result.brier < 0.02
        assert result.roc_auc == pytest.approx(1.0)

    def test_evaluate_on_an_always_wrong_predictor(self):
        cases = [([0.9, 0.1], 1), ([0.1, 0.9], 0)]
        result = evaluate(cases)
        assert result.accuracy == pytest.approx(0.0)
        assert result.log_loss > 2.0

    def test_calibration_detects_overconfidence(self):
        # Claims 90% confidence, right only half the time.
        confidences = [0.9] * 10
        correct = [True] * 5 + [False] * 5
        _, ece = calibration_bins(confidences, correct)
        assert ece == pytest.approx(0.4, abs=0.01)

    def test_calibration_is_near_zero_when_honest(self):
        confidences = [0.7] * 10
        correct = [True] * 7 + [False] * 3
        _, ece = calibration_bins(confidences, correct)
        assert ece < 0.01

    def test_handles_variable_option_counts(self):
        cases = [([0.5, 0.3, 0.2], 0), ([0.6, 0.4], 1)]
        result = evaluate(cases)
        assert result.n == 2
        assert 0.0 <= result.accuracy <= 1.0

    def test_roc_auc_is_none_with_one_class(self):
        assert evaluate([([1.0], 0)]).roc_auc is None

    def test_by_category_breakdown(self):
        cases = [([0.9, 0.1], 0), ([0.8, 0.2], 0), ([0.4, 0.6], 1), ([0.3, 0.7], 1)]
        result = evaluate(cases, categories=["career", "career", "finance", "finance"])
        assert set(result.by_category) == {"career", "finance"}

    def test_empty_input(self):
        result = evaluate([])
        assert result.n == 0
        assert result.accuracy == 0.0
