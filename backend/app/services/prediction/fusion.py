"""Layer 4: combining the layers into one calibrated distribution.

Pooling is **log-linear** (a weighted geometric mean), not a weighted average.
The difference matters: with an arithmetic mean, one confident layer can drag
the result to an extreme on its own. With a geometric mean, a layer that says
"almost certainly not this option" can veto, but no single layer can force
certainty past what the others support - which is the behaviour you want from a
system whose main failure mode is over-confidence.

Weights are *adaptive*: each layer earns weight according to how much evidence
it actually had. A statistical model trained on 14 decisions is downweighted
against one trained on 90; a retrieval layer whose best match scored 0.15 gets
almost none.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

_EPSILON = 1e-6


@dataclass
class LayerOutput:
    """One layer's opinion, with the evidence that justifies weighting it."""

    name: str
    probabilities: list[float]
    weight: float
    available: bool = True
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "weight": round(self.weight, 4),
            "probabilities": [round(p, 4) for p in self.probabilities],
            "detail": self.detail,
        }


@dataclass
class FusionResult:
    probabilities: list[float]
    layers: list[LayerOutput]
    agreement: float
    confidence_score: float
    confidence_label: str
    confidence_inputs: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "probabilities": [round(p, 4) for p in self.probabilities],
            "layers": [layer.to_dict() for layer in self.layers],
            "agreement": round(self.agreement, 4),
            "confidence_score": round(self.confidence_score, 4),
            "confidence_label": self.confidence_label,
            "confidence_inputs": {k: round(v, 4) for k, v in self.confidence_inputs.items()},
        }


def _normalise(values: Sequence[float]) -> list[float]:
    total = sum(max(0.0, v) for v in values)
    if total <= 0:
        n = max(1, len(values))
        return [1.0 / n] * n
    return [max(0.0, v) / total for v in values]


def log_linear_pool(layers: Sequence[LayerOutput]) -> list[float]:
    """Weighted geometric mean of the available layer distributions."""
    active = [layer for layer in layers if layer.available and layer.weight > 0]
    if not active:
        return []
    n_options = len(active[0].probabilities)
    total_weight = sum(layer.weight for layer in active) or 1.0

    log_scores = [0.0] * n_options
    for layer in active:
        share = layer.weight / total_weight
        for index in range(n_options):
            p = max(_EPSILON, layer.probabilities[index])
            log_scores[index] += share * math.log(p)

    top = max(log_scores)
    return _normalise([math.exp(value - top) for value in log_scores])


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """JS divergence in bits, in [0, 1] for two distributions."""
    m = [(a + b) / 2.0 for a, b in zip(p, q, strict=False)]

    def kl(x: Sequence[float], y: Sequence[float]) -> float:
        return sum(
            xi * math.log2(xi / yi)
            for xi, yi in zip(x, y, strict=False)
            if xi > _EPSILON and yi > _EPSILON
        )

    return float(max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def layer_agreement(layers: Sequence[LayerOutput]) -> float:
    """1.0 when every available layer says the same thing, 0.0 when opposed."""
    active = [layer for layer in layers if layer.available and layer.weight > 0]
    if len(active) < 2:
        return 0.5  # nothing to corroborate: neither agreement nor conflict
    divergences = [
        jensen_shannon(active[i].probabilities, active[j].probabilities)
        for i in range(len(active))
        for j in range(i + 1, len(active))
    ]
    return float(1.0 - sum(divergences) / len(divergences))


def apply_calibration(probabilities: Sequence[float], temperature: float) -> list[float]:
    """Sharpen or flatten a distribution using the fitted calibration temperature."""
    t = max(0.2, min(5.0, float(temperature or 1.0)))
    if abs(t - 1.0) < 1e-6:
        return list(probabilities)
    powered = [max(_EPSILON, p) ** (1.0 / t) for p in probabilities]
    return _normalise(powered)


def confidence_from(
    *,
    agreement: float,
    decision_count: int,
    factor_coverage: float,
    trait_confidence: float,
    retrieval_support: float,
    target_decisions: int = 25,
) -> tuple[float, str, dict[str, float]]:
    """Blend the things that actually determine whether to trust a prediction.

    Deliberately *not* a function of the predicted probability. A model can be
    90% sure and have no right to be; confidence here measures the evidence
    behind the estimate, not the sharpness of the estimate itself.

    Evidence volume enters twice: once as a term, and again as a multiplier that
    caps the whole score. That is not double counting - it encodes that nothing
    substitutes for having data. Without the cap, a scenario that parsed cleanly
    and two layers that happened to agree could report medium confidence off a
    single recorded decision, which is exactly the kind of unearned certainty
    this system exists to avoid.
    """
    evidence = min(1.0, decision_count / float(target_decisions))
    inputs = {
        "evidence_volume": evidence,
        "layer_agreement": max(0.0, min(1.0, agreement)),
        "scenario_coverage": max(0.0, min(1.0, factor_coverage)),
        "trait_confidence": max(0.0, min(1.0, trait_confidence)),
        "retrieval_support": max(0.0, min(1.0, retrieval_support)),
    }
    weights = {
        "evidence_volume": 0.30,
        "layer_agreement": 0.25,
        "scenario_coverage": 0.15,
        "trait_confidence": 0.20,
        "retrieval_support": 0.10,
    }
    blended = sum(inputs[key] * weights[key] for key in weights)
    sufficiency = 0.45 + 0.55 * evidence
    score = blended * sufficiency
    inputs["evidence_sufficiency"] = sufficiency

    if score < 0.35:
        label = "low"
    elif score < 0.62:
        label = "medium"
    else:
        label = "high"
    return float(score), label, inputs


def fuse(
    layers: Sequence[LayerOutput],
    *,
    decision_count: int,
    factor_coverage: float,
    trait_confidence: float,
    retrieval_support: float,
    calibration_temperature: float = 1.0,
) -> FusionResult | None:
    active = [layer for layer in layers if layer.available and layer.weight > 0]
    if not active:
        return None

    pooled = log_linear_pool(active)
    calibrated = apply_calibration(pooled, calibration_temperature)
    agreement = layer_agreement(active)
    score, label, inputs = confidence_from(
        agreement=agreement,
        decision_count=decision_count,
        factor_coverage=factor_coverage,
        trait_confidence=trait_confidence,
        retrieval_support=retrieval_support,
    )
    return FusionResult(
        probabilities=calibrated,
        layers=list(layers),
        agreement=agreement,
        confidence_score=score,
        confidence_label=label,
        confidence_inputs=inputs,
    )
