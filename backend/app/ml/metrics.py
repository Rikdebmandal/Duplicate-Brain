"""Evaluation metrics for a variable-option-count prediction problem.

Accuracy alone is close to useless here. A person who accepts 80% of the time
gives a "predict accept always" baseline 80% accuracy while telling you nothing
about *when* they decline. What matters is whether the probabilities are
honest, so the primary metrics are the Brier score, log loss and expected
calibration error, with accuracy reported alongside rather than optimised.

All metrics accept the same shape: a list of ``(probabilities, chosen_index)``
pairs, where ``probabilities`` has one entry per option available in that
decision. Option counts may differ between decisions.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

Case = tuple[Sequence[float], int]


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    observed_accuracy: float

    def to_dict(self) -> dict:
        return {
            "lower": round(self.lower, 3),
            "upper": round(self.upper, 3),
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 4),
            "observed_accuracy": round(self.observed_accuracy, 4),
            "gap": round(self.mean_confidence - self.observed_accuracy, 4),
        }


@dataclass
class EvaluationResult:
    n: int = 0
    accuracy: float = 0.0
    baseline_accuracy: float = 0.0
    brier: float = 0.0
    log_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float | None = None
    ece: float = 0.0
    mean_confidence: float = 0.0
    calibration_bins: list[CalibrationBin] = field(default_factory=list)
    by_category: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "brier": round(self.brier, 4),
            "log_loss": round(self.log_loss, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": None if self.roc_auc is None else round(self.roc_auc, 4),
            "ece": round(self.ece, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "calibration_bins": [b.to_dict() for b in self.calibration_bins],
            "by_category": self.by_category,
        }


def multiclass_brier(probabilities: Sequence[float], chosen_index: int) -> float:
    """Sum of squared error across all options for one decision."""
    return float(
        sum((p - (1.0 if i == chosen_index else 0.0)) ** 2 for i, p in enumerate(probabilities))
    )


def negative_log_likelihood(probabilities: Sequence[float], chosen_index: int) -> float:
    p = probabilities[chosen_index] if 0 <= chosen_index < len(probabilities) else 1e-9
    return float(-math.log(max(1e-9, p)))


def calibration_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> tuple[list[CalibrationBin], float]:
    """Reliability diagram plus expected calibration error."""
    bins: list[CalibrationBin] = []
    total = len(confidences)
    if total == 0:
        return bins, 0.0

    ece = 0.0
    for i in range(n_bins):
        lower, upper = i / n_bins, (i + 1) / n_bins
        members = [
            (c, ok)
            for c, ok in zip(confidences, correct, strict=False)
            if (lower < c <= upper) or (i == 0 and c <= upper)
        ]
        if not members:
            continue
        mean_conf = sum(c for c, _ in members) / len(members)
        observed = sum(1 for _, ok in members if ok) / len(members)
        bins.append(CalibrationBin(lower, upper, len(members), mean_conf, observed))
        ece += (len(members) / total) * abs(mean_conf - observed)

    return bins, float(ece)


def _roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank-based AUC with tie handling; None when only one class is present."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1

    rank_sum = sum(r for r, label in zip(ranks, labels, strict=False) if label == 1)
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def evaluate(
    cases: Sequence[Case],
    *,
    categories: Sequence[str] | None = None,
    baseline_cases: Sequence[Case] | None = None,
    n_bins: int = 10,
) -> EvaluationResult:
    """Score a set of predictions.

    ``precision``/``recall``/``f1``/``roc_auc`` are computed at the *option*
    level - each (decision, option) pair is one binary example labelled "was
    this the option taken" - which is the only framing that stays well defined
    when different decisions offer different numbers of options.
    """
    result = EvaluationResult(n=len(cases))
    if not cases:
        return result

    confidences: list[float] = []
    correct: list[bool] = []
    option_scores: list[float] = []
    option_labels: list[int] = []
    brier_total = 0.0
    nll_total = 0.0
    hits = 0

    for probabilities, chosen_index in cases:
        probs = list(probabilities)
        if not probs:
            continue
        predicted_index = max(range(len(probs)), key=lambda i: probs[i])
        is_correct = predicted_index == chosen_index
        hits += int(is_correct)
        confidences.append(probs[predicted_index])
        correct.append(is_correct)
        brier_total += multiclass_brier(probs, chosen_index)
        nll_total += negative_log_likelihood(probs, chosen_index)
        for i, p in enumerate(probs):
            option_scores.append(p)
            option_labels.append(1 if i == chosen_index else 0)

    n = len(correct)
    result.accuracy = hits / n if n else 0.0
    result.brier = brier_total / n if n else 0.0
    result.log_loss = nll_total / n if n else 0.0
    result.mean_confidence = sum(confidences) / n if n else 0.0
    result.calibration_bins, result.ece = calibration_bins(confidences, correct, n_bins)

    # Option-level classification quality at a 0.5 threshold.
    tp = sum(1 for s, y in zip(option_scores, option_labels, strict=False) if s >= 0.5 and y == 1)
    fp = sum(1 for s, y in zip(option_scores, option_labels, strict=False) if s >= 0.5 and y == 0)
    fn = sum(1 for s, y in zip(option_scores, option_labels, strict=False) if s < 0.5 and y == 1)
    result.precision = tp / (tp + fp) if (tp + fp) else 0.0
    result.recall = tp / (tp + fn) if (tp + fn) else 0.0
    result.f1 = (
        2 * result.precision * result.recall / (result.precision + result.recall)
        if (result.precision + result.recall) else 0.0
    )
    result.roc_auc = _roc_auc(option_scores, option_labels)

    if baseline_cases:
        baseline_hits = 0
        for probabilities, chosen_index in baseline_cases:
            probs = list(probabilities)
            if not probs:
                continue
            baseline_hits += int(
                max(range(len(probs)), key=lambda i: probs[i]) == chosen_index
            )
        result.baseline_accuracy = baseline_hits / len(baseline_cases)

    if categories:
        grouped: dict[str, list[Case]] = {}
        for case, category in zip(cases, categories, strict=False):
            grouped.setdefault(category or "general", []).append(case)
        for category, subset in grouped.items():
            if len(subset) < 2:
                continue
            sub = evaluate(subset, n_bins=min(5, n_bins))
            result.by_category[category] = {
                "n": sub.n,
                "accuracy": round(sub.accuracy, 4),
                "brier": round(sub.brier, 4),
                "log_loss": round(sub.log_loss, 4),
            }

    return result
