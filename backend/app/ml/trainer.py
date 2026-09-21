"""The statistical layer: a conditional-logit style learner over option rows.

One row per (decision, option); label 1 for the option taken. At inference the
classifier scores every option and the scores are renormalised over the option
set, which keeps the model valid for two-option and five-option decisions
alike.

Two estimators are fitted and the better-calibrated one wins:

* **logistic regression** - few parameters, stable on the tens-of-decisions
  scale this system actually operates at, and its coefficients are directly
  readable as feature importances;
* **gradient boosting** - can pick up interactions the utility model misses,
  but needs materially more data before it stops overfitting.

Model selection uses *time-ordered* expanding-window validation, never a random
split. Predicting your own past from your own future is the one mistake that
would make every reported number meaningless here.
"""
from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.logging_conf import get_logger
from app.ml.metrics import EvaluationResult, evaluate
from app.models import ModelArtifact
from app.services.decision_engine.features import FEATURE_NAMES, build_features
from app.services.decision_engine.utility import UtilityModel
from app.services.profile.service import TrainingCase

log = get_logger(__name__)

#: Below this many decisions the statistical layer is not fitted at all; the
#: prediction falls back to the utility model plus retrieval, which degrade
#: gracefully instead of memorising a handful of rows.
MIN_DECISIONS = settings.min_decisions_for_ml
#: Gradient boosting is only considered once there is enough data for it to be
#: anything other than a lookup table.
MIN_DECISIONS_FOR_BOOSTING = 40


@dataclass
class TrainedModel:
    """A fitted estimator plus everything needed to use and explain it."""

    kind: str
    estimator: object
    feature_names: list[str]
    importances: dict[str, float]
    metrics: dict
    n_train_rows: int
    n_train_decisions: int
    version: str

    def score_options(self, feature_rows: Sequence[Sequence[float]]) -> list[float]:
        """Normalised probability over the options of a single decision."""
        matrix = np.array(feature_rows, dtype=float)
        raw = self.estimator.predict_proba(matrix)[:, 1]
        raw = np.clip(raw, 1e-6, 1 - 1e-6)
        total = float(raw.sum())
        if total <= 0:
            return [1.0 / len(raw)] * len(raw)
        return [float(v) for v in (raw / total)]


def build_dataset(
    cases: Sequence[TrainingCase], model: UtilityModel
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Flatten decisions into option rows. ``groups`` marks decision boundaries."""
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []

    for group_index, case in enumerate(cases):
        rows = build_features(case.factors, case.stances, model)
        for option_index, row in enumerate(rows):
            features.append(row.values)
            labels.append(1 if option_index == case.chosen_index else 0)
            groups.append(group_index)

    if not features:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,)), []
    return np.array(features, dtype=float), np.array(labels, dtype=int), groups


def _fit_estimator(kind: str, X: np.ndarray, y: np.ndarray):
    if kind == "logistic":
        estimator = LogisticRegression(
            C=0.5,                 # deliberately strong regularisation: n is small
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    else:
        estimator = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=2,           # stumps-plus: interactions without memorising
            subsample=0.9,
            random_state=17,
        )
    estimator.fit(X, y)
    return estimator


def _predict_grouped(
    estimator, X: np.ndarray, groups: Sequence[int], cases: Sequence[TrainingCase]
) -> list[tuple[list[float], int]]:
    """Group row-level scores back into per-decision distributions."""
    if len(X) == 0:
        return []
    raw = np.clip(estimator.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    by_group: dict[int, list[float]] = {}
    for value, group in zip(raw, groups, strict=False):
        by_group.setdefault(group, []).append(float(value))

    results: list[tuple[list[float], int]] = []
    for group in sorted(by_group):
        scores = by_group[group]
        total = sum(scores)
        probs = [s / total for s in scores] if total > 0 else [1 / len(scores)] * len(scores)
        results.append((probs, cases[group].chosen_index))
    return results


def time_series_validate(
    kind: str,
    cases: Sequence[TrainingCase],
    model: UtilityModel,
    n_splits: int = 3,
) -> EvaluationResult | None:
    """Expanding-window validation: always train on the past, test on the future."""
    n = len(cases)
    if n < MIN_DECISIONS:
        return None

    initial = max(MIN_DECISIONS - 4, int(n * 0.5))
    if initial >= n - 1:
        return None
    fold_size = max(1, (n - initial) // n_splits)

    collected: list[tuple[list[float], int]] = []
    categories: list[str] = []

    start = initial
    while start < n:
        end = min(n, start + fold_size)
        train_cases = list(cases[:start])
        test_cases = list(cases[start:end])
        if not test_cases:
            break

        X_train, y_train, _ = build_dataset(train_cases, model)
        if len(set(y_train.tolist())) < 2:
            start = end
            continue
        try:
            estimator = _fit_estimator(kind, X_train, y_train)
        except Exception as exc:  # pragma: no cover - degenerate folds
            log.warning("fold fit failed for %s: %s", kind, exc)
            start = end
            continue

        X_test, _, groups_test = build_dataset(test_cases, model)
        collected.extend(_predict_grouped(estimator, X_test, groups_test, test_cases))
        categories.extend(c.category for c in test_cases)
        start = end

    if not collected:
        return None
    return evaluate(collected, categories=categories)


def _importances(kind: str, estimator, feature_names: Sequence[str]) -> dict[str, float]:
    if kind == "logistic":
        values = np.asarray(estimator.coef_).ravel()
    else:
        values = np.asarray(estimator.feature_importances_).ravel()
    return {
        name: round(float(value), 6)
        for name, value in zip(feature_names, values, strict=False)
    }


def train_for_user(
    db: Session,
    user_id: str,
    cases: Sequence[TrainingCase],
    model: UtilityModel,
) -> TrainedModel | None:
    """Fit, validate and persist the best statistical model for one person."""
    if len(cases) < MIN_DECISIONS:
        log.info(
            "skipping ML training for user=%s: %d decisions (need %d)",
            user_id, len(cases), MIN_DECISIONS,
        )
        return None

    candidates = ["logistic"]
    if len(cases) >= MIN_DECISIONS_FOR_BOOSTING:
        candidates.append("gradient_boosting")

    scored: list[tuple[str, EvaluationResult]] = []
    for kind in candidates:
        result = time_series_validate(kind, cases, model)
        if result is not None:
            scored.append((kind, result))
            log.info(
                "cv %s: acc=%.3f brier=%.3f ece=%.3f",
                kind, result.accuracy, result.brier, result.ece,
            )

    if not scored:
        return None

    # Selection is on the Brier score, not accuracy: an honest 0.6 beats an
    # over-confident 0.95 that is wrong when it matters.
    best_kind, best_metrics = min(scored, key=lambda item: item[1].brier)

    X, y, _ = build_dataset(cases, model)
    if len(set(y.tolist())) < 2:
        return None
    estimator = _fit_estimator(best_kind, X, y)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    trained = TrainedModel(
        kind=best_kind,
        estimator=estimator,
        feature_names=list(FEATURE_NAMES),
        importances=_importances(best_kind, estimator, FEATURE_NAMES),
        metrics=best_metrics.to_dict(),
        n_train_rows=int(len(X)),
        n_train_decisions=len(cases),
        version=version,
    )
    _persist(db, user_id, trained)
    return trained


def _persist(db: Session, user_id: str, trained: TrainedModel) -> ModelArtifact:
    db.execute(
        update(ModelArtifact)
        .where(ModelArtifact.user_id == user_id)
        .values(is_active=0)
    )
    artifact = ModelArtifact(
        user_id=user_id,
        kind=trained.kind,
        version=trained.version,
        is_active=1,
        payload=pickle.dumps(trained.estimator),
        metrics=trained.metrics,
        feature_names=trained.feature_names,
        importances=trained.importances,
        n_train_rows=trained.n_train_rows,
        n_train_decisions=trained.n_train_decisions,
        cv_accuracy=float(trained.metrics.get("accuracy", 0.0)),
        cv_brier=float(trained.metrics.get("brier", 1.0)),
        trained_at=datetime.now(timezone.utc),
    )
    db.add(artifact)
    db.flush()
    log.info(
        "persisted %s model v%s for user=%s (%d decisions)",
        trained.kind, trained.version, user_id, trained.n_train_decisions,
    )
    return artifact


def load_active_model(db: Session, user_id: str) -> TrainedModel | None:
    """Load the current model, tolerating an unusable pickle."""
    artifact = db.execute(
        select(ModelArtifact)
        .where(ModelArtifact.user_id == user_id, ModelArtifact.is_active == 1)
        .order_by(ModelArtifact.trained_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if artifact is None or not artifact.payload:
        return None

    try:
        estimator = pickle.loads(artifact.payload)
    except Exception as exc:
        # A library upgrade can invalidate a pickle. That must degrade the
        # prediction, never break it: the other layers still work.
        log.warning("could not load model artifact %s: %s", artifact.id, exc)
        return None

    if list(artifact.feature_names or []) != list(FEATURE_NAMES):
        log.warning(
            "feature schema changed since model %s was trained; ignoring it",
            artifact.version,
        )
        return None

    return TrainedModel(
        kind=artifact.kind,
        estimator=estimator,
        feature_names=list(artifact.feature_names or []),
        importances=dict(artifact.importances or {}),
        metrics=dict(artifact.metrics or {}),
        n_train_rows=artifact.n_train_rows,
        n_train_decisions=artifact.n_train_decisions,
        version=artifact.version,
    )
