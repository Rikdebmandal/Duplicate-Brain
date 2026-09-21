"""Compare prediction approaches on a held-out block of a person's history.

    python -m scripts.evaluate --email demo@example.com
    python -m scripts.evaluate --email demo@example.com --with-llm

The ``--with-llm`` arm is separated from the API endpoint on purpose: it makes
one model call per test decision, costs money, and is the only arm whose result
is not reproducible run to run.
"""
from __future__ import annotations

import argparse
import json

from sqlalchemy import select

from app.database.session import session_scope
from app.logging_conf import configure_logging
from app.ml.evaluation import run_evaluation
from app.models import User
from app.services.llm.client import get_llm_client
from app.services.profile.service import TrainingCase, load_training_cases

METRIC_COLUMNS = [
    ("accuracy", "acc", "{:.3f}"),
    ("brier", "brier", "{:.3f}"),
    ("log_loss", "logloss", "{:.3f}"),
    ("precision", "prec", "{:.3f}"),
    ("recall", "recall", "{:.3f}"),
    ("f1", "f1", "{:.3f}"),
    ("roc_auc", "auc", "{:.3f}"),
    ("ece", "ece", "{:.3f}"),
]


def make_llm_predictor(traits_source: dict):
    """Build a predictor callable for the LLM arm, or None when unavailable."""
    client = get_llm_client()
    if not client.available:
        print(f"  (LLM arm skipped: {client.reason_unavailable})")
        return None

    def predict(case: TrainingCase, traits: dict, retriever) -> list[float] | None:
        labels = [s.label for s in case.stances]
        neighbours = []
        # Reuse the evaluation retriever's own training-block-only case list, so
        # the LLM arm sees exactly the same evidence as the other arms.
        for other in retriever.cases[:5]:
            neighbours.append(
                {
                    "similarity": 0.5,
                    "scenario_text": other.scenario_text,
                    "chosen_option": other.stances[other.chosen_index].label,
                    "reason": other.reason,
                    "occurred_at": other.occurred_at.isoformat(),
                    "importance": other.importance,
                }
            )
        result = client.predict(
            scenario=case.scenario_text,
            options=labels,
            factors=case.factors,
            traits=traits,
            trait_confidences={k: 0.5 for k in traits},
            neighbours=neighbours,
            patterns=[],
            structured_probabilities={label: 1.0 / len(labels) for label in labels},
            decision_count=len(retriever.cases),
        )
        if not result.available:
            return None
        return [result.probabilities[label] for label in labels]

    return predict


def render(results: dict) -> None:
    header = f"{'approach':<12}" + "".join(f"{short:>9}" for _, short, _ in METRIC_COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in results.items():
        row = f"{name:<12}"
        for key, _, fmt in METRIC_COLUMNS:
            value = metrics.get(key)
            row += f"{'n/a':>9}" if value is None else f"{fmt.format(value):>9}"
        print(row)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--with-llm", action="store_true", help="include the LLM arm")
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = parser.parse_args()

    with session_scope() as db:
        user = db.execute(
            select(User).where(User.email == args.email.lower())
        ).scalar_one_or_none()
        if user is None:
            raise SystemExit(f"No account found for {args.email}")

        cases = load_training_cases(db, user.id)
        llm_predictor = make_llm_predictor({}) if args.with_llm else None

        result = run_evaluation(
            cases,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            llm_predictor=llm_predictor,
        )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if result.get("status") != "ok":
        print(result.get("message", "evaluation could not run"))
        return

    split = result["split"]
    print(f"\nDecisions: {result['n_decisions']}")
    print(
        f"Split (time-ordered): train {split['train']['n']} / "
        f"validation {split['validation']['n']} / test {split['test']['n']}"
    )
    print(f"Base rate of taking the active option: {result['approach_base_rate']:.0%}")
    print(f"Fitted parameters: {result['fitted_params']}")

    render(result["results"])

    print(f"\nBest by Brier score: {result['best_by_brier']}")
    print(f"\n{result['note']}\n")

    by_category = result["results"][result["best_by_brier"]].get("by_category") or {}
    if by_category:
        print("Best model, by scenario category:")
        for category, metrics in sorted(by_category.items()):
            print(
                f"  {category:<14} n={metrics['n']:<3} "
                f"acc={metrics['accuracy']:.3f} brier={metrics['brier']:.3f}"
            )
        print()


if __name__ == "__main__":
    main()
