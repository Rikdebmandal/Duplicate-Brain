"""The prediction pipeline.

    parse -> profile -> [utility | statistical | retrieval | LLM] -> fuse
          -> calibrate -> explain -> persist

Each layer is optional except the utility model. Missing layers do not break
the prediction; their weight is redistributed, the confidence score drops, and
the response says which layers ran. That is what lets the same code path serve
a brand-new user with three decisions and no API key, and an established one
with two hundred decisions and every layer live.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.errors import NotFoundError
from app.logging_conf import get_logger
from app.ml import trainer
from app.models import MemoryItem, Prediction, PredictionOption, Scenario, User
from app.services.decision_engine.features import build_features
from app.services.decision_engine.parser import (
    OptionStance,
    ParsedScenario,
    parse_scenario,
    resolve_stances,
)
from app.services.decision_engine.taxonomy import FACTOR_BY_KEY, FACTOR_KEYS
from app.services.decision_engine.utility import UtilityModel, softmax
from app.services.embeddings.store import Neighbour, VectorStore
from app.services.explainability import service as explain
from app.services.llm.client import get_llm_client
from app.services.prediction import fusion, retrieval
from app.services.profile.service import ProfileService, load_training_cases

log = get_logger(__name__)

#: Base weights before evidence scaling. The utility model is the floor: it is
#: always available and always contributes.
BASE_WEIGHTS = {
    "profile": 1.00,
    "statistical": 1.35,
    "retrieval": 1.10,
    "llm": 0.85,
}
#: Decisions at which the statistical layer reaches full weight.
ML_FULL_WEIGHT_AT = 40


@dataclass
class PredictionRequest:
    scenario: str
    options: list[str]
    context: dict = field(default_factory=dict)
    category: str = "general"
    factor_overrides: dict[str, float] | None = None
    use_llm: bool = True
    top_k: int = settings.retrieval_top_k
    persist: bool = True
    parent_prediction_id: str | None = None
    is_counterfactual: bool = False


class PredictionEngine:
    """Runs the pipeline for one user."""

    def __init__(self, db: Session, user: User) -> None:
        self.db = db
        self.user = user
        self.profiles = ProfileService(db)
        self.store = VectorStore(db)

    # -- public API ----------------------------------------------------------
    def predict(self, request: PredictionRequest) -> dict:
        if len(request.options) < 2:
            raise ValueError("a decision needs at least two options")

        parsed = self._parse(request)
        traits = self.profiles.traits_dict(self.user.id)
        params = self.profiles.model_params(self.user.id)
        model = UtilityModel(traits, params)

        cases = load_training_cases(self.db, self.user.id)
        decision_count = len(cases)

        layers: list[fusion.LayerOutput] = []

        # -- layer 1: the interpretable utility model ------------------------
        profile_probs, scored_options = model.probabilities(parsed.factors, parsed.options)
        layers.append(
            fusion.LayerOutput(
                name="profile",
                probabilities=profile_probs,
                weight=BASE_WEIGHTS["profile"],
                available=True,
                detail={
                    "temperature": params.temperature,
                    "status_quo_bias": params.status_quo_bias,
                    "description": "Trait-weighted utility model over the extracted factors.",
                },
            )
        )

        # -- layer 2: the statistical model -----------------------------------
        trained = trainer.load_active_model(self.db, self.user.id)
        feature_rows = build_features(parsed.factors, parsed.options, model)
        feature_values = [row.values for row in feature_rows]
        if trained is not None:
            ml_probs = trained.score_options(feature_values)
            quality = max(0.15, min(1.0, 1.0 - float(trained.metrics.get("brier", 0.5))))
            ramp = min(1.0, decision_count / ML_FULL_WEIGHT_AT)
            layers.append(
                fusion.LayerOutput(
                    name="statistical",
                    probabilities=ml_probs,
                    weight=BASE_WEIGHTS["statistical"] * quality * ramp,
                    available=True,
                    detail={
                        "kind": trained.kind,
                        "version": trained.version,
                        "cv_accuracy": trained.metrics.get("accuracy"),
                        "cv_brier": trained.metrics.get("brier"),
                        "trained_on_decisions": trained.n_train_decisions,
                        "description": "Conditional-logit classifier over option features.",
                    },
                )
            )
        else:
            layers.append(
                fusion.LayerOutput(
                    name="statistical", probabilities=[], weight=0.0, available=False,
                    detail={
                        "reason": f"needs at least {settings.min_decisions_for_ml} decisions; "
                                  f"{decision_count} on record",
                    },
                )
            )

        # -- layer 3: retrieval over the person's own history -----------------
        neighbours = self.store.similar_decisions(
            user_id=self.user.id,
            query_text=request.scenario,
            query_factors=parsed.factors,
            k=request.top_k,
        )
        retrieval_result = retrieval.predict_from_neighbours(neighbours, parsed.options)
        layers.append(
            fusion.LayerOutput(
                name="retrieval",
                probabilities=retrieval_result.probabilities,
                weight=BASE_WEIGHTS["retrieval"] * retrieval_result.support,
                available=retrieval_result.support > 0 and bool(neighbours),
                detail={
                    "n_neighbours": retrieval_result.n_neighbours,
                    "support": round(retrieval_result.support, 4),
                    "votes": [v.to_dict() for v in retrieval_result.votes[:6]],
                    "description": "Case-based vote from the most similar recorded decisions.",
                },
            )
        )

        # -- layer 4: LLM reasoning -------------------------------------------
        llm_layer, llm_output = self._llm_layer(
            request, parsed, traits, profile_probs, neighbours, decision_count
        )
        layers.append(llm_layer)

        # -- fusion -------------------------------------------------------------
        trait_rows = {r.trait_key: r for r in self.profiles.trait_rows(self.user.id)}
        trait_confidences = {k: float(v.confidence) for k, v in trait_rows.items()}
        result = fusion.fuse(
            layers,
            decision_count=decision_count,
            factor_coverage=parsed.coverage,
            trait_confidence=self._weighted_trait_confidence(
                scored_options, trait_confidences
            ),
            retrieval_support=retrieval_result.support,
            calibration_temperature=params.calibration_temperature,
        )
        if result is None:  # pragma: no cover - profile layer is never absent
            raise RuntimeError("no prediction layer produced an answer")

        probabilities = result.probabilities
        predicted_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        predicted_option = request.options[predicted_index]

        explanation = self._explain(
            parsed=parsed,
            scored_options=scored_options,
            predicted_index=predicted_index,
            probabilities=probabilities,
            neighbours=neighbours,
            result=result,
            traits=traits,
            trait_confidences=trait_confidences,
            trained=trained,
            feature_values=feature_values,
            llm_output=llm_output,
            decision_count=decision_count,
        )

        payload = {
            "scenario": request.scenario,
            "category": request.category,
            "options": [
                {
                    "label": label,
                    "probability": round(probabilities[i], 4),
                    "utility": round(scored_options[i].utility, 4),
                    "stance": parsed.options[i].to_dict(),
                    "contributions": [c.to_dict() for c in scored_options[i].contributions],
                }
                for i, label in enumerate(request.options)
            ],
            "predicted_option": predicted_option,
            "predicted_probability": round(probabilities[predicted_index], 4),
            "confidence_label": result.confidence_label,
            "confidence_score": round(result.confidence_score, 4),
            "confidence_inputs": result.confidence_inputs,
            "factors": {k: round(v, 4) for k, v in parsed.factors.items()},
            "factor_source": explanation["factor_source"],
            "layers": [layer.to_dict() for layer in layers],
            "similar_decisions": [n.to_dict() for n in neighbours],
            "explanation": explanation,
            "evidence_count": decision_count,
            "model_version": trained.version if trained else "profile-only",
            "is_counterfactual": request.is_counterfactual,
            "overrides": request.factor_overrides or {},
            "disclaimer": (
                "A behavioural estimate from recorded decisions - not a simulation of a "
                "person's mind, and not a statement of what they will do."
            ),
        }

        if request.persist:
            prediction = self._persist(request, parsed, payload, result, neighbours)
            payload["id"] = prediction.id
            payload["created_at"] = prediction.created_at.isoformat()
            payload["scenario_id"] = prediction.scenario_id

        return payload

    # -- counterfactuals -------------------------------------------------------
    def counterfactual(
        self,
        prediction_id: str,
        overrides: dict[str, float],
        *,
        persist: bool = True,
    ) -> dict:
        """Re-run an existing prediction with some factors moved.

        The scenario text, option set and every other factor are held fixed, so
        the difference between the two distributions is attributable to the
        overrides alone. The LLM layer is deliberately skipped: it would
        re-interpret the prose and reintroduce variation that has nothing to do
        with the slider that moved.
        """
        original = self._load_prediction(prediction_id)
        scenario = self.db.get(Scenario, original.scenario_id)
        if scenario is None:
            raise NotFoundError("The scenario for that prediction no longer exists.")

        base_factors = dict(original.factors or {})
        merged = dict(base_factors)
        for key, value in (overrides or {}).items():
            if key in FACTOR_KEYS and value is not None:
                merged[key] = float(min(1.0, max(0.0, value)))

        request = PredictionRequest(
            scenario=scenario.text,
            options=[o.label for o in sorted(original.options, key=lambda o: o.position)],
            context=dict(scenario.context or {}),
            category=scenario.category,
            factor_overrides=merged,
            use_llm=False,
            persist=persist,
            parent_prediction_id=original.id,
            is_counterfactual=True,
        )
        modified = self.predict(request)

        original_probs = {
            o.label: round(o.probability, 4)
            for o in sorted(original.options, key=lambda o: o.position)
        }
        modified_probs = {o["label"]: o["probability"] for o in modified["options"]}
        changed = [
            {
                "factor": key,
                "label": FACTOR_BY_KEY[key].label,
                "from": round(base_factors.get(key, 0.0), 4),
                "to": round(value, 4),
            }
            for key, value in (overrides or {}).items()
            if key in FACTOR_BY_KEY
        ]

        return {
            "original_prediction_id": original.id,
            "changed_factors": changed,
            "original": {
                "probabilities": original_probs,
                "predicted_option": original.predicted_option,
            },
            "modified": {
                "probabilities": modified_probs,
                "predicted_option": modified["predicted_option"],
            },
            "flipped": modified["predicted_option"] != original.predicted_option,
            "deltas": {
                label: round(modified_probs.get(label, 0.0) - original_probs.get(label, 0.0), 4)
                for label in original_probs
            },
            "prediction": modified,
        }

    def landscape(
        self,
        prediction_id: str,
        factor_keys: Sequence[str] | None = None,
        steps: int = 5,
    ) -> dict:
        """Sweep each factor across its range to map the decision boundary.

        This is what turns a single prediction into an actual answer to "what
        would have to be true for me to choose differently" - it reports, per
        factor, the value at which the predicted option flips.
        """
        original = self._load_prediction(prediction_id)
        scenario = self.db.get(Scenario, original.scenario_id)
        if scenario is None:
            raise NotFoundError("The scenario for that prediction no longer exists.")

        options = [o.label for o in sorted(original.options, key=lambda o: o.position)]
        base_factors = dict(original.factors or {})
        keys = list(factor_keys) if factor_keys else [
            k for k in FACTOR_KEYS if FACTOR_BY_KEY[k].sign_under_approach != 0
        ]
        grid = [i / (steps - 1) for i in range(steps)] if steps > 1 else [0.5]

        traits = self.profiles.traits_dict(self.user.id)
        params = self.profiles.model_params(self.user.id)
        model = UtilityModel(traits, params)
        stances = resolve_stances(options)
        trained = trainer.load_active_model(self.db, self.user.id)

        sweeps: list[dict] = []
        for key in keys:
            if key not in FACTOR_BY_KEY:
                continue
            points: list[dict] = []
            for value in grid:
                factors = dict(base_factors)
                factors[key] = value
                probabilities = self._fast_probabilities(factors, stances, model, trained)
                best = max(range(len(options)), key=lambda i: probabilities[i])
                points.append(
                    {
                        "value": round(value, 3),
                        "probabilities": {
                            label: round(probabilities[i], 4)
                            for i, label in enumerate(options)
                        },
                        "predicted_option": options[best],
                    }
                )

            flip = None
            for previous, current in zip(points, points[1:], strict=False):
                if previous["predicted_option"] != current["predicted_option"]:
                    flip = {
                        "between": [previous["value"], current["value"]],
                        "from": previous["predicted_option"],
                        "to": current["predicted_option"],
                    }
                    break

            sweeps.append(
                {
                    "factor": key,
                    "label": FACTOR_BY_KEY[key].label,
                    "current_value": round(base_factors.get(key, 0.0), 4),
                    "points": points,
                    "flip_point": flip,
                    "sensitivity": round(
                        max(
                            abs(
                                points[-1]["probabilities"][options[0]]
                                - points[0]["probabilities"][options[0]]
                            ),
                            0.0,
                        ),
                        4,
                    ),
                }
            )

        sweeps.sort(key=lambda s: -s["sensitivity"])
        return {
            "prediction_id": original.id,
            "options": options,
            "current_prediction": original.predicted_option,
            "sweeps": sweeps,
            "note": (
                "Sweeps use the profile and statistical layers only. Retrieval and "
                "LLM reasoning are held out so that the change shown is attributable "
                "to the moved factor alone."
            ),
        }

    # -- internals --------------------------------------------------------------
    def _parse(self, request: PredictionRequest) -> ParsedScenario:
        context_text = " ".join(
            str(v) for v in (request.context or {}).values() if isinstance(v, str | int | float)
        )
        parsed = parse_scenario(
            request.scenario,
            request.options,
            context_text=context_text,
            factor_overrides=request.factor_overrides,
        )
        # Only consult the LLM for factor refinement on a fresh prediction where
        # the person has opted in, and never on a counterfactual re-run.
        if (
            request.use_llm
            and not request.is_counterfactual
            and not request.factor_overrides
            and self.user.allow_llm_processing
        ):
            client = get_llm_client()
            refined = client.refine_factors(request.scenario, request.options, parsed.factors)
            if refined:
                parsed.factors = {**parsed.factors, **refined["factors"]}
                parsed.coverage = max(parsed.coverage, 0.6)
                parsed.llm_factor_notes = refined.get("notes", {})
                parsed.factor_source = "llm"
        return parsed

    def _llm_layer(
        self,
        request: PredictionRequest,
        parsed: ParsedScenario,
        traits: dict[str, float],
        profile_probs: Sequence[float],
        neighbours: Sequence[Neighbour],
        decision_count: int,
    ) -> tuple[fusion.LayerOutput, object | None]:
        if request.is_counterfactual or not request.use_llm:
            return (
                fusion.LayerOutput(
                    "llm", [], 0.0, available=False,
                    detail={"reason": "skipped for a deterministic re-run"},
                ),
                None,
            )
        if not self.user.allow_llm_processing:
            return (
                fusion.LayerOutput(
                    "llm", [], 0.0, available=False,
                    detail={"reason": "the account has not opted in to LLM processing"},
                ),
                None,
            )

        client = get_llm_client()
        if not client.available:
            return (
                fusion.LayerOutput(
                    "llm", [], 0.0, available=False,
                    detail={"reason": client.reason_unavailable},
                ),
                None,
            )

        trait_rows = {r.trait_key: r for r in self.profiles.trait_rows(self.user.id)}
        patterns = [
            m.content
            for m in self.db.execute(
                select(MemoryItem).where(
                    MemoryItem.user_id == self.user.id,
                    MemoryItem.kind == "pattern",
                )
            ).scalars().all()
        ]

        output = client.predict(
            scenario=request.scenario,
            options=request.options,
            factors=parsed.factors,
            traits=traits,
            trait_confidences={k: float(v.confidence) for k, v in trait_rows.items()},
            neighbours=[n.to_dict() for n in neighbours],
            patterns=patterns,
            structured_probabilities={
                label: profile_probs[i] for i, label in enumerate(request.options)
            },
            decision_count=decision_count,
        )
        if not output.available:
            return (
                fusion.LayerOutput(
                    "llm", [], 0.0, available=False, detail={"reason": output.error}
                ),
                output,
            )

        return (
            fusion.LayerOutput(
                name="llm",
                probabilities=[output.probabilities[label] for label in request.options],
                weight=BASE_WEIGHTS["llm"],
                available=True,
                detail={
                    "model": output.model,
                    "self_reported_confidence": output.confidence,
                    "contextual_factors": output.contextual_factors,
                    "description": "Narrative reasoning over the same evidence.",
                },
            ),
            output,
        )

    def _fast_probabilities(
        self,
        factors: dict[str, float],
        stances: Sequence[OptionStance],
        model: UtilityModel,
        trained,
    ) -> list[float]:
        """Profile (+ statistical) probabilities without retrieval or the LLM."""
        utilities = [u.utility for u in model.score_options(factors, stances)]
        probabilities = softmax(utilities, model.params.temperature)
        if trained is None:
            return probabilities

        rows = [row.values for row in build_features(factors, stances, model)]
        ml_probs = trained.score_options(rows)
        pooled = fusion.log_linear_pool(
            [
                fusion.LayerOutput("profile", probabilities, BASE_WEIGHTS["profile"]),
                fusion.LayerOutput("statistical", ml_probs, BASE_WEIGHTS["statistical"]),
            ]
        )
        return fusion.apply_calibration(pooled, model.params.calibration_temperature)

    @staticmethod
    def _weighted_trait_confidence(
        scored_options: Sequence, trait_confidences: dict[str, float]
    ) -> float:
        """Confidence of the traits that actually mattered for this scenario."""
        factor_to_trait = {
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
        weighted = 0.0
        total = 0.0
        for option in scored_options:
            for contribution in option.contributions:
                trait_key = factor_to_trait.get(contribution.factor)
                if not trait_key:
                    continue
                weight = abs(contribution.value)
                weighted += weight * trait_confidences.get(trait_key, 0.0)
                total += weight
        if total <= 0:
            values = list(trait_confidences.values())
            return sum(values) / len(values) if values else 0.0
        return weighted / total

    def _explain(
        self,
        *,
        parsed: ParsedScenario,
        scored_options: Sequence,
        predicted_index: int,
        probabilities: Sequence[float],
        neighbours: Sequence[Neighbour],
        result: fusion.FusionResult,
        traits: dict[str, float],
        trait_confidences: dict[str, float],
        trained,
        feature_values: Sequence[Sequence[float]],
        llm_output,
        decision_count: int,
    ) -> dict:
        chosen = scored_options[predicted_index]
        factors = explain.factor_explanations(chosen.contributions, parsed.hits)

        patterns = [
            {"content": m.content, "confidence": m.confidence}
            for m in self.db.execute(
                select(MemoryItem).where(
                    MemoryItem.user_id == self.user.id,
                    MemoryItem.kind == "pattern",
                )
            ).scalars().all()
        ]

        layer_summary = [layer.to_dict() for layer in result.layers]
        summary = explain.natural_language_summary(
            predicted_label=chosen.label,
            probability=probabilities[predicted_index],
            confidence_label=result.confidence_label,
            factors=factors,
            neighbours=neighbours,
            decision_count=decision_count,
        )

        return {
            "summary": summary,
            "llm_reasoning": llm_output.reasoning if llm_output and llm_output.available else "",
            "llm_caveats": llm_output.caveats if llm_output and llm_output.available else [],
            "important_factors": [f.to_dict() for f in factors],
            "supporting": [f.to_dict() for f in factors if f.contribution > 0],
            "opposing": [f.to_dict() for f in factors if f.contribution < 0],
            "reasoning_trace": explain.build_reasoning_trace(
                predicted_label=chosen.label,
                probability=probabilities[predicted_index],
                factors=factors,
                neighbours=neighbours,
                layer_summary=layer_summary,
                confidence_label=result.confidence_label,
            ),
            "observed": explain.observed_statements(neighbours, patterns, decision_count),
            "inferred": explain.inferred_statements(
                traits, trait_confidences, chosen.contributions
            ),
            "ml_attribution": explain.ml_attribution(trained, feature_values, predicted_index),
            "factor_source": getattr(parsed, "factor_source", "lexical"),
            "factor_evidence": [h.to_dict() for h in parsed.hits[:40]],
            "emotions_detected": parsed.emotions,
            "limitations": explain.LIMITATIONS,
        }

    def _persist(
        self,
        request: PredictionRequest,
        parsed: ParsedScenario,
        payload: dict,
        result: fusion.FusionResult,
        neighbours: Sequence[Neighbour],
    ) -> Prediction:
        scenario = Scenario(
            user_id=self.user.id,
            text=request.scenario,
            context=request.context or {},
            category=request.category,
            factors=payload["factors"],
            factor_source=payload["factor_source"],
            coverage=parsed.coverage,
            factor_evidence={"hits": [h.to_dict() for h in parsed.hits[:60]]},
            emotions=parsed.emotions,
            is_hypothetical=True,
        )
        self.db.add(scenario)
        self.db.flush()
        self.store.index_scenario(scenario)

        prediction = Prediction(
            user_id=self.user.id,
            scenario_id=scenario.id,
            predicted_option=payload["predicted_option"],
            predicted_probability=payload["predicted_probability"],
            confidence_label=payload["confidence_label"],
            confidence_score=payload["confidence_score"],
            factors=payload["factors"],
            traits_snapshot=self.profiles.traits_dict(self.user.id),
            layer_outputs={
                "layers": payload["layers"],
                "agreement": round(result.agreement, 4),
                "confidence_inputs": result.confidence_inputs,
            },
            explanation=payload["explanation"],
            similar_decisions=[n.to_dict() for n in neighbours],
            model_version=payload["model_version"],
            evidence_count=payload["evidence_count"],
            is_counterfactual=request.is_counterfactual,
            parent_prediction_id=request.parent_prediction_id,
            overrides=request.factor_overrides or {},
        )
        self.db.add(prediction)
        self.db.flush()

        for index, option in enumerate(payload["options"]):
            self.db.add(
                PredictionOption(
                    prediction_id=prediction.id,
                    label=option["label"],
                    position=index,
                    probability=option["probability"],
                    utility=option["utility"],
                    stance=option["stance"],
                    contributions=option["contributions"],
                )
            )
        self.db.flush()
        return prediction

    def _load_prediction(self, prediction_id: str) -> Prediction:
        prediction = self.db.get(Prediction, prediction_id)
        if prediction is None or prediction.user_id != self.user.id:
            raise NotFoundError("Prediction not found.")
        return prediction


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
