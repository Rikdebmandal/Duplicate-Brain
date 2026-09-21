"""Anthropic client wrapper for the reasoning layer.

Every method here is allowed to fail. The prediction pipeline treats the LLM as
an *optional contributor*: if there is no API key, no network, or a malformed
response, the layer reports itself unavailable and the fusion step redistributes
its weight to the layers that did produce an answer. Nothing in the system
blocks on it.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.config import settings
from app.logging_conf import get_logger
from app.services.decision_engine.taxonomy import FACTOR_KEYS
from app.services.llm import prompts

log = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LLMPrediction:
    """Structured output of the reasoning layer."""

    probabilities: dict[str, float] = field(default_factory=dict)
    key_factors: list[dict] = field(default_factory=list)
    reasoning: str = ""
    contextual_factors: list[str] = field(default_factory=list)
    confidence: float = 0.0
    caveats: list[str] = field(default_factory=list)
    available: bool = False
    error: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "model": self.model,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "key_factors": self.key_factors,
            "reasoning": self.reasoning,
            "contextual_factors": self.contextual_factors,
            "confidence": round(self.confidence, 4),
            "caveats": self.caveats,
            "error": self.error,
        }


def _extract_json(raw: str) -> dict | None:
    """Pull the first JSON object out of a response, tolerating prose around it."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMClient:
    """Thin wrapper that is always safe to construct."""

    def __init__(self) -> None:
        self._client = None
        self.model = settings.llm_model
        self.reason_unavailable = ""

        if not settings.llm_enabled:
            self.reason_unavailable = "LLM layer disabled by configuration"
            return
        if not settings.anthropic_api_key:
            self.reason_unavailable = "no ANTHROPIC_API_KEY configured"
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - import/config path
            self.reason_unavailable = f"anthropic client unavailable: {exc}"
            log.warning("LLM layer disabled: %s", self.reason_unavailable)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _complete(self, system: str, prompt: str) -> str | None:
        if not self.available:
            return None
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=settings.llm_max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
        except Exception as exc:
            log.warning("LLM request failed: %s", exc)
            return None

    # -- prediction ---------------------------------------------------------
    def predict(
        self,
        *,
        scenario: str,
        options: Sequence[str],
        factors: dict[str, float],
        traits: dict[str, float],
        trait_confidences: dict[str, float],
        neighbours: Sequence[dict],
        patterns: Sequence[str],
        structured_probabilities: dict[str, float],
        decision_count: int,
    ) -> LLMPrediction:
        if not self.available:
            return LLMPrediction(available=False, error=self.reason_unavailable)

        prompt = prompts.build_prediction_prompt(
            scenario=scenario,
            options=options,
            factors=factors,
            traits=traits,
            trait_confidences=trait_confidences,
            neighbours=neighbours,
            patterns=patterns,
            structured_probabilities=structured_probabilities,
            decision_count=decision_count,
        )
        raw = self._complete(prompts.SYSTEM_PROMPT, prompt)
        payload = _extract_json(raw or "")
        if payload is None:
            return LLMPrediction(
                available=False, model=self.model,
                error="LLM returned no parseable JSON",
            )

        probabilities = _normalise_probabilities(payload.get("probabilities"), options)
        if not probabilities:
            return LLMPrediction(
                available=False, model=self.model,
                error="LLM probabilities did not match the option set",
            )

        return LLMPrediction(
            probabilities=probabilities,
            key_factors=[f for f in payload.get("key_factors", []) if isinstance(f, dict)][:8],
            reasoning=str(payload.get("reasoning", ""))[:2000],
            contextual_factors=[str(c)[:300] for c in payload.get("contextual_factors", [])][:6],
            confidence=float(min(1.0, max(0.0, float(payload.get("confidence", 0.5) or 0.0)))),
            caveats=[str(c)[:300] for c in payload.get("caveats", [])][:6],
            available=True,
            model=self.model,
        )

    # -- factor refinement ---------------------------------------------------
    def refine_factors(
        self, scenario: str, options: Sequence[str], lexical: dict[str, float]
    ) -> dict | None:
        """Ask the model to correct the keyword parser. Returns None on failure."""
        if not self.available:
            return None
        raw = self._complete(
            prompts.SYSTEM_PROMPT,
            prompts.build_factor_prompt(scenario, options, lexical),
        )
        payload = _extract_json(raw or "")
        if payload is None:
            return None

        factors = {}
        for key, value in (payload.get("factors") or {}).items():
            if key in FACTOR_KEYS:
                try:
                    factors[key] = float(min(1.0, max(0.0, float(value))))
                except (TypeError, ValueError):
                    continue
        if not factors:
            return None
        return {
            "factors": factors,
            "notes": {
                k: str(v)[:300]
                for k, v in (payload.get("notes") or {}).items()
                if k in FACTOR_KEYS
            },
            "category": str(payload.get("category", "general"))[:64].lower(),
            "missing_context": [str(c)[:300] for c in payload.get("missing_context", [])][:5],
        }


def _normalise_probabilities(
    raw: object, options: Sequence[str]
) -> dict[str, float]:
    """Map the model's keys back onto the exact option strings and renormalise."""
    if not isinstance(raw, dict):
        return {}

    lookup = {str(o).strip().lower(): o for o in options}
    matched: dict[str, float] = {}
    for key, value in raw.items():
        option = lookup.get(str(key).strip().lower())
        if option is None:
            # Tolerate light paraphrasing of the option text.
            for candidate_key, candidate in lookup.items():
                if candidate_key in str(key).lower() or str(key).lower() in candidate_key:
                    option = candidate
                    break
        if option is None:
            continue
        try:
            matched[option] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue

    if len(matched) < len(options):
        for option in options:
            matched.setdefault(option, 0.0)
    total = sum(matched.values())
    if total <= 0:
        return {}
    return {option: matched[option] / total for option in options}


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    """Test hook: forces the client to be rebuilt from current settings."""
    global _client
    _client = None
