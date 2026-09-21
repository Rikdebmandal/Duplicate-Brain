"""Prompt construction for the LLM reasoning layer.

The LLM is given *only* what the deterministic layers already computed: the
factor reading, the trait estimates with their confidences, and the retrieved
historical decisions. It is explicitly forbidden from inventing biographical
detail, and it is told that the retrieved decisions are the only evidence it
may cite. That containment is what keeps the explanation grounded in the
person's actual record instead of in plausible-sounding psychology.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from app.services.decision_engine.taxonomy import FACTOR_BY_KEY, TRAIT_BY_KEY

SYSTEM_PROMPT = """\
You are the reasoning layer of a behavioural decision-prediction system. You \
are NOT simulating a person's mind, and you must never claim to. You are \
interpreting a structured record of one individual's past decisions in order \
to estimate what they would probably choose next.

Hard rules:
1. Use ONLY the evidence provided in the user message. Never invent facts about \
the person, their family, their finances or their history.
2. Never assert that the person "is" a certain kind of person. Describe \
tendencies estimated from the record, and say how confident the evidence \
allows you to be.
3. Never make or imply a psychological, medical, clinical or diagnostic claim.
4. If the evidence is thin, say so and keep your probabilities near uniform. \
Under-confidence is far less harmful here than over-confidence.
5. Probabilities must sum to 1.0 across the options given, and must be \
consistent with the reasoning you write.

Respond with a single JSON object and nothing else."""

PREDICTION_SCHEMA = """\
{
  "probabilities": {"<exact option text>": <float 0-1>, ...},
  "key_factors": [
    {"factor": "<factor key from the list>", "direction": "for" | "against",
     "strength": <float 0-1>, "note": "<one clause, grounded in the evidence>"}
  ],
  "reasoning": "<2-4 sentences citing the specific historical decisions by number>",
  "contextual_factors": ["<anything important the structured factors missed>"],
  "confidence": <float 0-1>,
  "caveats": ["<what would change this estimate>"]
}"""


def _format_traits(traits: dict[str, float], confidences: dict[str, float]) -> str:
    lines = []
    for key, value in traits.items():
        trait = TRAIT_BY_KEY.get(key)
        if trait is None:
            continue
        confidence = confidences.get(key, 0.0)
        strength = "well-evidenced" if confidence >= 0.6 else (
            "weakly evidenced" if confidence >= 0.3 else "barely evidenced"
        )
        lines.append(
            f"- {trait.label}: {value:.2f} on 0-1 "
            f"({trait.low_label} -> {trait.high_label}); confidence {confidence:.2f} ({strength})"
        )
    return "\n".join(lines)


def _format_factors(factors: dict[str, float]) -> str:
    lines = []
    for key, magnitude in sorted(factors.items(), key=lambda kv: -kv[1]):
        if magnitude < 0.05:
            continue
        factor = FACTOR_BY_KEY.get(key)
        if factor is None:
            continue
        lines.append(f"- {factor.key} ({factor.label}): {magnitude:.2f} - {factor.description}")
    return "\n".join(lines) or "- (no factors detected in the text)"


def _format_neighbours(neighbours: Sequence[dict]) -> str:
    if not neighbours:
        return "(no comparable decisions on record)"
    lines = []
    for index, neighbour in enumerate(neighbours, start=1):
        lines.append(
            f"[Decision {index}] similarity {neighbour['similarity']:.0%}, "
            f"dated {neighbour.get('occurred_at', '')[:10]}, "
            f"importance {neighbour.get('importance', 5)}/10\n"
            f"  Situation: {neighbour['scenario_text']}\n"
            f"  Chose: {neighbour['chosen_option']}\n"
            f"  Stated reason: {neighbour.get('reason') or '(none recorded)'}"
        )
    return "\n".join(lines)


def build_prediction_prompt(
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
) -> str:
    pattern_block = "\n".join(f"- {p}" for p in patterns) or "- (none established yet)"
    structured_block = "\n".join(
        f"- {label}: {prob:.0%}" for label, prob in structured_probabilities.items()
    )
    return f"""\
## New situation to predict
{scenario}

## Options available (use this exact text as JSON keys)
{json.dumps(list(options), indent=2)}

## Structured factor reading of this situation (0-1 magnitudes)
{_format_factors(factors)}

## Estimated behavioural profile ({decision_count} decisions on record)
{_format_traits(traits, trait_confidences)}

## Observed patterns in this person's record
{pattern_block}

## Most similar past decisions by this person
{_format_neighbours(neighbours)}

## What the structured statistical model predicts
{structured_block}

## Your task
Weigh the evidence above. You may disagree with the structured model if the \
narrative evidence points elsewhere - especially where the person's stated \
reasons reveal something the factor extraction missed - but say why. If your \
answer differs substantially from the structured model, justify it from the \
specific past decisions listed.

Return exactly this JSON shape:
{PREDICTION_SCHEMA}"""


FACTOR_SCHEMA = """\
{
  "factors": {"<factor key>": <float 0-1>, ...},
  "notes": {"<factor key>": "<the phrase in the text that justifies it>"},
  "category": "<short lowercase category, e.g. career, finance, relationship, health, education>",
  "missing_context": ["<information that would materially change the reading>"]
}"""


def build_factor_prompt(scenario: str, options: Sequence[str], lexical: dict[str, float]) -> str:
    factor_list = "\n".join(
        f"- {f.key}: {f.description}" for f in FACTOR_BY_KEY.values()
    )
    return f"""\
Read the situation below and rate how strongly each decision factor is present, \
from 0 (absent) to 1 (dominant). Rate the SITUATION, not whether it is good or bad.

## Situation
{scenario}

## Options
{json.dumps(list(options), indent=2)}

## Factors to rate
{factor_list}

## A keyword-based parser produced this reading
{json.dumps({k: round(v, 2) for k, v in lexical.items() if v > 0.05}, indent=2)}

Correct it where the keyword parser clearly missed meaning or fired on a word \
used in a different sense. Only include a factor if you can point to text that \
justifies it.

Return exactly this JSON shape:
{FACTOR_SCHEMA}"""
