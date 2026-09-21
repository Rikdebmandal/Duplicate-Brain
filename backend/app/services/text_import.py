"""Reading decisions and values out of unstructured personal text.

Journals, notes and interview transcripts are the richest available source of
decision evidence, and the least structured. This module finds candidate
decisions in prose and proposes them - it never commits them. The person
confirms, edits or discards each candidate, because a mis-parsed sentence
silently entering the training history would corrupt every downstream estimate
with no audit trail.

Everything here is deterministic and offline. If the account has opted in to
LLM processing, the API layer can additionally ask the model to improve the
factor reading of a confirmed candidate.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.services.decision_engine.parser import (
    classify_option,
    extract_emotions,
    extract_factors,
    tokenize,
)

#: First-person past-tense markers that a choice was actually made.
_DECISION_MARKERS = [
    "i decided", "i chose", "i went with", "i picked", "i opted",
    "i accepted", "i took the", "i said yes", "i agreed to",
    "i rejected", "i declined", "i turned down", "i said no", "i refused",
    "i quit", "i stayed", "i left", "i walked away", "i backed out",
    "i ended up", "i settled on", "i committed to", "i passed on",
    "we decided", "i had to choose", "i gave up",
]

#: Phrases that introduce the alternative that was not taken.
_ALTERNATIVE_MARKERS = [
    "instead of", "rather than", "over ", "as opposed to", "in place of",
    "even though i could have", "could have", "the other option",
    "the alternative was", "or i could",
]

#: Phrases that introduce the person's reasoning.
_REASON_MARKERS = [
    "because", "since", "as i", "so that", "in order to", "the reason",
    "mainly", "primarily", "what mattered", "i felt", "i thought",
    "i wanted", "i needed", "it came down to",
]

_VALUE_MARKERS = [
    "i value", "i believe", "matters to me", "important to me", "i care about",
    "i would never", "i always", "i never", "my priority", "i refuse to",
    "what matters most", "i can't compromise", "i cannot compromise",
]

_GOAL_MARKERS = [
    "i want to", "i plan to", "my goal", "i hope to", "i am working towards",
    "i'm working towards", "i intend to", "i aim to", "eventually i want",
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")

#: Options assigned when the text describes a choice without naming the branches.
_DEFAULT_OPTIONS = ["Took the action described", "Did not take it"]


@dataclass
class CandidateDecision:
    """A proposed decision, awaiting confirmation by the person."""

    situation: str
    options: list[str]
    decision: str
    reason: str = ""
    confidence: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)
    emotions: dict[str, float] = field(default_factory=dict)
    source_excerpt: str = ""
    matched_marker: str = ""
    needs_review: bool = True

    def to_dict(self) -> dict:
        return {
            "situation": self.situation,
            "options": self.options,
            "decision": self.decision,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "factors": {k: round(v, 3) for k, v in self.factors.items() if v > 0.05},
            "emotions": self.emotions,
            "source_excerpt": self.source_excerpt,
            "matched_marker": self.matched_marker,
            "needs_review": self.needs_review,
        }


@dataclass
class TextAnalysis:
    candidates: list[CandidateDecision] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    emotions: dict[str, float] = field(default_factory=dict)
    factors: dict[str, float] = field(default_factory=dict)
    recurring_terms: list[dict] = field(default_factory=list)
    word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "values": self.values,
            "goals": self.goals,
            "emotions": self.emotions,
            "factors": {k: round(v, 3) for k, v in self.factors.items() if v > 0.05},
            "recurring_terms": self.recurring_terms,
            "word_count": self.word_count,
            "note": (
                "Candidates are proposals extracted from the text. Nothing is added "
                "to the decision history until it is confirmed."
            ),
        }


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def _find_marker(sentence: str, markers: Sequence[str]) -> str | None:
    lowered = sentence.lower()
    for marker in markers:
        if marker in lowered:
            return marker
    return None


def _split_reason(sentence: str) -> tuple[str, str]:
    """Separate the action clause from the justification clause."""
    lowered = sentence.lower()
    for marker in _REASON_MARKERS:
        index = lowered.find(marker)
        if index > 0:
            return sentence[:index].strip(" ,;-"), sentence[index:].strip(" ,;-")
    return sentence.strip(), ""


def _derive_options(sentence: str, action: str) -> tuple[list[str], str]:
    """Infer the option set, using an explicit alternative where one is stated."""
    lowered = sentence.lower()
    for marker in _ALTERNATIVE_MARKERS:
        index = lowered.find(marker)
        if index <= 0:
            continue
        alternative = sentence[index + len(marker):].strip(" ,.;-")
        alternative = _SENTENCE_SPLIT.split(alternative)[0].strip()
        if 2 <= len(alternative.split()) <= 14:
            chosen = action.strip(" ,.;-") or _DEFAULT_OPTIONS[0]
            return [chosen, alternative], chosen

    chosen = action.strip(" ,.;-")
    if not chosen or len(chosen.split()) < 2:
        return list(_DEFAULT_OPTIONS), _DEFAULT_OPTIONS[0]

    stance = classify_option(chosen)
    counterpart = (
        "Did not do it" if stance.approach >= 0.5 else "Did it after all"
    )
    return [chosen, counterpart], chosen


def analyse_text(text: str, *, context_window: int = 1) -> TextAnalysis:
    """Extract candidate decisions, stated values, goals and emotional tone."""
    analysis = TextAnalysis(word_count=len(tokenize(text)))
    if not text or not text.strip():
        return analysis

    sentences = _sentences(text)
    overall = extract_factors(text)
    analysis.factors = overall.factors
    analysis.emotions = extract_emotions(text)

    for index, sentence in enumerate(sentences):
        marker = _find_marker(sentence, _DECISION_MARKERS)
        if marker:
            analysis.candidates.append(
                _build_candidate(sentences, index, sentence, marker, context_window)
            )
            continue
        if _find_marker(sentence, _VALUE_MARKERS):
            analysis.values.append(sentence.strip()[:400])
        elif _find_marker(sentence, _GOAL_MARKERS):
            analysis.goals.append(sentence.strip()[:400])

    analysis.recurring_terms = _recurring_terms(text)
    return analysis


def _build_candidate(
    sentences: Sequence[str],
    index: int,
    sentence: str,
    marker: str,
    context_window: int,
) -> CandidateDecision:
    lower = max(0, index - context_window)
    upper = min(len(sentences), index + context_window + 1)
    situation = " ".join(sentences[lower:upper]).strip()

    action, reason = _split_reason(sentence)
    options, chosen = _derive_options(sentence, action)
    extraction = extract_factors(situation)

    # Confidence in the *extraction*, not in the person. It is high when the
    # sentence states an alternative and a reason, low when we had to fall back
    # to a generic did/did-not option pair.
    confidence = 0.35
    if reason:
        confidence += 0.2
    if options != _DEFAULT_OPTIONS:
        confidence += 0.25
    if extraction.coverage > 0.2:
        confidence += 0.15

    return CandidateDecision(
        situation=situation[:1200],
        options=options,
        decision=chosen,
        reason=reason[:600],
        confidence=min(1.0, confidence),
        factors=extraction.factors,
        emotions=extract_emotions(situation),
        source_excerpt=sentence[:400],
        matched_marker=marker,
        needs_review=True,
    )


_COMMON = {
    "would", "could", "should", "really", "think", "thought", "know", "time",
    "going", "want", "wanted", "make", "made", "like", "just", "much", "even",
    "still", "back", "well", "also", "thing", "things", "something", "because",
}


def _recurring_terms(text: str, limit: int = 12) -> list[dict]:
    """Words the person returns to - a cheap proxy for what is on their mind."""
    counts: dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 5 or token in _COMMON:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [{"term": term, "count": count} for term, count in ranked[:limit] if count >= 2]
