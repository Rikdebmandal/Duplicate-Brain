"""Turn free-text scenarios and options into the structured factor space.

This is the *deterministic* half of scenario understanding. It runs with no
network access and no model weights, which matters for three reasons:

1. the system must work when no LLM key is configured;
2. the same text must always produce the same factors, otherwise a
   counterfactual ("what if risk were lower?") would not be comparable to the
   original prediction;
3. every extracted number can be traced back to the exact words that produced
   it, which is what ``evidence`` means in this codebase.

The LLM layer (``app.services.llm``) may *refine* these numbers, but it never
replaces them silently - refinements are recorded separately and are always
visible in the API response as ``source: "llm"``.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.services.decision_engine.taxonomy import (
    FACTOR_KEYS,
    FACTORS,
    blank_factors,
)

# --------------------------------------------------------------------------
# lexical helpers
# --------------------------------------------------------------------------

_NEGATORS = {
    "no", "not", "never", "without", "nothing", "none", "hardly", "barely",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "dont", "don't",
    "doesnt", "doesn't", "wont", "won't", "cannot", "cant", "can't", "little",
    "zero", "minimal", "lack", "lacks", "lacking", "free",
}

_INTENSIFIERS = {
    "very": 1.6, "extremely": 2.0, "hugely": 1.9, "massively": 1.9, "highly": 1.6,
    "really": 1.3, "so": 1.2, "quite": 1.15, "significantly": 1.5, "deeply": 1.7,
    "completely": 1.9, "totally": 1.8, "incredibly": 1.8, "enormously": 1.9,
    "somewhat": 0.7, "slightly": 0.55, "a bit": 0.6, "mildly": 0.6, "fairly": 0.9,
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")
#: "2x", "3.5x", "200%", "50 percent", "double", "triple"
_MULTIPLIER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:x|times)\b")
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\b")

#: How steeply a factor saturates as more cues are found. A single strong cue
#: should already register clearly; ten cues should not overflow.
_SATURATION = 0.85

# Option stance lexicons -----------------------------------------------------

_APPROACH_CUES = [
    "accept", "take it", "take the", "take", "say yes", "yes", "go for it", "go for",
    "go ahead", "join", "switch", "move", "relocate", "apply", "start", "launch",
    "buy", "invest", "quit", "resign", "leave the job", "change jobs", "pursue",
    "agree", "sign", "do it", "proceed", "commit", "enrol", "enroll", "register",
    "sign up", "hire", "make the", "accept the offer", "make the change", "put",
    "allocate", "attend", "present",
]
_AVOID_CUES = [
    "reject", "decline", "refuse", "say no", "turn down", "turn it down", "pass",
    "pass on", "skip", "stay", "remain", "keep", "retain", "continue", "stick with",
    "status quo", "do nothing", "not accept", "don't", "do not", "avoid", "withdraw",
    "back out", "stay put", "stay out", "maintain", "hold on to", "no change",
    "keep current", "let it go", "let the",
]
_DEFER_CUES = [
    "wait", "postpone", "delay", "defer", "think about it", "sleep on it", "revisit",
    "later", "ask for time", "gather more", "more information", "research first",
    "hold off", "not yet", "reconsider later", "take time", "keep searching",
    "keep looking", "check properly", "check first", "look into it", "for now",
]
_COMPROMISE_CUES = [
    "negotiate", "counter", "counter-offer", "part time", "part-time", "partial",
    "hybrid", "trial", "try it for", "pilot", "both", "middle ground", "compromise",
    "half", "phased", "temporarily", "remote arrangement", "ask for",
]

# The accept/decline lexicon above only works when options are *phrased* as
# accept/decline. "Take the stock options" and "Take the cash bonus" both read as
# "accept", yet one takes on risk and the other protects against it. These two
# lexicons capture that second axis - what the option does, rather than how it is
# worded - and disagreement between the axes lowers confidence rather than
# silently picking one.
_ACTIVE_MARKERS = [
    "new", "newer", "different", "another", "switch", "change", "equity", "stock",
    "stocks", "shares", "options", "fund", "index fund", "market", "startup",
    "venture", "unproven", "larger", "bigger", "more", "upside", "potential",
    "expand", "abroad", "overseas", "sabbatical", "freelance", "partner",
    "co-founder", "founder", "promotion", "loan", "borrow", "debt", "credit",
    "exit", "leave", "package", "relocate", "allocation", "growth", "lead",
]
_CONSERVATIVE_MARKERS = [
    "current", "existing", "present", "fixed", "fixed deposit", "guaranteed",
    "cash", "conservative", "safe", "secure", "outright", "proven", "established",
    "familiar", "smaller", "small", "unchanged", "certain", "predictable", "stable",
    "same", "as is", "savings", "deposit", "home", "hands on", "hands-on",
]

_EMOTION_LEXICON: dict[str, list[str]] = {
    "anxious": ["anxious", "nervous", "worried", "scared", "afraid", "fear", "panic",
                "stressed", "uneasy", "apprehensive", "dread"],
    "excited": ["excited", "thrilled", "eager", "energised", "energized", "enthusiastic",
                "pumped", "can't wait", "cannot wait"],
    "uncertain": ["uncertain", "confused", "torn", "conflicted", "unsure", "doubt",
                  "second-guessing", "ambivalent", "hesitant"],
    "confident": ["confident", "sure", "certain", "convinced", "clear", "decisive",
                  "no doubt"],
    "pressured": ["pressured", "rushed", "forced", "obligated", "expected to", "guilt",
                  "have to", "no choice"],
    "calm": ["calm", "relaxed", "at peace", "comfortable", "settled", "fine with"],
    "frustrated": ["frustrated", "angry", "annoyed", "fed up", "resentful", "irritated"],
    "sad": ["sad", "unhappy", "down", "disappointed", "regret", "miserable"],
}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------------
# factor extraction
# --------------------------------------------------------------------------


@dataclass
class FactorHit:
    """One piece of textual evidence for one factor."""

    factor: str
    cue: str
    weight: float
    negated: bool
    context: str

    def to_dict(self) -> dict:
        return {
            "factor": self.factor,
            "cue": self.cue,
            "weight": round(self.weight, 3),
            "negated": self.negated,
            "context": self.context,
        }


@dataclass
class FactorExtraction:
    """Result of reading factor magnitudes out of a block of text."""

    factors: dict[str, float] = field(default_factory=blank_factors)
    hits: list[FactorHit] = field(default_factory=list)
    #: 0..1 - how much textual signal was found at all. Low values mean the
    #: prediction rests mostly on the profile rather than on this scenario.
    coverage: float = 0.0

    def evidence_for(self, factor: str) -> list[FactorHit]:
        return [h for h in self.hits if h.factor == factor]

    def to_dict(self) -> dict:
        return {
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "coverage": round(self.coverage, 3),
            "hits": [h.to_dict() for h in self.hits],
        }


def _find_cue_positions(tokens: Sequence[str], joined: str, cue: str) -> list[int]:
    """Return token indices where ``cue`` occurs.

    Multi-word cues are matched on the joined token string so that punctuation
    and spacing differences do not matter; single words are matched exactly so
    that "pay" does not fire inside "payment plan for the company".
    """
    if " " in cue:
        positions: list[int] = []
        needle = " " + cue + " "
        hay = " " + joined + " "
        start = 0
        while True:
            idx = hay.find(needle, start)
            if idx < 0:
                break
            positions.append(hay[:idx].count(" "))
            start = idx + 1
        return positions
    return [i for i, tok in enumerate(tokens) if tok == cue]


def _negation_scale(tokens: Sequence[str], position: int, window: int = 3) -> float:
    """1.0 when the cue stands, 0.0 when a negator immediately precedes it."""
    lo = max(0, position - window)
    for offset, tok in enumerate(tokens[lo:position]):
        if tok in _NEGATORS:
            distance = position - (lo + offset)
            return 0.0 if distance <= 2 else 0.35
    return 1.0


def _intensity_scale(tokens: Sequence[str], position: int, window: int = 2) -> float:
    lo = max(0, position - window)
    scale = 1.0
    for tok in tokens[lo:position]:
        if tok in _INTENSIFIERS:
            scale *= _INTENSIFIERS[tok]
    return scale


def _context_snippet(tokens: Sequence[str], position: int, span: int = 5) -> str:
    lo = max(0, position - span)
    hi = min(len(tokens), position + span + 1)
    return " ".join(tokens[lo:hi])


def extract_factors(text: str, *, weight: float = 1.0) -> FactorExtraction:
    """Read factor magnitudes in [0, 1] out of free text.

    ``weight`` scales the contribution of this text block, so that (for
    example) option labels can count for less than the scenario body.
    """
    result = FactorExtraction()
    tokens = tokenize(text)
    if not tokens:
        return result
    joined = " ".join(tokens)

    raw: dict[str, float] = {k: 0.0 for k in FACTOR_KEYS}

    for factor in FACTORS:
        for cue, base in [(c, 1.0) for c in factor.cues] + [(c, 2.4) for c in factor.intense_cues]:
            cue_norm = " ".join(tokenize(cue))
            if not cue_norm:
                continue
            for pos in _find_cue_positions(tokens, joined, cue_norm):
                neg = _negation_scale(tokens, pos)
                intensity = _intensity_scale(tokens, pos)
                contribution = base * intensity * weight
                if neg == 0.0:
                    # An explicit denial is itself evidence - it pushes the
                    # magnitude down rather than merely failing to raise it.
                    raw[factor.key] -= contribution * 0.6
                    result.hits.append(
                        FactorHit(factor.key, cue, -contribution * 0.6, True,
                                  _context_snippet(tokens, pos))
                    )
                else:
                    contribution *= neg
                    raw[factor.key] += contribution
                    result.hits.append(
                        FactorHit(factor.key, cue, contribution, False,
                                  _context_snippet(tokens, pos))
                    )

    # Numeric amplifiers: "2x salary", "a 40% raise" are strong reward signals.
    multiplier_boost = 0.0
    for match in _MULTIPLIER_RE.finditer(joined):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 1.2 <= value <= 100:
            multiplier_boost = max(multiplier_boost, min(2.5, (value - 1.0) * 1.6))
    for match in _PERCENT_RE.finditer(joined):
        try:
            pct = float(match.group(1))
        except ValueError:
            continue
        if 5 <= pct <= 1000:
            multiplier_boost = max(multiplier_boost, min(2.5, pct / 60.0))
    if multiplier_boost and raw["financial_reward"] > 0:
        raw["financial_reward"] += multiplier_boost * weight
        result.hits.append(
            FactorHit("financial_reward", "numeric magnitude",
                      multiplier_boost * weight, False, "numeric multiplier in text")
        )

    total_signal = sum(max(0.0, v) for v in raw.values())
    for key, value in raw.items():
        # Saturating transform: keeps magnitudes in [0, 1] and makes the first
        # cue count for much more than the tenth.
        clipped = max(0.0, value)
        result.factors[key] = round(1.0 - math.exp(-_SATURATION * clipped), 4)

    result.coverage = round(min(1.0, total_signal / 6.0), 4)
    return result


def merge_extractions(*extractions: FactorExtraction) -> FactorExtraction:
    """Combine several text blocks into one factor reading (noisy-OR)."""
    merged = FactorExtraction()
    for key in FACTOR_KEYS:
        surviving = 1.0
        for ext in extractions:
            surviving *= 1.0 - ext.factors.get(key, 0.0)
        merged.factors[key] = round(1.0 - surviving, 4)
    for ext in extractions:
        merged.hits.extend(ext.hits)
    merged.coverage = round(min(1.0, sum(e.coverage for e in extractions)), 4)
    return merged


# --------------------------------------------------------------------------
# option stance classification
# --------------------------------------------------------------------------


@dataclass
class OptionStance:
    """How an option relates to the active ("do the thing") direction."""

    label: str
    approach: float          # 0 = pure status quo, 1 = fully takes the action
    defer: float
    compromise: float
    #: 0..1 confidence that the stance was actually detected rather than guessed
    confidence: float
    matched_cues: list[str] = field(default_factory=list)

    @property
    def stance(self) -> str:
        if self.defer >= 0.5 and self.defer >= self.compromise:
            return "defer"
        if self.compromise >= 0.5:
            return "compromise"
        return "approach" if self.approach >= 0.5 else "avoid"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "stance": self.stance,
            "approach": round(self.approach, 3),
            "defer": round(self.defer, 3),
            "compromise": round(self.compromise, 3),
            "confidence": round(self.confidence, 3),
            "matched_cues": self.matched_cues,
        }


def _cue_score(tokens: Sequence[str], joined: str, cues: Iterable[str]) -> tuple[float, list[str]]:
    score = 0.0
    matched: list[str] = []
    for cue in cues:
        cue_norm = " ".join(tokenize(cue))
        if not cue_norm:
            continue
        positions = _find_cue_positions(tokens, joined, cue_norm)
        for pos in positions:
            weight = 1.0 + 0.35 * (cue_norm.count(" "))  # longer cues are more specific
            if _negation_scale(tokens, pos) == 0.0:
                continue
            score += weight
            matched.append(cue)
            break
    return score, matched


@dataclass
class _OptionSignals:
    """The raw, pre-normalisation reading of a single option."""

    label: str
    activeness: float      # signed, roughly [-1.85, 1.85]
    strength: float        # 0-1, how much evidence there was either way
    agreement: float       # 0-1, do the two axes point the same way
    defer: float
    compromise: float
    matched_cues: list[str]


def _axis(positive: float, negative: float) -> tuple[float, float]:
    """Collapse two opposing counts into a signed value and a strength."""
    total = positive + negative
    if total <= 0:
        return 0.0, 0.0
    return (positive - negative) / total, min(1.0, total / 2.0)


def _factor_activeness(label: str) -> float:
    """How much exposure the option's own words imply, via the factor lexicon.

    Sums each detected factor's magnitude signed by ``sign_under_approach``, so
    an option mentioning risk and reward reads as active while one mentioning
    security and family reads as protective.
    """
    extraction = extract_factors(label, weight=1.0)
    total = 0.0
    for factor in FACTORS:
        if factor.sign_under_approach == 0:
            continue
        total += factor.sign_under_approach * extraction.factors.get(factor.key, 0.0)
    return max(-1.0, min(1.0, total))


def _signals(label: str) -> _OptionSignals:
    tokens = tokenize(label)
    joined = " ".join(tokens)

    approach_raw, approach_cues = _cue_score(tokens, joined, _APPROACH_CUES)
    avoid_raw, avoid_cues = _cue_score(tokens, joined, _AVOID_CUES)
    defer_raw, defer_cues = _cue_score(tokens, joined, _DEFER_CUES)
    comp_raw, comp_cues = _cue_score(tokens, joined, _COMPROMISE_CUES)
    active_raw, active_cues = _cue_score(tokens, joined, _ACTIVE_MARKERS)
    conservative_raw, conservative_cues = _cue_score(tokens, joined, _CONSERVATIVE_MARKERS)

    if any(tok in _NEGATORS for tok in tokens):
        avoid_raw += 1.0

    cue_value, cue_strength = _axis(approach_raw, avoid_raw)

    factor_bias = _factor_activeness(label)
    semantic_value, semantic_strength = _axis(
        active_raw + max(0.0, factor_bias),
        conservative_raw + max(0.0, -factor_bias),
    )

    if cue_strength > 0 and semantic_strength > 0:
        agreement = 1.0 - abs(cue_value - semantic_value) / 2.0
    else:
        agreement = 0.75

    return _OptionSignals(
        label=label,
        activeness=cue_value + 0.85 * semantic_value,
        strength=max(cue_strength, semantic_strength),
        agreement=agreement,
        defer=min(1.0, defer_raw / 1.5),
        compromise=min(1.0, comp_raw / 1.5),
        matched_cues=sorted(
            set(approach_cues + avoid_cues + defer_cues + comp_cues
                + active_cues + conservative_cues)
        ),
    )


#: Below this spread the options are not separable on the approach axis at all
#: (a lateral choice such as "the smaller flat" vs "the larger flat"). Rather
#: than guess, every option is reported at 0.5 with near-zero confidence, which
#: makes the decision contribute almost nothing to trait inference.
MIN_STANCE_SPREAD = 0.35


def resolve_stances(labels: Sequence[str]) -> list[OptionStance]:
    """Classify a whole option set *contrastively*.

    Classifying options one at a time is what produces "both options are an
    approach": every branch of a decision usually starts with an active verb.
    What actually identifies the active branch is which option is *more* active
    than the others, so activeness is scored per option and then normalised
    across the set.
    """
    signals = [_signals(label) for label in labels]
    if not signals:
        return []

    values = [s.activeness for s in signals]
    lo, hi = min(values), max(values)
    spread = hi - lo
    separable = spread >= MIN_STANCE_SPREAD

    stances: list[OptionStance] = []
    for signal in signals:
        if separable:
            approach = (signal.activeness - lo) / spread
            confidence = (
                min(1.0, spread / 1.2)
                * (0.4 + 0.6 * signal.strength * signal.agreement)
            )
        else:
            approach = 0.5
            confidence = 0.08

        if signal.defer > 0 or signal.compromise > 0:
            confidence = max(confidence, 0.35 * max(signal.defer, signal.compromise))
            # Deferring is neither approach nor avoid: pull it to the middle.
            approach = approach * (1.0 - 0.6 * signal.defer) + 0.3 * signal.defer

        stances.append(
            OptionStance(
                label=signal.label,
                approach=round(float(min(1.0, max(0.0, approach))), 3),
                defer=round(float(signal.defer), 3),
                compromise=round(float(signal.compromise), 3),
                confidence=round(float(min(1.0, max(0.0, confidence))), 3),
                matched_cues=signal.matched_cues,
            )
        )
    return stances


def classify_option(label: str, alternatives: Sequence[str] = ()) -> OptionStance:
    """Classify one option, ideally against the alternatives it competed with.

    Always prefer :func:`resolve_stances` when the whole option set is
    available - a stance is only meaningful relative to what it was chosen
    over. This wrapper exists for the paths that legitimately hold one label.
    """
    labels = [label, *alternatives]
    return resolve_stances(labels)[0]


# --------------------------------------------------------------------------
# emotion + full scenario parse
# --------------------------------------------------------------------------


def extract_emotions(text: str) -> dict[str, float]:
    """Detect emotional colouring of a decision narrative.

    This is *linguistic*, not clinical: it reports which emotion words the
    person used, and is never used to infer a psychological condition.
    """
    tokens = tokenize(text)
    if not tokens:
        return {}
    joined = " ".join(tokens)
    scores: dict[str, float] = {}
    for emotion, cues in _EMOTION_LEXICON.items():
        raw, _ = _cue_score(tokens, joined, cues)
        if raw > 0:
            scores[emotion] = round(min(1.0, raw / 2.0), 3)
    return scores


@dataclass
class ParsedScenario:
    """Everything the deterministic layer can say about a situation."""

    text: str
    factors: dict[str, float]
    coverage: float
    hits: list[FactorHit]
    options: list[OptionStance]
    emotions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "coverage": round(self.coverage, 3),
            "options": [o.to_dict() for o in self.options],
            "emotions": self.emotions,
            "evidence": [h.to_dict() for h in self.hits],
        }


def parse_scenario(
    text: str,
    options: Sequence[str],
    *,
    context_text: str = "",
    factor_overrides: dict[str, float] | None = None,
) -> ParsedScenario:
    """Parse a scenario body plus its options into factors and stances.

    Option labels contribute to the factor reading at reduced weight - "accept
    the Bangalore offer" genuinely tells us relocation is involved - but the
    resulting factor vector stays a single, sliderable description of the
    *situation*, which is what the counterfactual engine needs.
    """
    extractions = [extract_factors(text, weight=1.0)]
    if context_text:
        extractions.append(extract_factors(context_text, weight=0.8))
    for option in options:
        extractions.append(extract_factors(option, weight=0.55))

    merged = merge_extractions(*extractions)
    factors = dict(merged.factors)
    if factor_overrides:
        for key, value in factor_overrides.items():
            if key in factors and value is not None:
                factors[key] = float(min(1.0, max(0.0, value)))

    return ParsedScenario(
        text=text,
        factors=factors,
        coverage=merged.coverage,
        hits=merged.hits,
        options=resolve_stances(options),
        emotions=extract_emotions(f"{text} {context_text}"),
    )
