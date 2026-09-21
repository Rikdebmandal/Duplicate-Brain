"""The shared vocabulary of the twin: decision *factors* and behavioural *traits*.

Everything downstream - feature vectors, the utility model, trait inference,
explanations and the counterfactual sliders - is defined against these two
lists, so adding a factor or trait here propagates through the whole system.

Design note
-----------
A *factor* describes the **situation**: "how much financial reward is on the
table", "how much stability is at stake". It is always a magnitude in [0, 1] -
never a signed judgement. Whether a factor pushes towards or away from a given
option is decided by two separate things:

* ``sign_under_approach`` - does taking the active option *increase exposure*
  to this factor (+1: you gain the reward, you take on the risk) or *spend
  down* a protected good (-1: you give up stability, you incur a family cost)?
* the person's traits - how much this individual weights that factor.

Keeping magnitude (situation) separate from sign and weight (person) is what
lets the same scenario produce different predictions for different people, and
what makes the explanations decomposable into per-factor contributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Factor:
    """A dimension of a situation, measured as a magnitude in [0, 1]."""

    key: str
    label: str
    description: str
    #: +1 -> the active/approach option takes on more of this factor.
    #: -1 -> the active/approach option spends down this protected good.
    #:  0 -> the factor modulates other factors rather than valencing directly.
    sign_under_approach: Literal[-1, 0, 1]
    #: Words/phrases that raise this factor's magnitude when parsing free text.
    cues: list[str] = field(default_factory=list)
    #: Cues that specifically indicate a *high* level of the factor.
    intense_cues: list[str] = field(default_factory=list)


FACTORS: list[Factor] = [
    Factor(
        key="financial_reward",
        label="Financial reward",
        description="Money, compensation or material gain available from the active option.",
        sign_under_approach=1,
        cues=["salary", "pay", "paid", "money", "bonus", "raise", "income", "compensation",
              "profit", "return", "earn", "earning", "lakh", "crore", "rupee", "dollar",
              "revenue", "stipend", "package", "financial gain", "payout", "equity", "funding"],
        intense_cues=["double", "2x", "3x", "triple", "twice", "huge", "massive",
                      "highest", "lucrative", "windfall", "life-changing money"],
    ),
    Factor(
        key="risk",
        label="Risk",
        description="Probability and severity of a bad outcome if the active option is taken.",
        sign_under_approach=1,
        cues=["risk", "risky", "gamble", "could fail", "might fail", "failure", "lose", "loss",
              "downside", "volatile", "unproven", "startup", "no guarantee", "bet", "danger",
              "unsafe", "chance of losing", "speculative", "crash"],
        intense_cues=["very risky", "high risk", "huge risk", "all my savings", "everything",
                      "could go bankrupt", "lose everything", "life savings"],
    ),
    Factor(
        key="uncertainty",
        label="Uncertainty",
        description="How much about the outcome is simply unknown or unknowable.",
        sign_under_approach=1,
        cues=["uncertain", "unclear", "unknown", "not sure", "unsure", "ambiguous",
              "no information", "hard to say", "unpredictable", "vague", "undecided",
              "no data", "only been operating", "new company", "untested", "first time",
              "never done", "no track record", "early stage", "early-stage",
              "recently founded", "newly formed", "just started", "first year",
              "one year old", "two years old", "a year old", "brand new"],
        intense_cues=["completely unknown", "no idea", "total unknown", "no way to know"],
    ),
    Factor(
        key="stability",
        label="Stability at stake",
        description="How much settled, predictable life the active option would disturb.",
        sign_under_approach=-1,
        cues=["stable", "stability", "secure", "security", "steady", "settled", "permanent",
              "safe job", "current job", "predictable", "routine", "established", "tenure",
              "guaranteed", "reliable", "comfortable", "status quo"],
        intense_cues=["very stable", "completely secure", "rock solid", "lifetime"],
    ),
    Factor(
        key="family_impact",
        label="Family / relationship cost",
        description="How much the active option would disrupt family or close relationships.",
        sign_under_approach=-1,
        cues=["family", "parents", "mother", "father", "spouse", "wife", "husband",
              "partner", "children", "kids", "son", "daughter", "relocate", "move away",
              "another city", "different city", "abroad", "long distance", "away from home",
              "relationship", "marriage", "caregiving", "sibling", "brother", "sister"],
        intense_cues=["elderly parents", "sick parent", "newborn", "leave my family",
                      "far from family", "another country", "overseas"],
    ),
    Factor(
        key="career_growth",
        label="Career growth",
        description="Skill, status or long-term professional advancement on offer.",
        sign_under_approach=1,
        cues=["career", "promotion", "growth", "learn", "learning", "skill", "senior", "lead",
              "leadership", "opportunity", "experience", "portfolio", "resume", "advance",
              "title", "responsibility", "mentor", "exposure", "role"],
        intense_cues=["dream role", "huge opportunity", "once in a lifetime", "big break"],
    ),
    Factor(
        key="social_consequence",
        label="Social consequence",
        description="Judgement, reputation or friction with the person's social circle.",
        sign_under_approach=-1,
        cues=["what people think", "judge", "judged", "reputation", "friends", "colleagues",
              "peers", "society", "embarrass", "awkward", "conflict", "disappoint",
              "expectations", "boss", "team", "criticism", "gossip", "look bad", "let down"],
        intense_cues=["everyone would know", "public", "humiliating", "burn bridges"],
    ),
    Factor(
        key="moral_weight",
        label="Moral weight",
        description="Ethical cost or value conflict created by the active option.",
        sign_under_approach=-1,
        cues=["ethical", "ethics", "moral", "morally", "right thing", "wrong", "honest",
              "dishonest", "lie", "cheat", "fair", "unfair", "integrity", "guilt", "guilty",
              "principle", "values", "conscience", "harm"],
        intense_cues=["deeply unethical", "against my values", "could not live with"],
    ),
    Factor(
        key="time_pressure",
        label="Time pressure",
        description="How little time there is to decide. Modulates, rather than valences.",
        sign_under_approach=0,
        cues=["deadline", "urgent", "immediately", "today", "tomorrow", "asap", "right away",
              "expires", "limited time", "this week", "quickly", "no time",
              "must decide", "closing soon", "48 hours", "24 hours"],
        intense_cues=["decide today", "within hours", "final notice", "last chance"],
    ),
    Factor(
        key="effort_cost",
        label="Effort cost",
        description="Work, hassle or sustained discipline the active option demands.",
        sign_under_approach=1,
        cues=["effort", "hard work", "difficult", "demanding", "hours", "overtime", "commute",
              "exhausting", "tiring", "workload", "burnout", "grind", "study", "training",
              "discipline", "sacrifice", "hectic", "stressful"],
        intense_cues=["extremely demanding", "80 hours", "no weekends", "brutal"],
    ),
    Factor(
        key="autonomy",
        label="Autonomy",
        description="Independence and control over one's own time and choices.",
        sign_under_approach=1,
        cues=["independent", "independence", "own boss", "freedom", "flexible", "autonomy",
              "control", "own terms", "remote", "entrepreneur", "freelance", "start my own"],
        intense_cues=["complete freedom", "total control", "fully independent"],
    ),
    Factor(
        key="long_term_benefit",
        label="Long-term benefit",
        description="Payoff that arrives later rather than now.",
        sign_under_approach=1,
        cues=["long term", "long-term", "future", "years", "eventually", "later", "retirement",
              "invest", "investment", "compound", "down the line", "in the long run", "someday",
              "five years", "ten years", "sustainable", "build"],
        intense_cues=["sets me up for life", "decades"],
    ),
]

FACTOR_INDEX: dict[str, int] = {f.key: i for i, f in enumerate(FACTORS)}
FACTOR_KEYS: list[str] = [f.key for f in FACTORS]
FACTOR_BY_KEY: dict[str, Factor] = {f.key: f for f in FACTORS}


@dataclass(frozen=True)
class Trait:
    """A learned behavioural parameter of the person, in [0, 1]."""

    key: str
    label: str
    description: str
    low_label: str
    high_label: str
    #: How the trait is read in explanations: "tolerance" traits are centred at
    #: 0.5 (below = repelled by the factor, above = attracted); "priority"
    #: traits run 0 -> 1 as "cares little" -> "cares a lot".
    kind: Literal["tolerance", "priority"]


TRAITS: list[Trait] = [
    Trait("risk_tolerance", "Risk tolerance",
          "Willingness to accept a chance of a bad outcome in exchange for upside.",
          "risk-averse", "risk-seeking", "tolerance"),
    Trait("uncertainty_tolerance", "Uncertainty tolerance",
          "Comfort acting when the outcome is genuinely unknown.",
          "needs certainty", "comfortable with ambiguity", "tolerance"),
    Trait("financial_priority", "Financial priority",
          "Weight placed on money and material outcomes.",
          "money is secondary", "money is decisive", "priority"),
    Trait("family_priority", "Family priority",
          "Weight placed on family proximity and close relationships.",
          "low weight", "high weight", "priority"),
    Trait("career_priority", "Career priority",
          "Weight placed on professional advancement and growth.",
          "low weight", "high weight", "priority"),
    Trait("social_influence", "Social influence",
          "How much other people's expectations and judgement shape the choice.",
          "self-directed", "strongly influenced", "priority"),
    Trait("long_term_orientation", "Long-term orientation",
          "Preference for later, larger payoffs over immediate ones.",
          "present-focused", "future-focused", "priority"),
    Trait("loss_aversion", "Loss aversion",
          "How much more a potential loss weighs than an equivalent gain.",
          "symmetric", "losses dominate", "priority"),
    Trait("reward_sensitivity", "Reward sensitivity",
          "How strongly a large upside pulls the decision.",
          "muted response", "strong pull", "priority"),
    Trait("planning_tendency", "Planning tendency",
          "Preference for deliberating and gathering information before acting.",
          "acts quickly", "plans thoroughly", "priority"),
    Trait("moral_sensitivity", "Moral sensitivity",
          "Weight placed on ethical considerations and personal values.",
          "pragmatic", "principle-driven", "priority"),
]

TRAIT_INDEX: dict[str, int] = {t.key: i for i, t in enumerate(TRAITS)}
TRAIT_KEYS: list[str] = [t.key for t in TRAITS]
TRAIT_BY_KEY: dict[str, Trait] = {t.key: t for t in TRAITS}

#: Prior used before any evidence exists. 0.5 = "no information", which keeps
#: the twin explicitly agnostic rather than inventing a personality.
NEUTRAL_TRAIT_VALUE = 0.5

#: Option "stances" - how an option relates to the active/approach direction.
OPTION_STANCES: list[str] = ["approach", "avoid", "defer", "compromise"]


def blank_factors() -> dict[str, float]:
    """A factor dict with every magnitude at zero."""
    return {f.key: 0.0 for f in FACTORS}


def neutral_traits() -> dict[str, float]:
    """A trait dict at the no-information prior."""
    return {t.key: NEUTRAL_TRAIT_VALUE for t in TRAITS}
