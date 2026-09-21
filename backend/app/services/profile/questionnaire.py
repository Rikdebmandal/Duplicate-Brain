"""The structured self-report instrument.

Scope note: this is a *preference* questionnaire, not a psychometric test. It
measures how someone says they trade things off when deciding. It does not
screen for, score, or infer any psychological or medical condition, and no item
is worded to do so. Self-report is weighted well below observed behaviour in
:mod:`app.services.profile.inference` precisely because people describe
themselves more consistently than they act.

Every item is a 1-5 Likert statement mapped to exactly one trait, with
``direction = -1`` marking reverse-scored items.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionnaireItem:
    key: str
    construct: str
    prompt: str
    trait: str
    direction: int  # +1 agreeing raises the trait, -1 agreeing lowers it


SCALE_LABELS = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]

ITEMS: list[QuestionnaireItem] = [
    # -- risk tolerance ---------------------------------------------------
    QuestionnaireItem("risk_1", "Risk tolerance",
                      "I am comfortable making a choice that could go badly wrong if it also "
                      "has a big upside.", "risk_tolerance", 1),
    QuestionnaireItem("risk_2", "Risk tolerance",
                      "I would rather have a guaranteed modest result than a chance at a much "
                      "better one.", "risk_tolerance", -1),
    QuestionnaireItem("risk_3", "Risk tolerance",
                      "I have walked away from opportunities mainly because they felt too "
                      "risky.", "risk_tolerance", -1),

    # -- uncertainty tolerance -------------------------------------------
    QuestionnaireItem("unc_1", "Uncertainty tolerance",
                      "I can act decisively even when important information is missing.",
                      "uncertainty_tolerance", 1),
    QuestionnaireItem("unc_2", "Uncertainty tolerance",
                      "Not knowing how something will turn out bothers me a great deal.",
                      "uncertainty_tolerance", -1),
    QuestionnaireItem("unc_3", "Uncertainty tolerance",
                      "I need to see how something has worked for others before I will try it.",
                      "uncertainty_tolerance", -1),

    # -- financial priority ----------------------------------------------
    QuestionnaireItem("fin_1", "Financial priorities",
                      "Financial outcomes are usually the deciding factor for me.",
                      "financial_priority", 1),
    QuestionnaireItem("fin_2", "Financial priorities",
                      "I would accept clearly less money for work I found more meaningful.",
                      "financial_priority", -1),
    QuestionnaireItem("fin_3", "Financial priorities",
                      "I keep close track of how my choices affect my finances.",
                      "financial_priority", 1),

    # -- family priority ---------------------------------------------------
    QuestionnaireItem("fam_1", "Family and relationships",
                      "Staying physically close to my family shapes my major decisions.",
                      "family_priority", 1),
    QuestionnaireItem("fam_2", "Family and relationships",
                      "I would relocate far from the people closest to me for the right "
                      "opportunity.", "family_priority", -1),
    QuestionnaireItem("fam_3", "Family and relationships",
                      "My obligations to my family come before my own plans.",
                      "family_priority", 1),

    # -- career priority ---------------------------------------------------
    QuestionnaireItem("car_1", "Career",
                      "Professional growth is one of the things I optimise my life around.",
                      "career_priority", 1),
    QuestionnaireItem("car_2", "Career",
                      "I would turn down a promotion that cost me time outside work.",
                      "career_priority", -1),

    # -- social influence / independence / conformity ----------------------
    QuestionnaireItem("soc_1", "Social preferences",
                      "What the people around me think matters when I decide.",
                      "social_influence", 1),
    QuestionnaireItem("soc_2", "Independence",
                      "I make important decisions on my own, whatever others advise.",
                      "social_influence", -1),
    QuestionnaireItem("soc_3", "Conformity",
                      "I am uncomfortable making a choice that most people around me would "
                      "consider strange.", "social_influence", 1),

    # -- long-term orientation / patience ----------------------------------
    QuestionnaireItem("lto_1", "Long-term orientation",
                      "I willingly give up something now for a clearly better result later.",
                      "long_term_orientation", 1),
    QuestionnaireItem("lto_2", "Patience",
                      "Waiting for a delayed reward is hard for me.",
                      "long_term_orientation", -1),
    QuestionnaireItem("lto_3", "Long-term orientation",
                      "I think about where a decision leaves me in five or ten years.",
                      "long_term_orientation", 1),

    # -- loss aversion ------------------------------------------------------
    QuestionnaireItem("loss_1", "Loss aversion",
                      "Losing something I already have upsets me more than gaining something "
                      "equivalent pleases me.", "loss_aversion", 1),
    QuestionnaireItem("loss_2", "Loss aversion",
                      "I protect what I have already built rather than trade it for something "
                      "better.", "loss_aversion", 1),
    QuestionnaireItem("loss_3", "Loss aversion",
                      "A setback does not weigh on me much once it has passed.",
                      "loss_aversion", -1),

    # -- reward sensitivity -------------------------------------------------
    QuestionnaireItem("rew_1", "Reward sensitivity",
                      "A large enough payoff will get me to reconsider a firm position.",
                      "reward_sensitivity", 1),
    QuestionnaireItem("rew_2", "Reward sensitivity",
                      "Exceptional opportunities do not tempt me out of my plans.",
                      "reward_sensitivity", -1),

    # -- planning tendency / impulsivity / emotional influence --------------
    QuestionnaireItem("plan_1", "Planning tendency",
                      "I research thoroughly before committing to anything significant.",
                      "planning_tendency", 1),
    QuestionnaireItem("plan_2", "Impulsivity",
                      "I often decide quickly and deal with the consequences later.",
                      "planning_tendency", -1),
    QuestionnaireItem("plan_3", "Emotional influence",
                      "How I feel in the moment changes what I decide.",
                      "planning_tendency", -1),
    QuestionnaireItem("plan_4", "Planning tendency",
                      "I write things down or compare options explicitly before choosing.",
                      "planning_tendency", 1),

    # -- moral sensitivity ---------------------------------------------------
    QuestionnaireItem("mor_1", "Moral preferences",
                      "I have refused something advantageous because it conflicted with my "
                      "principles.", "moral_sensitivity", 1),
    QuestionnaireItem("mor_2", "Moral preferences",
                      "When the stakes are high I will take the pragmatic route over the "
                      "principled one.", "moral_sensitivity", -1),
]

ITEMS_BY_KEY: dict[str, QuestionnaireItem] = {item.key: item for item in ITEMS}


def normalise_answer(item_key: str, raw_value: int) -> float:
    """Map a 1-5 Likert answer to a [0, 1] observation for the item's trait."""
    item = ITEMS_BY_KEY.get(item_key)
    if item is None:
        raise KeyError(f"unknown questionnaire item: {item_key}")
    if not 1 <= int(raw_value) <= 5:
        raise ValueError("questionnaire answers must be between 1 and 5")
    normalised = (int(raw_value) - 1) / 4.0
    return normalised if item.direction > 0 else 1.0 - normalised


def items_payload() -> list[dict]:
    return [
        {
            "key": item.key,
            "construct": item.construct,
            "prompt": item.prompt,
            "trait": item.trait,
            "reverse_scored": item.direction < 0,
            "scale": SCALE_LABELS,
        }
        for item in ITEMS
    ]
