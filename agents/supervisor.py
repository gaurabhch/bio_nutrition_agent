from groq import AsyncGroq

from agents.state import AgentState
from utils.query_rewriter import rewrite_query
from config import (
    CRISIS_KEYWORDS,
    FALSE_POSITIVE_GUARD,
    HELPLINE_RESPONSE,
    EMOTIONAL_KEYWORDS,
    VAGUE_PATTERNS,
    DOMAIN_KEYWORDS,
    KEYWORD_ROUTING_THRESHOLD,
    RESPONSE_MODES,
    GROQ_MODEL,
    GROQ_TIMEOUT,
)

IMPLICIT_DOMAIN_HINTS = {
    "gut_health": [
        "bloating", "bloated", "digestion", "digestive", "constipation",
        "diarrhea", "gas", "acidity", "ibs", "gut"
    ],
    "meal_planning": [
        "meal plan", "one-day meal plan", "breakfast", "lunch", "dinner",
        "snack", "satiety", "full throughout the day", "what should i eat"
    ],
    "macronutrients": [
        "protein", "carbs", "carbohydrate", "fat", "fiber", "macros",
        "balanced meal", "insulin sensitivity"
    ],
    "weight_management": [
        "weight loss", "weight gain", "calorie", "portion", "obesity",
        "reduce inflammation", "anti-inflammatory"
    ],
    "supplements": [
        "supplement", "vitamin", "omega 3", "magnesium", "iron", "b12"
    ],
    "hydration": [
        "water", "hydration", "electrolyte", "dehydration"
    ],
    "dietary_restrictions": [
        "gluten", "lactose", "allergy", "vegan", "vegetarian", "restriction"
    ],
    "indian_guidance": [
        "indian diet", "indian woman", "roti", "rice", "dal", "paneer",
        "curd", "desi", "indian food"
    ],
    "pregnancy_nutrition": [
        "pregnancy", "pregnant", "prenatal", "trimester"
    ],
    "postpartum_nutrition": [
        "postpartum", "after delivery", "breastfeeding", "lactation"
    ],
    "fertility_nutrition": [
        "fertility", "ovulation", "conceive", "trying to conceive",
        "reproductive health", "infertility"
    ],
}


def _is_crisis(message: str) -> bool:
    lowered = message.lower()
    if any(fp in lowered for fp in FALSE_POSITIVE_GUARD):
        return False
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def _detect_response_mode(message: str, rewritten: str) -> str:
    if rewritten == "CLARIFICATION_NEEDED":
        return RESPONSE_MODES["CLARIFICATION"]

    lowered = message.lower()
    if any(kw in lowered for kw in EMOTIONAL_KEYWORDS):
        return RESPONSE_MODES["EMOTIONAL"]
    if any(p in lowered for p in VAGUE_PATTERNS):
        return RESPONSE_MODES["CLARIFICATION"]

    return RESPONSE_MODES["INFORMATION"]


def _score_domains(rewritten_query: str, use_case: str, user_tags: list[str]) -> dict[str, int]:
    lowered = rewritten_query.lower()
    scores: dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in lowered)

    primary = f"{use_case}_general"
    if primary in scores:
        scores[primary] += 1

    for tag in user_tags:
        if tag in scores:
            scores[tag] += 1

    return scores


def _score_implicit_domains(text: str) -> dict[str, int]:
    lowered = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS.keys()}

    for domain, hints in IMPLICIT_DOMAIN_HINTS.items():
        scores[domain] = sum(1 for hint in hints if hint in lowered)

    return scores


def _rule_based_domain_override(message: str, rewritten: str) -> str | None:
    text = f"{message} {rewritten}".lower()

    if any(k in text for k in [
        "meal plan", "one-day meal plan", "breakfast", "lunch", "dinner",
        "snack", "satiety", "full throughout the day"
    ]):
        return "meal_planning"

    if any(k in text for k in ["indian diet", "indian woman", "roti", "dal", "paneer", "desi diet"]):
        return "indian_guidance"

    if any(k in text for k in ["bloating", "digestion", "constipation", "gas", "gut", "acidity"]):
        return "gut_health"

    if any(k in text for k in ["protein", "fiber", "carbs", "fat", "macros", "insulin sensitivity"]):
        return "macronutrients"

    if any(k in text for k in ["reduce inflammation", "anti-inflammatory", "weight loss", "portion", "calorie"]):
        return "weight_management"

    if any(k in text for k in ["fertility", "ovulation", "conceive", "trying to conceive", "infertility"]):
        return "fertility_nutrition"

    return None


async def _llm_classify_domain(rewritten_query: str, groq_client: AsyncGroq) -> str:
    domain_list = ", ".join(DOMAIN_KEYWORDS.keys())
    prompt = (
        "Classify this nutrition/health query into exactly ONE domain from the list below.\n"
        f"Domains: {domain_list}\n\n"
        f"Query: {rewritten_query}\n\n"
        "Return only the domain string. No explanation."
    )

    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
            timeout=GROQ_TIMEOUT,
        )
        domain = (response.choices[0].message.content or "").strip().lower()
        return domain if domain in DOMAIN_KEYWORDS else "nutrition_general"
    except Exception:
        return "nutrition_general"


async def supervisor_node(state: AgentState, groq_client: AsyncGroq) -> AgentState:
    message = state["messages"][-1]["text"]

    if _is_crisis(message):
        return {
            **state,
            "is_crisis": True,
            "is_flagged": True,
            "flag_reason": "crisis_keyword",
            "final_response": HELPLINE_RESPONSE,
            "next_agent": "crisis_end",
            "routing_reason": "crisis",
            "response_mode": RESPONSE_MODES["CRISIS"],
            "rewritten_query": message,
        }

    history = state["messages"][-6:]
    rewritten = await rewrite_query(message, history, state["use_case"], groq_client)
    state = {**state, "rewritten_query": rewritten}

    mode = _detect_response_mode(message, rewritten)
    state = {**state, "response_mode": mode}

    if mode == RESPONSE_MODES["CLARIFICATION"]:
        return {
            **state,
            "next_agent": "clarification_agent",
            "routing_reason": "clarification",
        }

    forced_domain = _rule_based_domain_override(message, rewritten)
    if forced_domain:
        return {
            **state,
            "next_agent": f"{forced_domain}_agent",
            "routing_reason": "rule_override",
            "is_crisis": False,
            "is_flagged": False,
            "flag_reason": None,
        }

    domain_scores = _score_domains(rewritten, state["use_case"], state["user_tags"])
    best_domain = max(domain_scores, key=domain_scores.get)
    best_score = domain_scores[best_domain]

    if best_score > KEYWORD_ROUTING_THRESHOLD:
        return {
            **state,
            "next_agent": f"{best_domain}_agent",
            "routing_reason": "keyword",
            "is_crisis": False,
            "is_flagged": False,
            "flag_reason": None,
        }

    implicit_scores = _score_implicit_domains(f"{message} {rewritten}")
    implicit_best_domain = max(implicit_scores, key=implicit_scores.get)
    implicit_best_score = implicit_scores[implicit_best_domain]

    if implicit_best_score > 0:
        return {
            **state,
            "next_agent": f"{implicit_best_domain}_agent",
            "routing_reason": "implicit_routing",
            "is_crisis": False,
            "is_flagged": False,
            "flag_reason": None,
        }

    best_domain = await _llm_classify_domain(rewritten, groq_client)
    return {
        **state,
        "next_agent": f"{best_domain}_agent",
        "routing_reason": "llm_fallback",
        "is_crisis": False,
        "is_flagged": False,
        "flag_reason": None,
    }