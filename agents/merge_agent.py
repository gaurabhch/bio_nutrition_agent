from groq import AsyncGroq

from agents.state import AgentState
from config import GROQ_MODEL, GROQ_TIMEOUT, RESPONSE_MODES

_TAG_TONE_RULES: dict[str, str] = {
    "high_stress": "Open with a brief warm acknowledgment before the main answer.",
    "goal:actively_manage": "End with one practical next step if it fits naturally.",
    "goal:learn_more": "Keep the explanation slightly more educational and explain the reason behind recommendations.",
    "undiagnosed": "Use cautious language and avoid sounding definitive.",
}

_MODE_TONE_RULES: dict[str, str] = {
    RESPONSE_MODES["EMOTIONAL"]: "Use a supportive and calm tone.",
    RESPONSE_MODES["CLARIFICATION"]: "Ask exactly one focused follow-up question and do not add extra guidance.",
    RESPONSE_MODES["INFORMATION"]: "Be clear, direct, and helpful.",
    RESPONSE_MODES["CRISIS"]: "",
}


def _build_tone_prompt(user_tags: list[str], response_mode: str) -> str:
    rules: list[str] = []

    mode_rule = _MODE_TONE_RULES.get(response_mode)
    if mode_rule:
        rules.append(mode_rule)

    for tag in user_tags:
        if tag in _TAG_TONE_RULES:
            rules.append(_TAG_TONE_RULES[tag])

    tone_lines = "\n".join(f"- {r}" for r in rules) if rules else "- Be warm, clear, and supportive."

    return (
        "You are personalising a nutrition and health response.\n\n"
        "Apply these tone adjustments:\n"
        f"{tone_lines}\n\n"
        "Rules:\n"
        "- Do not add new medical or nutrition information.\n"
        "- Do not remove important facts already present.\n"
        "- Do NOT include citation labels, cluster IDs, or internal reference tags.\n"
        "- Preserve the original structure when useful. If the response is already a bullet list, numbered list, or meal plan, keep that format.\n"
        "- If the response is already direct and good, make only minimal tone edits.\n"
        "- Return only the final personalised response text.\n"
    )


async def merge_agent_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    if state.get("final_response"):
        return state

    verified = state.get("verified_response") or state.get("raw_response", "")
    user_tags = state.get("user_tags", [])
    response_mode = state.get("response_mode", RESPONSE_MODES["INFORMATION"])

    if not verified:
        return {
            **state,
            "final_response": "",
            "sources": [],
        }

    tone_prompt = _build_tone_prompt(user_tags, response_mode)
    user_prompt = f"Personalise this response for the user without changing its meaning:\n\n{verified}"

    try:
        result = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": tone_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=700,
            timeout=GROQ_TIMEOUT,
        )
        final_response = (result.choices[0].message.content or "").strip()
    except Exception:
        final_response = verified

    kb_sources: list[str] = []
    for chunk in state.get("retrieved_context", []):
        refs = chunk.get("reference_sources") or []
        if isinstance(refs, list):
            kb_sources.extend(refs)
        elif isinstance(refs, str) and refs:
            kb_sources.append(refs)

    kb_sources = list(dict.fromkeys(kb_sources))[:3]
    pubmed_citations = list(dict.fromkeys(state.get("citations", [])))[:2]
    all_sources = kb_sources + pubmed_citations

    return {
        **state,
        "final_response": final_response,
        "sources": all_sources[:5],
    }