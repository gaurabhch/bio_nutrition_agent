from groq import AsyncGroq
from config import GROQ_MODEL, GROQ_TIMEOUT

REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a nutrition and women's health assistant.

Your job:
- Rewrite the user's message into a clear, searchable query.
- Use conversation history to resolve pronouns and references ("that", "it", "this").
- Translate Hinglish or informal language into plain English.
- Preserve important task intent such as:
  - meal plan
  - list of foods or ingredients
  - specific number requested
  - Indian diet or cultural food context
  - goals like insulin sensitivity, inflammation, satiety, gut health, fertility, pregnancy, hydration
- Keep the rewrite concise, but do NOT remove essential constraints.
- If the message is too vague to rewrite meaningfully, return exactly: CLARIFICATION_NEEDED

Good examples:
"I've been feeling bloated after meals" -> "foods and diet strategies for bloating and digestion"
"Name five Indian foods good for PCOS" -> "five recommended foods in Indian diet for PCOS"
"Create a one-day meal plan for insulin sensitivity and satiety" -> "one-day meal plan for insulin sensitivity and satiety in PCOS"
"Mujhe constipation aur gas ho raha hai" -> "diet for constipation and gas relief"
"I don't feel like myself lately" -> CLARIFICATION_NEEDED

Return ONLY the rewritten query or CLARIFICATION_NEEDED. No explanation."""
    

async def rewrite_query(
    message: str,
    history: list[dict],
    use_case: str,
    groq_client: AsyncGroq,
) -> str:
    history_text = _format_history(history)
    prompt = (
        f"Health context: {use_case}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"User message: {message}\n\n"
        f"Rewrite:"
    )

    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=60,
            timeout=GROQ_TIMEOUT,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else message
    except Exception:
        return message


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no previous messages)"

    lines = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)