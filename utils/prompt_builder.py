def build_system_prompt(domain: str) -> str:
    readable_domain = domain.replace("_", " ")

    return (
        f"You are a medically informed and practical {readable_domain} nutrition assistant.\n\n"
        "Guidelines:\n"
        "- Base your response strictly on the retrieved knowledge provided below.\n"
        "- Do NOT include citation labels, cluster IDs, retrieval scores, or internal tags in your response.\n"
        "- Use plain, clear English and stay concise.\n"
        "- Be helpful and practical, but do not diagnose or prescribe medications or dosages.\n"
        "- If the user asks for foods, ingredients, tips, steps, examples, or a meal plan, answer in a structured list or bullets.\n"
        "- If the user asks for a specific count, provide that exact number when supported by the retrieved knowledge.\n"
        "- If the user asks for a meal plan, organize it clearly by meal (breakfast, lunch, dinner, snacks if relevant).\n"
        "- If the user asks 'why' or asks for benefits, give one brief reason per item.\n"
        "- If no relevant knowledge was retrieved, say that clearly and do not pretend the answer came from the document.\n"
        "- Keep the response grounded in the retrieved knowledge, not broad generic health advice.\n"
        "- Only include a healthcare-provider disclaimer when the query involves medical risk, diagnosis, treatment, pregnancy complications, or urgent symptoms.\n"
    )


def build_user_context_block(state: dict) -> str:
    tags = ", ".join(state.get("user_tags", [])) or "none"
    summary = state.get("user_health_summary", {})
    memory = state.get("user_memory", {})

    lines = [
        "--- USER CONTEXT ---",
        f"Health focus: {state.get('use_case', 'nutrition')}",
        f"User tags: {tags}",
    ]

    if summary:
        lines.append(f"Recent health summary: {summary}")
    if memory:
        lines.append(f"Long-term preferences: {memory}")

    lines.append("--- END USER CONTEXT ---")
    return "\n".join(lines)


def build_chunks_block(chunks: list[dict]) -> str:
    if not chunks:
        return (
            "--- RETRIEVED KNOWLEDGE ---\n"
            "No specific knowledge was retrieved for this query.\n"
            "--- END RETRIEVED KNOWLEDGE ---"
        )

    lines = ["--- RETRIEVED KNOWLEDGE ---"]
    for i, chunk in enumerate(chunks, start=1):
        field_type = chunk.get("field_type", "content")
        topic = chunk.get("topic", chunk.get("category", ""))
        text = chunk.get("chunk_text") or chunk.get("text", "")
        similarity = chunk.get("similarity", 0.0)

        lines.append(
            f"[{i}] ({field_type} — {topic}) [similarity: {similarity:.2f}]\n{text}"
        )

    lines.append("--- END RETRIEVED KNOWLEDGE ---")
    return "\n\n".join(lines)


def build_history_block(messages: list[dict]) -> str:
    recent = messages[-6:]
    if not recent:
        return ""

    lines = ["--- CONVERSATION HISTORY ---"]
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        text = msg.get("text", "")
        lines.append(f"{role}: {text}")
    lines.append("--- END CONVERSATION HISTORY ---")
    return "\n".join(lines)


def _build_task_instruction(original_message: str) -> str:
    msg = original_message.lower()

    if any(k in msg for k in ["meal plan", "one-day meal plan", "breakfast", "lunch", "dinner"]):
        return (
            "TASK INSTRUCTION:\n"
            "- Create a practical meal-plan style answer.\n"
            "- Organize by meal.\n"
            "- For each meal, explain briefly how it supports the user's stated goal.\n"
            "- Use bullets, not long paragraphs.\n"
        )

    if any(k in msg for k in ["name any", "list", "foods", "ingredients", "tips", "steps", "examples"]):
        return (
            "TASK INSTRUCTION:\n"
            "- Answer as a bullet list.\n"
            "- If the user asked for a specific number, provide exactly that many items if supported by the retrieved knowledge.\n"
            "- Each bullet should contain the item name plus one short benefit or reason.\n"
            "- Do not open with generic reassurance.\n"
        )

    return (
        "TASK INSTRUCTION:\n"
        "- Answer directly and clearly.\n"
        "- Use short paragraphs or bullets if that improves clarity.\n"
        "- Stay grounded in the retrieved knowledge.\n"
    )


def assemble_prompt(state: dict, chunks: list[dict]) -> str:
    original_message = state["messages"][-1]["text"]

    parts = [
        build_user_context_block(state),
        build_chunks_block(chunks),
        build_history_block(state.get("messages", [])),
        f"--- USER MESSAGE ---\n{original_message}\n--- END MESSAGE ---",
        _build_task_instruction(original_message),
        (
            "FINAL RESPONSE RULES:\n"
            "- Use the retrieved knowledge above as the primary source.\n"
            "- Do not mention internal retrieval mechanics.\n"
            "- Do not mention cluster IDs, similarity, or source labels.\n"
            "- Do not invent foods, benefits, or recommendations that are not supported by the retrieved knowledge.\n"
            "- If the retrieved knowledge is insufficient for the exact request, say so briefly and answer only what is supported.\n"
        ),
    ]

    return "\n\n".join(filter(None, parts))