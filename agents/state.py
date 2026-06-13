from typing import TypedDict, Optional


class AgentState(TypedDict):
    # ── INPUT ──────────────────────────────────────────────────────────────
    messages: list[dict]
    user_id: str
    use_case: str
    user_tags: list[str]
    user_health_summary: dict
    user_memory: dict
    conversation_id: str

    # ── INPUT GUARDRAIL WRITES ─────────────────────────────────────────────
    guardrail_passed: bool
    guardrail_block_reason: Optional[str]
    sanitized_query: str
    pii_redacted: bool
    implicit_health_context: bool

    # ── SUPERVISOR WRITES ──────────────────────────────────────────────────
    rewritten_query: str
    next_agent: str
    routing_reason: str
    response_mode: str
    is_crisis: bool
    is_flagged: bool
    flag_reason: Optional[str]

    # ── SPECIALIST WRITES ──────────────────────────────────────────────────
    retrieved_context: list[dict]
    raw_response: str
    agent_node_used: str
    confidence_score: float

    # ── VERIFIER WRITES ────────────────────────────────────────────────────
    verified_response: str
    citations: list[str]

    # ── MERGE / FINAL RESPONSE WRITES ─────────────────────────────────────
    final_response: str
    sources: list[str]