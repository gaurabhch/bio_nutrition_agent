from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any

from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

_STABLE_PATTERNS = {
    "preference": [
        r"i prefer",
        r"i like",
        r"i usually",
        r"i avoid",
        r"my preference",
    ],
    "recurring_concern": [
        r"again",
        r"keeping happening",
        r"keeps happening",
        r"always",
        r"frequently",
        r"often",
    ],
    "profile_signal": [
        r"i am \d{2}",
        r"i'm \d{2}",
        r"my age is \d{2}",
        r"diagnosed with",
        r"taking",
        r"not taking",
    ],
}

_LOW_VALUE_PATTERNS = [
    r"what is",
    r"how to",
    r"is it normal",
    r"could this be",
    r"should i",
]

@dataclass
class MemoryCandidate:
    memory_text: str
    memory_type: str
    confidence: float
    should_update_profile: bool = False


def _matches(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def _high_signal_user_fact(message: str) -> bool:
    t = message.lower()
    return any(k in t for k in [
        "i prefer", "i like", "i avoid", "my age is", "i am ", "i'm ",
        "i have been diagnosed", "i was diagnosed", "i take", "i don't take",
        "i cannot", "i can only", "i usually", "my symptoms are",
    ])


def extract_memory_candidates(user_message: str, assistant_response: str, state: dict[str, Any]) -> list[MemoryCandidate]:
    text = user_message.strip()
    if len(text) < 12:
        return []

    lower = text.lower()

    if _matches(lower, _LOW_VALUE_PATTERNS) and not _high_signal_user_fact(lower):
        return []

    candidates: list[MemoryCandidate] = []

    if _matches(lower, _STABLE_PATTERNS["preference"]):
        candidates.append(MemoryCandidate(
            memory_text=text,
            memory_type="preference",
            confidence=0.86,
            should_update_profile=True,
        ))

    if _matches(lower, _STABLE_PATTERNS["recurring_concern"]):
        candidates.append(MemoryCandidate(
            memory_text=text,
            memory_type="recurring_concern",
            confidence=0.8,
            should_update_profile=False,
        ))

    if _matches(lower, _STABLE_PATTERNS["profile_signal"]):
        candidates.append(MemoryCandidate(
            memory_text=text,
            memory_type="profile_signal",
            confidence=0.92,
            should_update_profile=True,
        ))

    if not candidates and any(k in lower for k in ["weight", "period", "periods", "acne", "hair fall", "hair loss", "fertility", "pregnant", "pregnancy"]):
        candidates.append(MemoryCandidate(
            memory_text=text,
            memory_type="health_context",
            confidence=0.62,
            should_update_profile=False,
        ))

    if state.get("response_mode") == "EMOTIONAL" and candidates:
        candidates = [c for c in candidates if c.confidence >= 0.8]

    return candidates


async def persist_long_term_memory(
    user_id: str,
    conversation_id: str,
    latest_user_message: str,
    assistant_response: str,
    state: dict[str, Any],
    db_session_factory,
) -> None:
    try:
        candidates = extract_memory_candidates(latest_user_message, assistant_response, state)
        if not candidates:
            logger.info("memory_extractor: no durable candidates for conversation_id=%s", conversation_id)
            return

        async with db_session_factory() as session:
            ltm = LongTermMemory(session)
            for candidate in candidates:
                await ltm.update_user_memory(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_text=candidate.memory_text,
                    memory_type=candidate.memory_type,
                    confidence=candidate.confidence,
                )

            profile_updates = [c for c in candidates if c.should_update_profile and c.confidence >= 0.85]
            if profile_updates:
                await ltm.update_user_profile(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    user_tags=state.get("user_tags", []),
                    health_summary={
                        "last_durable_signal": profile_updates[-1].memory_text,
                        "signal_type": profile_updates[-1].memory_type,
                    },
                )
        logger.info("memory_extractor: persisted %d candidate(s) for conversation_id=%s", len(candidates), conversation_id)
    except Exception:
        logger.exception("memory_extractor: persistence failed for conversation_id=%s", conversation_id)
