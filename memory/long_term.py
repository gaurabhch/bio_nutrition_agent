from __future__ import annotations

import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class LongTermMemory:
    """Reads and writes long-term user memory from PostgreSQL."""

    def __init__(self, session: AsyncSession = None):
        self.session = session

    def _json_value(self, value):
        if value is None:
            return "{}"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    async def ensure_chat_conversation(self, conversation_id: str, user_id: str) -> None:
        await self.session.execute(
            text(
                "INSERT INTO chat_conversations (id, user_id, last_message_at, created_at) "
                "VALUES (:conv_id, :user_id, NOW(), NOW()) "
                "ON CONFLICT (id) DO UPDATE SET last_message_at = NOW(), user_id = EXCLUDED.user_id"
            ),
            {"conv_id": conversation_id, "user_id": user_id},
        )
        await self.session.commit()

    async def get_user_memory(self, user_id: str) -> dict:
        result = await self.session.execute(
            text(
                "SELECT memory_data FROM user_memory "
                "WHERE user_id = :uid LIMIT 1"
            ),
            {"uid": user_id},
        )
        row = result.fetchone()
        return row[0] if row else {}

    async def get_user_profile(self, user_id: str) -> dict:
        result = await self.session.execute(
            text(
                "SELECT use_case, user_tags, health_summary "
                "FROM user_profiles WHERE user_id = :uid LIMIT 1"
            ),
            {"uid": user_id},
        )
        row = result.fetchone()

        if row is None:
            return {
                "use_case": "pcos",
                "user_tags": [],
                "health_summary": {},
            }

        return {
            "use_case": row[0],
            "user_tags": row[1] or [],
            "health_summary": row[2] or {},
        }

    async def update_user_memory(
        self,
        user_id: str,
        conversation_id: str,
        memory_text: str,
        memory_type: str,
        confidence: float,
    ) -> None:
        payload = {
            "memory_text": memory_text,
            "memory_type": memory_type,
            "confidence": confidence,
            "conversation_id": conversation_id,
        }

        await self.session.execute(
            text(
                "INSERT INTO user_memory (user_id, conversation_id, memory_data, updated_at) "
                "VALUES (:user_id, :conversation_id, :memory_data, NOW()) "
                "ON CONFLICT (user_id) DO UPDATE SET "
                "memory_data = EXCLUDED.memory_data, "
                "updated_at = NOW()"
            ),
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "memory_data": self._json_value(payload),
            },
        )
        await self.session.commit()

    async def update_user_profile(
        self,
        user_id: str,
        conversation_id: str,
        user_tags: list | None = None,
        health_summary: dict | None = None,
        use_case: str = "pcos",
    ) -> None:
        tags = user_tags or []
        summary = health_summary or {}
        await self.session.execute(
            text(
                "INSERT INTO user_profiles (user_id, use_case, user_tags, health_summary, created_at, updated_at) "
                "VALUES (:user_id, :use_case, :user_tags, :health_summary, NOW(), NOW()) "
                "ON CONFLICT (user_id) DO UPDATE SET "
                "use_case = EXCLUDED.use_case, "
                "user_tags = EXCLUDED.user_tags, "
                "health_summary = EXCLUDED.health_summary, "
                "updated_at = NOW()"
            ),
            {
                "user_id": user_id,
                "use_case": use_case,
                "user_tags": self._json_value(tags),
                "health_summary": self._json_value(summary),
            },
        )
        await self.session.commit()

    async def save_chat_message(self, state: dict, session: AsyncSession = None) -> None:
        db = session or self.session
        sources_value = self._json_value(state.get("sources", []))

        await db.execute(
            text(
                "INSERT INTO chat_messages "
                "(conversation_id, user_id, user_message, ai_response, "
                " agent_node_used, confidence_score, sources, "
                " is_flagged, flag_reason, created_at) "
                "VALUES "
                "(:conv_id, :user_id, :user_msg, :ai_resp, "
                " :agent_node, :confidence, :sources, "
                " :is_flagged, :flag_reason, NOW())"
            ),
            {
                "conv_id": state["conversation_id"],
                "user_id": state["user_id"],
                "user_msg": state["messages"][-1]["text"],
                "ai_resp": state["final_response"],
                "agent_node": state.get("agent_node_used", ""),
                "confidence": state.get("confidence_score", 0.0),
                "sources": sources_value,
                "is_flagged": state.get("is_flagged", False),
                "flag_reason": state.get("flag_reason"),
            },
        )

        await db.execute(
            text(
                "INSERT INTO chat_conversations (id, user_id, last_message_at, created_at) "
                "VALUES (:conv_id, :user_id, NOW(), NOW()) "
                "ON CONFLICT (id) DO UPDATE SET "
                "last_message_at = NOW(), "
                "user_id = EXCLUDED.user_id"
            ),
            {
                "conv_id": state["conversation_id"],
                "user_id": state["user_id"],
            },
        )

        await db.commit()


long_term_memory = LongTermMemory()