from groq import AsyncGroq
from fastembed import TextEmbedding
from sqlalchemy.ext.asyncio import AsyncSession

from agents.state import AgentState
from retrieval.searcher import retrieve_chunks
from utils.prompt_builder import build_system_prompt, assemble_prompt
from config import GROQ_MODEL, GROQ_TIMEOUT, GROQ_TIMEOUT_MESSAGE


class SpecialistAgent:
    def __init__(self, domain: str, embedder: TextEmbedding) -> None:
        self.domain = domain
        self.embedder = embedder
        self._system_prompt = build_system_prompt(domain)

    async def run(
        self,
        state: AgentState,
        groq_client: AsyncGroq,
        db_session: AsyncSession,
    ) -> AgentState:
        query_text = state.get("rewritten_query") or state["messages"][-1]["text"]
        query_embedding = self._embed_query(query_text)

        chunks = await retrieve_chunks(
            query_embedding=query_embedding,
            domain=self.domain,
            session=db_session,
            rewritten_query=state.get("rewritten_query", ""),
        )

        confidence = self._compute_confidence(chunks)
        user_prompt = assemble_prompt(state, chunks)
        raw_response = await self._generate(user_prompt, groq_client)

        return {
            **state,
            "retrieved_context": chunks,
            "raw_response": raw_response,
            "agent_node_used": f"{self.domain}_agent",
            "confidence_score": confidence,
        }

    def _embed_query(self, query: str) -> list[float]:
        vector = next(self.embedder.embed([query]))
        return vector.tolist()

    async def _generate(self, user_prompt: str, groq_client: AsyncGroq) -> str:
        try:
            response = await groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=700,
                timeout=GROQ_TIMEOUT,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return GROQ_TIMEOUT_MESSAGE

    def _compute_confidence(self, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        return round(float(chunks[0].get("similarity", 0.0) or 0.0), 4)