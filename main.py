# main.py
from pathlib import Path
import asyncio
import ssl
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from agents.graph import build_graph
from agents.state import AgentState
from memory.long_term import LongTermMemory
from memory.short_term import short_term_memory
from config import (
    DATABASE_URL_ASYNC,
    GROQ_API_KEY,
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    use_case: str = "nutrition"


class RetrievalDebugItem(BaseModel):
    rank: int
    cluster_id: str
    section_name: str
    topic: str
    similarity: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    agent_node_used: str
    confidence_score: float
    is_crisis: bool
    referred_clusters: list[str]
    retrieval_debug: list[RetrievalDebugItem]
    guardrail_blocked: bool
    flag_reason: Optional[str] = None
    rewritten_query: str
    routing_reason: str


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    app.state.groq_client = AsyncGroq(api_key=GROQ_API_KEY)

    engine = create_async_engine(
        DATABASE_URL_ASYNC,
        pool_size=10,
        max_overflow=20,
        connect_args={"ssl": ssl_ctx},
    )

    app.state.db_session_factory = async_sessionmaker(
        engine, expire_on_commit=False
    )

    app.state.graph = build_graph(
        groq_client=app.state.groq_client,
        db_session_factory=app.state.db_session_factory,
    )

    yield

    await app.state.groq_client.close()
    await short_term_memory.close()
    await engine.dispose()


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="BioCanvas Nutrition AI Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
INDEX_HTML = BASE_DIR / "templates" / "index.html"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_preview(text_value: str, limit: int = 220) -> str:
    text_value = (text_value or "").strip().replace("\n", " ")
    return text_value[:limit]


def _build_retrieval_debug(retrieved_context: list[dict]) -> list[RetrievalDebugItem]:
    debug_items: list[RetrievalDebugItem] = []

    for i, chunk in enumerate(retrieved_context or [], start=1):
        text_value = chunk.get("chunk_text") or chunk.get("text") or ""
        section_name = chunk.get("section_name") or chunk.get("field_type") or ""
        topic = chunk.get("topic") or chunk.get("category") or ""
        similarity = float(chunk.get("similarity", 0.0))

        debug_items.append(
            RetrievalDebugItem(
                rank=i,
                cluster_id=chunk.get("cluster_id", ""),
                section_name=section_name,
                topic=topic,
                similarity=round(similarity, 4),
                preview=_safe_preview(text_value),
            )
        )

    return debug_items


def _build_referred_clusters(retrieved_context: list[dict]) -> list[str]:
    seen = set()
    ordered_clusters = []

    for chunk in retrieved_context or []:
        cid = chunk.get("cluster_id")
        if cid and cid not in seen:
            seen.add(cid)
            ordered_clusters.append(cid)

    return ordered_clusters


async def _build_initial_state(
    request: ChatRequest,
    db_session_factory,
) -> AgentState:
    try:
        history = await short_term_memory.get_history(request.conversation_id)
    except Exception:
        history = []

    try:
        async with db_session_factory() as session:
            ltm = LongTermMemory(session)
            profile = await ltm.get_user_profile(request.user_id)
            memory = await ltm.get_user_memory(request.user_id)
    except Exception:
        profile = {"use_case": request.use_case, "user_tags": [], "health_summary": {}}
        memory = {}

    history.append({"role": "user", "text": request.message})

    return AgentState(
        messages=history,
        user_id=request.user_id,
        use_case=profile.get("use_case", request.use_case),
        user_tags=profile.get("user_tags", []),
        user_health_summary=profile.get("health_summary", {}),
        user_memory=memory,
        conversation_id=request.conversation_id,

        rewritten_query="",
        next_agent="",
        routing_reason="",
        response_mode="",
        is_crisis=False,
        is_flagged=False,
        flag_reason=None,

        retrieved_context=[],
        raw_response="",
        agent_node_used="",
        confidence_score=0.0,

        verified_response="",
        citations=[],

        final_response="",
        sources=[],
    )


async def _persist_after_response(state: AgentState, db_session_factory) -> None:
    try:
        await short_term_memory.append_message(
            state["conversation_id"], "assistant", state["final_response"]
        )
        await short_term_memory.append_message(
            state["conversation_id"], "user", state["messages"][-1]["text"]
        )

        async with db_session_factory() as session:
            ltm = LongTermMemory(session)
            await ltm.save_chat_message(state, session)
    except Exception:
        pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    initial_state = await _build_initial_state(
        request, app.state.db_session_factory
    )

    try:
        final_state: AgentState = await app.state.graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    asyncio.create_task(
        _persist_after_response(final_state, app.state.db_session_factory)
    )

    retrieved_context = final_state.get("retrieved_context", []) or []
    retrieval_debug = _build_retrieval_debug(retrieved_context)
    referred_clusters = _build_referred_clusters(retrieved_context)

    return ChatResponse(
        answer=final_state.get("final_response", ""),
        sources=final_state.get("sources", []),
        agent_node_used=final_state.get("agent_node_used", ""),
        confidence_score=final_state.get("confidence_score", 0.0),
        is_crisis=final_state.get("is_crisis", False),
        referred_clusters=referred_clusters,
        retrieval_debug=retrieval_debug,
        guardrail_blocked=final_state.get("is_flagged", False),
        flag_reason=final_state.get("flag_reason"),
        rewritten_query=final_state.get("rewritten_query", ""),
        routing_reason=final_state.get("routing_reason", ""),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "BioCanvas Nutrition AI Agent"}