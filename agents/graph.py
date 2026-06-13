from groq import AsyncGroq
from langgraph.graph import StateGraph, END
from fastembed import TextEmbedding

from agents.state import AgentState
from agents.supervisor import supervisor_node
from agents.specialist_agent import SpecialistAgent
from agents.pubmed_verifier import pubmed_verifier_node
from agents.merge_agent import merge_agent_node
from agents.guardrails.input_guardrail import input_guardrail_node
from agents.guardrails.output_guardrail import output_guardrail_node
from config import GROQ_TIMEOUT, GROQ_MODEL, EMBEDDING_MODEL_FAST


async def clarification_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    message = state["messages"][-1]["text"]
    prompt = (
        "The user sent a message that is too vague to answer specifically: "
        f"\"{message}\"\n\n"
        "Ask exactly one short, friendly follow-up question to clarify what they need. "
        "Do not provide medical advice yet."
    )

    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
            timeout=GROQ_TIMEOUT,
        )
        clarification_q = (response.choices[0].message.content or "").strip()
    except Exception:
        clarification_q = "Could you share a bit more detail about what kind of nutrition help you need?"

    return {
        **state,
        "raw_response": clarification_q,
        "verified_response": clarification_q,
        "retrieved_context": [],
        "confidence_score": 0.0,
        "agent_node_used": "clarification_agent",
    }


def _route_after_input_guardrail(state: AgentState) -> str:
    if not state.get("guardrail_passed", True):
        return "output_guardrail"
    return "supervisor"


def _route_after_supervisor(state: AgentState) -> str:
    use_case = state.get("use_case", "nutrition")
    default_agent = f"{use_case}_general_agent"
    next_agent = state.get("next_agent", default_agent)

    if next_agent == "crisis_end":
        return "merge_agent"

    if next_agent == "clarification_agent":
        return "clarification_agent"

    return next_agent


def build_graph(
    groq_client: AsyncGroq,
    db_session_factory,
    domain_keywords: dict[str, list[str]] | None = None,
) -> StateGraph:
    from config import DOMAIN_KEYWORDS

    active_domains = domain_keywords or DOMAIN_KEYWORDS
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL_FAST)

    specialist_agents: dict[str, SpecialistAgent] = {
        domain: SpecialistAgent(domain=domain, embedder=embedder)
        for domain in active_domains
    }

    async def _input_guardrail(state: AgentState) -> AgentState:
        return await input_guardrail_node(state, groq_client)

    async def _supervisor(state: AgentState) -> AgentState:
        return await supervisor_node(state, groq_client)

    async def _clarification(state: AgentState) -> AgentState:
        return await clarification_node(state, groq_client)

    async def _pubmed(state: AgentState) -> AgentState:
        return await pubmed_verifier_node(state, groq_client)

    async def _merge(state: AgentState) -> AgentState:
        return await merge_agent_node(state, groq_client)

    async def _output_guardrail(state: AgentState) -> AgentState:
        return await output_guardrail_node(state, groq_client)

    def _make_specialist_node(agent: SpecialistAgent):
        async def _node(state: AgentState) -> AgentState:
            async with db_session_factory() as session:
                return await agent.run(state, groq_client, session)
        return _node

    graph = StateGraph(AgentState)

    graph.add_node("input_guardrail", _input_guardrail)
    graph.add_node("output_guardrail", _output_guardrail)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("clarification_agent", _clarification)
    graph.add_node("pubmed_verifier", _pubmed)
    graph.add_node("merge_agent", _merge)

    for domain, agent in specialist_agents.items():
        graph.add_node(f"{domain}_agent", _make_specialist_node(agent))

    graph.set_entry_point("input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        _route_after_input_guardrail,
        {
            "supervisor": "supervisor",
            "output_guardrail": "output_guardrail",
        },
    )

    possible_next_nodes = (
        ["clarification_agent", "merge_agent"]
        + [f"{d}_agent" for d in active_domains]
    )

    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {node: node for node in possible_next_nodes},
    )

    for domain in active_domains:
        graph.add_edge(f"{domain}_agent", "pubmed_verifier")

    graph.add_edge("clarification_agent", "merge_agent")
    graph.add_edge("pubmed_verifier", "merge_agent")
    graph.add_edge("merge_agent", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()