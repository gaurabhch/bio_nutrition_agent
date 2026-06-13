import json
import httpx
from groq import AsyncGroq

from agents.state import AgentState
from config import PUBMED_TIMEOUT, GROQ_MODEL, GROQ_TIMEOUT

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

CLAIM_EXTRACTION_PROMPT = """Extract specific, verifiable health or nutrition claims from the text below.
Return ONLY a JSON array of short claim strings.

Examples:
["omega-3 intake may help reduce inflammation", "higher fiber intake may improve satiety"]

If there are no specific verifiable claims, return:
[]

Text:
{text}

Return ONLY the JSON array. No explanation."""

SOFTENING_PROMPT = """Rewrite the response below to soften this unverified claim.

Claim: "{claim}"

Rules:
- Change overly definitive language into cautious, evidence-aware language.
- Keep the rest of the response as unchanged as possible.
- Do NOT add citation labels, source tags, or internal reference markers.
- Return only the full rewritten response text.

Response:
{response}
"""


def _search_pubmed(claim: str) -> list[str]:
    try:
        r = httpx.get(
            f"{PUBMED_BASE}esearch.fcgi",
            params={
                "db": "pubmed",
                "term": claim,
                "retmax": 3,
                "retmode": "json",
            },
            timeout=PUBMED_TIMEOUT,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in ids]
    except Exception:
        return []


async def _extract_claims(text: str, groq_client: AsyncGroq) -> list[str]:
    prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)

    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
            timeout=GROQ_TIMEOUT,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        claims = json.loads(raw)
        return claims if isinstance(claims, list) else []
    except Exception:
        return []


async def _soften_claim(
    response_text: str,
    claim: str,
    groq_client: AsyncGroq,
) -> str:
    prompt = SOFTENING_PROMPT.format(claim=claim, response=response_text)

    try:
        result = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
            timeout=GROQ_TIMEOUT,
        )
        softened = (result.choices[0].message.content or "").strip()
        return softened or response_text
    except Exception:
        return response_text


async def pubmed_verifier_node(
    state: AgentState,
    groq_client: AsyncGroq,
) -> AgentState:
    if state.get("final_response"):
        return state

    raw_response = state.get("raw_response", "")
    if not raw_response:
        return {
            **state,
            "verified_response": raw_response,
            "citations": [],
        }

    claims = await _extract_claims(raw_response, groq_client)
    verified = raw_response
    citations: list[str] = []

    for claim in claims:
        urls = _search_pubmed(claim)
        if urls:
            citations.extend(urls)
        else:
            verified = await _soften_claim(verified, claim, groq_client)

    citations = list(dict.fromkeys(citations))[:2]

    return {
        **state,
        "verified_response": verified,
        "citations": citations,
    }