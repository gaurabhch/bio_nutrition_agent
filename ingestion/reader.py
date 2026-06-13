# pipeline_1/reader_md.py
# Parses biocanvas_nutrition_kb.md (or any same-schema .md KB) into
# QuestionObject and ContentChunkObject lists — identical output contract
# as the original reader.py so tagger, embedder, storage need zero changes.
#
# Schema per cluster (bold-delimited, not heading-delimited):
#   **NN. <Topic Title>**
#   | Entry ID | NUTR-XXX-001 |
#   | Cluster  | Cluster NN   |
#   | Category | ...          |
#   | Priority | HIGH/MEDIUM  |
#   **Trigger Questions (Sample User Queries)**
#   - question text
#   **Content**
#   paragraph text ...
#   **References**
#   - *Author — Title. URL*

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class QuestionObject:
    text: str
    cluster_id: str
    category: str
    topic: str
    priority: str
    obj_type: str = "question_variant"


@dataclass
class ContentChunkObject:
    text: str
    cluster_id: str
    category: str
    topic: str
    priority: str
    field_type: str
    reference_sources: List[str] = field(default_factory=list)
    obj_type: str = "content_chunk"


_CLUSTER_TITLE_RE = re.compile(r'^\*\*\d{2}\.\s+(.+?)\*\*\s*$')
_TABLE_ROW_RE = re.compile(r'^\s*\*{2}([^*]+?)\*{2}\s{2,}(.+?)\s*$')
_SECTION_RE       = re.compile(r'^\*\*(.+?)\*\*\s*$')
_URL_RE           = re.compile(r'https?://\S+')
_URL_CLEAN_RE     = re.compile(r'[*./]+$')

_SECTION_TRIGGER    = "trigger questions"
_SECTION_CONTENT    = "content"
_SECTION_REFERENCES = "references"


def _clean_url(raw: str) -> str:
    return _URL_CLEAN_RE.sub("", raw)


def _flush_cluster(cluster: dict, question_variants: list, content_chunks: list) -> None:
    cid      = cluster.get("cluster_id", "")
    category = cluster.get("category", "")
    topic    = cluster.get("topic", "")
    priority = cluster.get("priority", "")
    refs     = cluster.get("refs", [])

    for q in cluster.get("questions", []):
        if q.strip():
            question_variants.append(
                QuestionObject(text=q.strip(), cluster_id=cid, category=category, topic=topic,priority=priority)
            )

    content_text = " ".join(cluster.get("content_lines", [])).strip()
    if content_text:
        content_chunks.append(
            ContentChunkObject(
                text=content_text,
                cluster_id=cid,
                category=category,
                topic=topic,
                priority=priority,
                field_type="content",
                reference_sources=refs,
            )
        )


def read_knowledge_base_md(
    md_path: str,
) -> Tuple[List[QuestionObject], List[ContentChunkObject]]:
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()


    question_variants: List[QuestionObject] = []
    content_chunks: List[ContentChunkObject] = []
    current_cluster: dict | None = None
    current_section: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()

        title_match = _CLUSTER_TITLE_RE.match(line)
        if title_match:
            if current_cluster:
                _flush_cluster(current_cluster, question_variants, content_chunks)
            current_cluster = {
                "topic": title_match.group(1).strip(),
                "cluster_id": "", "category": "", "priority": "",
                "questions": [], "content_lines": [], "refs": [],
            }
            current_section = None
            continue

        if current_cluster is None:
            continue

        table_match = _TABLE_ROW_RE.match(line)
        if table_match:
            key = table_match.group(1).strip().lower()
            val = table_match.group(2).strip()
            if "entry id" in key:
                current_cluster["cluster_id"] = val
            elif "category" in key:
                current_cluster["category"] = val
            elif "priority" in key:
                current_cluster["priority"] = val
            continue

        section_match = _SECTION_RE.match(line)
        if section_match:
            label = section_match.group(1).strip().lower()
            if _SECTION_TRIGGER in label:
                current_section = _SECTION_TRIGGER
            elif label == _SECTION_CONTENT:
                current_section = _SECTION_CONTENT
            elif _SECTION_REFERENCES in label:
                current_section = _SECTION_REFERENCES
            else:
                current_section = None
            continue

        stripped = line.strip()
        if not stripped:
            continue

        if current_section == _SECTION_TRIGGER:
            if stripped.startswith("- "):
                current_cluster["questions"].append(stripped[2:].strip())

        elif current_section == _SECTION_CONTENT:
            clean = re.sub(r'\*+', "", stripped)
            if clean:
                current_cluster["content_lines"].append(clean)

        elif current_section == _SECTION_REFERENCES:
            urls = _URL_RE.findall(stripped)
            current_cluster["refs"].extend(_clean_url(u) for u in urls)

    if current_cluster:
        _flush_cluster(current_cluster, question_variants, content_chunks)

    return question_variants, content_chunks