import re
from ingestion.reader import ContentChunkObject
from config import MAX_FIELD_WORDS


def split_at_sentence(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    mid = len(sentences) // 2
    part1 = ' '.join(sentences[:mid]).strip()
    part2 = ' '.join(sentences[mid:]).strip()
    return [part1, part2]


def prepare_chunks(question_variants: list, content_chunks: list):
    final_questions = question_variants

    final_content = []
    for chunk in content_chunks:
        word_count = len(chunk.text.split())
        if word_count > MAX_FIELD_WORDS:
            parts = split_at_sentence(chunk.text)           #split_at_sentence func call
            for i, part in enumerate(parts):
                new_chunk = ContentChunkObject(
                    text=part,
                    cluster_id=chunk.cluster_id,
                    category=chunk.category,
                    topic=chunk.topic,
                    priority=chunk.priority,
                    field_type="content",
                    reference_sources=chunk.reference_sources,
                )
                final_content.append(new_chunk)
        else:
            final_content.append(chunk)

    return final_questions, final_content