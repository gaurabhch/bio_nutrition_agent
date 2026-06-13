from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def to_pg_array(lst):
    lst = lst or []
    if not lst:
        return "{}"
    return "{" + ",".join(f'"{str(s)}"' for s in lst) + "}"

def store_all(question_variants: list, content_chunks: list, db_session):
    all_objects = question_variants + content_chunks

    rows = [
        {
            "cluster_id":        o.cluster_id,
            "category":          o.category,
            "topic":             o.topic,
            "priority":          getattr(o, "priority", None),
            # "field_type":        getattr(o, "field_type", None),
            "chunk_text":        o.text,
            "embedding":         "[" + ",".join(map(str, o.embedding)) + "]",
            "reference_sources": to_pg_array(getattr(o, "reference_sources", [])),
        }
        for o in all_objects
    ]

    db_session.execute(
        text("""
            INSERT INTO nutrition_knowledge_base 
            (cluster_id, category, topic,
            chunk_text, embedding, reference_sources)
        VALUES
            (:cluster_id, :category, :topic,
            :chunk_text, CAST(:embedding AS vector), CAST(:reference_sources AS text[]))
        """),
        rows
    )
    db_session.commit()
    print(f"Stored {len(rows)} rows in nutrition_knowledge_base")



def build_hnsw_index(db_session):
    db_session.execute(text("""
        CREATE INDEX IF NOT EXISTS nutrition_knowledge_base_hnsw_idx ON
        nutrition_knowledge_base
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """))
    db_session.commit()
    print("HNSW index built successfully")