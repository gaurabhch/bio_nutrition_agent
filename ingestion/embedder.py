from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, BATCH_SIZE


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_batch(self, texts: list) -> list:
        return self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()

    def embed_all(self, question_variants: list, content_chunks: list):
        qtexts = [q.text for q in question_variants]
        ctexts = [c.text for c in content_chunks]

        qvecs = self.embed_batch(qtexts)
        cvecs = self.embed_batch(ctexts)

        for q, vec in zip(question_variants, qvecs):
            q.embedding = vec

        for c, vec in zip(content_chunks, cvecs):
            c.embedding = vec

        return question_variants, content_chunks