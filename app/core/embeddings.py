from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
