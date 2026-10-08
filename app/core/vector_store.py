import chromadb

from app import config

_client = None


def _collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_or_create_collection(config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def add(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    if ids:
        _collection().upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def query(query_embedding: list[float], k: int = config.RETRIEVE_K) -> list[dict]:
    col = _collection()
    n = col.count()
    if n == 0:
        return []
    res = col.query(query_embeddings=[query_embedding], n_results=min(k, n),
                    include=["documents", "metadatas", "distances", "embeddings"])
    out = []
    for i, cid in enumerate(res["ids"][0]):
        out.append({"embedding_id": cid, "text": res["documents"][0][i], "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i], "embedding": list(res["embeddings"][0][i])})
    return out


def reset() -> None:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        _client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
