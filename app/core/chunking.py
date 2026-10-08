from app import config


def split_into_chunks(text: str, chunk_words: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks
