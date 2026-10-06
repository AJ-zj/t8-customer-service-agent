"""Small adapter around the existing production Embedding client."""

from app.config import get_siliconflow_api_key
from app.retrieval.t8_retrieval import request_embeddings


def embed_fact(text: str) -> list[float]:
    return request_embeddings([text], get_siliconflow_api_key())[0]
