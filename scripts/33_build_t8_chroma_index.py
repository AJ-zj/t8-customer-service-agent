"""Build Chroma using verified Qwen document cache, or the same provider if absent."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_siliconflow_api_key
from app.retrieval import t8_retrieval as retrieval
from app.retrieval.t8_vector_store import T8VectorStore


def main():
    try:
        records, corpus_hash = retrieval.load_corpus()
        try:
            vectors = retrieval.load_cached_document_vectors(records, corpus_hash)
            reused = True
        except RuntimeError:
            vectors = retrieval.request_embeddings(
                [retrieval.embedding_text(record) for record in records], get_siliconflow_api_key())
            reused = False
        store = T8VectorStore.build_index(records, corpus_hash, vectors)
        print(json.dumps({**store.validate_collection(), "reused_document_cache": reused},
                         ensure_ascii=False, indent=2))
        return 0
    except Exception:
        print("建索引失败：请核对知识库、Embedding 缓存、API 配置与 Chroma 依赖；未输出敏感异常。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
