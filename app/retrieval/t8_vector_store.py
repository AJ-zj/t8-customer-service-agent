"""Persistent Chroma Dense retrieval only; embeddings come from the existing provider."""

import math
from pathlib import Path

from app.config import PROJECT_ROOT, get_siliconflow_api_key

PERSIST_PATH = PROJECT_ROOT / "data/vector_store/chroma"
COLLECTION_NAME = "t8_manual_qwen3_embedding_0_6b"


class VectorStoreError(RuntimeError):
    """Safe errors; never include backend responses or credentials."""


def _client(path):
    try:
        import chromadb
        from chromadb.config import Settings
        return chromadb.PersistentClient(path=str(Path(path).resolve()),
                                         settings=Settings(anonymized_telemetry=False))
    except Exception:
        raise VectorStoreError("Chroma 持久化库不可用，请检查依赖和索引目录。") from None


def _close_client(client):
    try:
        client.close()
    except Exception:
        pass  # Cleanup must not replace the operation's original outcome.


def _record_metadata(index, record):
    if (not isinstance(record.get("text"), str) or not record["text"].strip()
            or type(record.get("pdf_page")) is not int or record["pdf_page"] < 1):
        raise VectorStoreError("知识记录正文或 PDF 页码无效。")
    for key in ("title", "section"):
        if record.get(key) is not None and not isinstance(record[key], str):
            raise VectorStoreError("知识记录标题或章节无效。")
    return {"record_index": index, "pdf_page": record["pdf_page"],
            "title": record.get("title") or "", "section": record.get("section") or ""}


def _checked_vectors(vectors, count):
    from app.retrieval.t8_retrieval import validate_vectors
    try:
        checked = validate_vectors(vectors, count)
        if any(not any(value != 0 for value in vector) for vector in checked):
            raise ValueError("zero vector")
        return checked
    except Exception:
        raise VectorStoreError("向量数量、维度或数值无效；未构建或查询索引。") from None


class T8VectorStore:
    def __init__(self, records, corpus_hash, persist_path=PERSIST_PATH):
        self.records = records
        self.corpus_hash = corpus_hash
        self.persist_path = Path(persist_path).resolve()
        self.client = _client(self.persist_path)
        try:
            self.collection = self.client.get_collection(
                COLLECTION_NAME, embedding_function=None)
            self.validate_collection()
        except VectorStoreError:
            _close_client(self.client)
            raise
        except Exception:
            _close_client(self.client)
            raise VectorStoreError("Chroma collection 不存在或不可读取，请先运行建索引脚本。") from None

    @classmethod
    def build_index(cls, records, corpus_hash, vectors, persist_path=PERSIST_PATH):
        from app.retrieval.t8_retrieval import EMBEDDING_MODEL
        if not records:
            raise VectorStoreError("知识库为空。")
        metadatas = [_record_metadata(i, record) for i, record in enumerate(records)]
        vectors = _checked_vectors(vectors, len(records))
        dimension = len(vectors[0])
        client = _client(persist_path)
        metadata = {"embedding_model": EMBEDDING_MODEL, "corpus_sha256": corpus_hash,
                    "embedding_dimension": dimension}
        try:
            collection = client.get_or_create_collection(
                COLLECTION_NAME, embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}}, metadata=metadata)
            saved = collection.metadata or {}
            if (saved.get("embedding_model") != EMBEDDING_MODEL
                    or saved.get("embedding_dimension") != dimension
                    or collection.configuration["hnsw"]["space"] != "cosine"):
                raise VectorStoreError("已有 collection 模型、维度或距离配置不匹配；停止写入。")
            ids = [f"t8_{i}" for i in range(len(records))]
            collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas,
                              documents=[record["text"] for record in records])
            stale = set(collection.get(include=[])["ids"]) - set(ids)
            if stale:
                collection.delete(ids=sorted(stale))
            collection.modify(metadata=metadata)
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError("Chroma 索引构建失败，请检查本地存储。") from None
        finally:
            _close_client(client)
        return cls(records, corpus_hash, persist_path)

    def validate_collection(self):
        from app.retrieval.t8_retrieval import EMBEDDING_MODEL
        try:
            metadata = self.collection.metadata or {}
            if (not self.records or self.collection.count() != len(self.records)
                    or metadata.get("embedding_model") != EMBEDDING_MODEL
                    or metadata.get("corpus_sha256") != self.corpus_hash
                    or self.collection.configuration["hnsw"]["space"] != "cosine"):
                raise VectorStoreError("Chroma 索引数量、模型、指纹或距离配置与知识库不一致。")
            rows = self.collection.get(include=["documents", "metadatas", "embeddings"])
            ids, metas, docs = rows["ids"], rows["metadatas"], rows["documents"]
            if not (len(ids) == len(metas) == len(docs) == len(self.records)):
                raise VectorStoreError("Chroma 索引记录不完整。")
            seen = set()
            for record_id, meta, doc in zip(ids, metas, docs):
                index = meta.get("record_index") if isinstance(meta, dict) else None
                if (type(index) is not int or not 0 <= index < len(self.records)
                        or index in seen or record_id != f"t8_{index}"
                        or meta != _record_metadata(index, self.records[index])
                        or doc != self.records[index]["text"]):
                    raise VectorStoreError("Chroma 记录 ID、metadata 或文本与知识库不一致。")
                seen.add(index)
            vectors = _checked_vectors(rows["embeddings"].tolist(), len(self.records))
            self.dimension = len(vectors[0])
            if metadata.get("embedding_dimension") != self.dimension:
                raise VectorStoreError("Chroma 索引向量维度与 metadata 不一致。")
            return {"collection_name": COLLECTION_NAME, "source_record_count": len(self.records),
                    "stored_record_count": len(ids), "embedding_dimension": self.dimension,
                    "persist_path": str(self.persist_path)}
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError("Chroma 索引完整性检查失败。") from None

    def search_dense(self, question, limit=10):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("检索问题不能为空。")
        self._check_limit(limit)
        from app.retrieval.t8_retrieval import request_embeddings
        vectors = request_embeddings([question], get_siliconflow_api_key())
        return self.search_by_vector(vectors[0] if len(vectors) == 1 else [], limit)

    def _check_limit(self, limit):
        if type(limit) is not int or not 1 <= limit <= len(self.records):
            raise ValueError("limit 必须为 1 到知识记录数量之间的整数。")

    def search_by_vector(self, query_vector, limit=10):
        self._check_limit(limit)
        vector = _checked_vectors([query_vector], 1)[0]
        if len(vector) != self.dimension:
            raise VectorStoreError("查询向量维度与 Chroma 索引不一致。")
        try:
            if self.collection.count() != len(self.records):
                raise VectorStoreError("Chroma 索引数量发生变化，请重新核验索引。")
            response = self.collection.query(query_embeddings=[vector], n_results=limit,
                                             include=["documents", "metadatas", "distances"])
            ids, metas = response["ids"][0], response["metadatas"][0]
            docs, distances = response["documents"][0], response["distances"][0]
            if not (len(ids) == len(metas) == len(docs) == len(distances) == limit):
                raise VectorStoreError("Chroma 查询返回数量异常。")
            results, seen = [], set()
            for rid, meta, doc, distance in zip(ids, metas, docs, distances):
                index = meta.get("record_index") if isinstance(meta, dict) else None
                if (type(index) is not int or not 0 <= index < len(self.records)
                        or index in seen or rid != f"t8_{index}"
                        or meta != _record_metadata(index, self.records[index])
                        or doc != self.records[index]["text"] or not math.isfinite(distance)):
                    raise VectorStoreError("Chroma 查询返回记录或距离无效。")
                seen.add(index)
                results.append({**self.records[index], "record_index": index,
                                "distance": float(distance), "dense_score": 1 - float(distance)})
            return results
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError("Chroma Dense 查询失败；未回退旧检索。") from None
