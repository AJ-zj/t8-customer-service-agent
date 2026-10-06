"""Local readiness only: no model calls, index rebuilds or schema writes."""
from sqlalchemy import inspect, text

from app.models import QAReviewRecord, AgentMemoryFact


class ReadinessError(RuntimeError):
    pass


def check_readiness():
    try:
        from app.database import engine, configured_database_name
        if configured_database_name() != "roborock":
            raise ValueError("Wrong database")
        with engine.connect() as connection:
            if connection.scalar(text("SELECT current_database()")) != "roborock":
                raise ValueError("Wrong database")
            inspector = inspect(connection)
            for model in (QAReviewRecord, AgentMemoryFact):
                table = model.__table__
                if not inspector.has_table(table.name):
                    raise ValueError("Missing table")
                names = {column["name"] for column in inspector.get_columns(table.name)}
                if not set(table.columns.keys()) <= names:
                    raise ValueError("Incomplete schema")
        from app.retrieval.t8_retrieval import load_corpus
        from app.retrieval.t8_vector_store import T8VectorStore
        records, fingerprint = load_corpus()
        if len(records) != 41:
            raise ValueError("Incomplete corpus")
        store = T8VectorStore(records, fingerprint)
        report = store.validate_collection()
        if report["stored_record_count"] != 41 or report["embedding_dimension"] != 1024:
            raise ValueError("Incomplete index")
    except Exception:
        raise ReadinessError("本地依赖尚未就绪，请检查数据库初始化和说明书索引。") from None
