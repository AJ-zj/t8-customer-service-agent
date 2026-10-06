FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/t8
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt
RUN groupadd --gid 10001 t8 && useradd --uid 10001 --gid t8 --create-home t8
COPY --chown=t8:t8 app ./app
COPY --chown=t8:t8 scripts/14_create_review_table.py scripts/21_create_memory_facts_table.py scripts/32_upgrade_review_publication.py scripts/40_upgrade_memory_lifecycle.py scripts/50_initialize_database.py ./scripts/
COPY --chown=t8:t8 data/processed/t8_pages.jsonl ./data/processed/t8_pages.jsonl
COPY --chown=t8:t8 data/vector_store/chroma ./data/vector_store/chroma
USER t8
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
