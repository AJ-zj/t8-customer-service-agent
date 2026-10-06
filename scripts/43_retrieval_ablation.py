"""Read-only production retrieval channel ablation, checkpointed per question."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from rag_eval_common import *


def main():
    dataset, records, fingerprint = load_golden()
    before = frozen_files()
    pipeline = retrieval.create_pipeline()
    store, api_key = pipeline[-2:]
    output = RESULTS/'retrieval_ablation_v1.json'
    rows = []
    if output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if previous['golden_sha256'] != fingerprint:
            raise ValueError('Existing results belong to another Golden')
        rows = previous['cases']
    done = {r['id'] for r in rows}
    for case in dataset['cases']:
        if case['id'] in done:
            continue
        row = dict(case, retrieval_query=normalize_retrieval_query(case['question']))
        try:
            rankings, details = evaluate_rankings(row['retrieval_query'], records, pipeline[3], store, api_key)
            row.update(rankings=rankings, details=details, error=None)
            row['retrieved_records'] = {name: [{'record_index': i, **records[i]} for i in ids[:5]]
                                        for name, ids in rankings.items()}
        except Exception as error:
            row.update(rankings={}, error=safe_error(error))
        rows.append(row)
        save(output, {'golden_sha256': fingerprint, 'corpus_sha256': dataset['corpus_sha256'],
          'ranking_definition': 'Hit=any relevant; Recall=fraction of required records; MRR=first relevant in actual candidate list (Reranker max10); NDCG=binary relevance',
          'parameters': {'bm25_min_score': 1, 'channel_limit': 10, 'rrf_k': 60, 'rerank_limit': 10, 'production_top_k': 3},
          'cases': rows, 'summary': {m: aggregate(rows,m) for m in ('BM25','Dense','RRF','Reranker')}})
        print(f'{row["id"]}: {"OK" if row["error"] is None else row["error"]["status"]}', flush=True)
    if frozen_files() != before:
        raise RuntimeError('Frozen production/source files changed')
    print(json.dumps({m:aggregate(rows,m) for m in ('BM25','Dense','RRF','Reranker')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
