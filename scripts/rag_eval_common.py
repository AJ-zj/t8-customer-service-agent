"""Offline evaluation helpers; no Agent, review or database write entry points."""
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean

from app.config import PROJECT_ROOT
from app.retrieval import t8_retrieval as retrieval
from app.services.qa_service import normalize_retrieval_query

ROOT = PROJECT_ROOT
GOLDEN = ROOT / 'data/evaluation/t8_rag_golden_v1.json'
RESULTS = ROOT / 'data/evaluation/results'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate_golden(dataset, records):
    ids = set()
    for case in dataset['cases']:
        if case['id'] in ids or not isinstance(case['question'], str) or not case['question'].strip():
            raise ValueError('Duplicate id or empty question')
        ids.add(case['id'])
        indices = case['relevant_record_indices']
        if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(records) for i in indices):
            raise ValueError('Invalid relevant record index')
        if type(case['answerable']) is not bool:
            raise ValueError('answerable must be bool')
        if case['answerable']:
            if not indices or not isinstance(case['reference_answer'], str) or not case['reference_answer'].strip():
                raise ValueError('Missing reference/evidence')
            proofs = case.get('source_evidence', [])
            if set(p['record_index'] for p in proofs) != set(indices):
                raise ValueError('Evidence does not cover labels')
            for proof in proofs:
                record = records[proof['record_index']]
                if proof['pdf_page'] != record['pdf_page'] or proof['text_sha256'] != sha(record['text'].encode()):
                    raise ValueError('Source mismatch')
                if not proof['quote'] or proof['quote'] not in record['text']:
                    raise ValueError('Source quote mismatch')
            if case['reference_answer'] != '\n'.join(p['quote'] for p in proofs):
                raise ValueError('Reference must consist only of verified source quotes')
        elif indices or case['reference_answer'] is not None or case.get('source_evidence'):
            raise ValueError('Unanswerable has invented evidence')
    return True


def load_golden():
    raw = GOLDEN.read_bytes()
    dataset = json.loads(raw)
    records, corpus_hash = retrieval.load_corpus()
    if dataset['corpus_sha256'] != corpus_hash:
        raise ValueError('Corpus fingerprint changed')
    pdf = ROOT / dataset['pdf_path']
    if sha(pdf.read_bytes()) != dataset['pdf_sha256']:
        raise ValueError('PDF fingerprint changed')
    if GOLDEN.with_suffix('.sha256').read_text().strip() != sha(raw):
        raise ValueError('Frozen Golden fingerprint changed')
    validate_golden(dataset, records)
    return dataset, records, sha(raw)


def ranking_metrics(ids, relevant):
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate ranked record')
    relevant = set(relevant)
    if not relevant:
        raise ValueError('Retrieval metrics require answerable evidence')
    result = {}
    for k in (1, 3, 5):
        hits = sum(i in relevant for i in ids[:k])
        result[f'hit@{k}'] = int(hits > 0)
        if k != 1:
            result[f'recall@{k}'] = hits / len(relevant)
            dcg = sum(1 / math.log2(rank + 1) for rank, i in enumerate(ids[:k], 1) if i in relevant)
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
            result[f'ndcg@{k}'] = dcg / idcg
    result['mrr'] = next((1 / rank for rank, i in enumerate(ids, 1) if i in relevant), 0.0)
    return result


def aggregate(rows, method):
    eligible = [r for r in rows if r['answerable']]
    successful = [r for r in eligible if r.get('rankings', {}).get(method) is not None]
    scores = [ranking_metrics(r['rankings'][method], r['relevant_record_indices']) for r in successful]
    return {'eligible': len(eligible), 'successful': len(scores), 'failed': len(eligible)-len(scores),
            **({k: mean(s[k] for s in scores) for k in scores[0]} if scores else {})}


def evaluate_rankings(query, records, index, store, api_key, query_vector=None):
    scores, ranks, bm_candidates = retrieval.rank_bm25(query, records, index)
    bm_ids = sorted(ranks, key=ranks.get)
    # Zero/sub-threshold BM25 records are not returned as reliable retrieval results.
    bm_ids = [i for i in bm_ids if scores[i] >= retrieval.BM25_MIN_SCORE]
    dense = store.search_by_vector(query_vector, len(records)) if query_vector is not None else store.search_dense(query, len(records))
    dense_ids = [r['record_index'] for r in dense]
    fused = retrieval.reciprocal_rank_fusion(bm_candidates, [(i, rank) for rank, i in enumerate(dense_ids[:10], 1)])
    rrf_ids = [i for i, *_ in fused]
    candidates = [(i, records[i]) for i in rrf_ids[:retrieval.RERANK_CANDIDATE_LIMIT]]
    reranked = retrieval.request_rerank(query, candidates, api_key)
    return {'BM25': bm_ids, 'Dense': dense_ids, 'RRF': rrf_ids,
            'Reranker': [i for i, _ in reranked]}, {'bm25_scores': [float(x) for x in scores],
              'dense_scores': [r['dense_score'] for r in dense], 'rrf': fused, 'reranker': reranked}


def citation_check(answer, evidence_ids):
    matches = re.findall(r'\[E(\d+)\]|【E(\d+)】', answer)
    ids = list(dict.fromkeys('E' + (a or b) for a, b in matches))
    invalid = [i for i in ids if i not in evidence_ids]
    # Detect citation-like strings not accepted by the exact supported formats.
    malformed = [s for s in re.findall(r'[\[【][^\]】\n]*[Ee][^\]】\n]*[\]】]', answer)
                 if not re.fullmatch(r'\[E\d+\]|【E\d+】', s)]
    return {'ids': ids, 'invalid': invalid, 'malformed': malformed,
            'has_citation': bool(ids), 'number_valid': not invalid and not malformed}


def strict_abstention(answer):
    # Full-string refusal only. A refusal phrase followed by facts is NOT accepted.
    return bool(re.fullmatch(r'根据当前提供的说明书资料无法确认[。.!！]?\s*', answer.strip()))


def frozen_files():
    paths = list((ROOT/'app').rglob('*.py')) + [retrieval.CORPUS, retrieval.EMBEDDING_CACHE,
        ROOT/'data/evaluation/t8_retrieval_candidates_v1.json',
        ROOT/'data/evaluation/t8_query_normalization_regression_v1.json', ROOT/'requirements.txt']
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}


def safe_error(error):
    # Class only: upstream bodies, credentials and request data never persisted.
    name = type(error).__name__
    timeout = isinstance(error, TimeoutError) or 'timeout' in name.lower()
    return {'type': name, 'status': 'timeout' if timeout else 'failure'}
