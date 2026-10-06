"""Isolated character-window experiment; never overwrites production corpus/index."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from concurrent.futures import ThreadPoolExecutor
from rag_eval_common import *
from app.retrieval.t8_vector_store import T8VectorStore

CONFIGS = [(256,32),(256,64),(512,32),(512,96),(768,96)]
BASE = ROOT/'data/evaluation/chunk_ablation'


def split_records(records, size, overlap):
    if size < 1 or not 0 <= overlap < size:
        raise ValueError('Invalid chunk window')
    chunks=[]
    for parent, record in enumerate(records):
        text=record['text']
        for start in range(0,len(text),size-overlap):
            end=min(start+size,len(text))
            chunks.append({**record,'text':text[start:end], 'parent_record_index':parent,
                           'start_char':start,'end_char':end})
            if end==len(text):
                break
    return chunks


def parent_ranking(chunk_ids, chunks):
    # One knowledge record cannot receive multiple rank slots just because it was split.
    return list(dict.fromkeys(chunks[i]['parent_record_index'] for i in chunk_ids))


def chunk_slot_metrics(chunk_ids, chunks, relevant):
    # Preserve actual chunk rank slots. A repeated parent gets zero additional gain,
    # rather than being removed and incorrectly moving later answers forward.
    seen=set()
    slots=[]
    for rank,i in enumerate(chunk_ids,1):
        parent=chunks[i]['parent_record_index']
        slots.append(parent if parent not in seen else -rank)
        seen.add(parent)
    return ranking_metrics(slots,relevant)


def evidence_coverage(case, chunk_ids, chunks, records):
    fractions=[]
    for proof in case.get('source_evidence',[]):
        parent=proof['record_index']
        start=records[parent]['text'].index(proof['quote'])
        end=start+len(proof['quote'])
        covered=set()
        for i in chunk_ids:
            c=chunks[i]
            if c['parent_record_index']==parent:
                covered.update(range(max(start,c['start_char']),min(end,c['end_char'])))
        fractions.append(len(covered)/(end-start))
    return mean(fractions) if fractions else None


def main():
    dataset, records, fingerprint=load_golden()
    before=frozen_files()
    key= retrieval.get_siliconflow_api_key()
    BASE.mkdir(parents=True,exist_ok=True)
    cases=[c for c in dataset['cases'] if c['answerable']]
    queries=[normalize_retrieval_query(c['question']) for c in cases]
    query_path=BASE/'query_vectors.json'
    if query_path.exists():
        cached=json.loads(query_path.read_text(encoding='utf-8'))
        if cached['golden_sha256']!=fingerprint or cached['model']!=retrieval.EMBEDDING_MODEL or cached['queries']!=queries:
            raise ValueError('Query cache mismatch')
        query_vectors=retrieval.validate_vectors(cached['vectors'],len(cases))
    else:
        query_vectors=retrieval.request_embeddings(queries,key)
        save(query_path,{'golden_sha256':fingerprint,'model':retrieval.EMBEDDING_MODEL,'queries':queries,'vectors':query_vectors})
    all_configs=[]
    for size,overlap in CONFIGS:
        path=BASE/f'chars_{size}_overlap_{overlap}'
        path.mkdir(exist_ok=True)
        chunks=split_records(records,size,overlap)
        corpus_bytes=('\n'.join(json.dumps(c,ensure_ascii=False) for c in chunks)+'\n').encode()
        corpus_hash=sha(corpus_bytes)
        (path/'chunks.jsonl').write_bytes(corpus_bytes)
        vector_path=path/'document_vectors.json'
        if vector_path.exists():
            saved=json.loads(vector_path.read_text(encoding='utf-8'))
            if saved['corpus_sha256']!=corpus_hash or saved['model']!=retrieval.EMBEDDING_MODEL:
                raise ValueError('Chunk cache mismatch')
            vectors=retrieval.validate_vectors(saved['vectors'],len(chunks))
        else:
            vectors=retrieval.request_embeddings([retrieval.embedding_text(c) for c in chunks],key)
            save(vector_path,{'model':retrieval.EMBEDDING_MODEL,'corpus_sha256':corpus_hash,'vectors':vectors})
        index=retrieval.build_bm25_index(chunks)
        isolated=path/'chroma'
        if (isolated/'chroma.sqlite3').exists():
            store=T8VectorStore(chunks,corpus_hash,isolated)
        else:
            store=T8VectorStore.build_index(chunks,corpus_hash,vectors,isolated)
        result_path=path/'results.json'
        rows=json.loads(result_path.read_text(encoding='utf-8'))['cases'] if result_path.exists() else []
        done={r['id'] for r in rows}
        def run(item):
            case,query,vector=item
            row=dict(case,retrieval_query=query)
            try:
                ranks,details=evaluate_rankings(query,chunks,index,store,key,vector)
                row.update(rankings={m:parent_ranking(ids,chunks) for m,ids in ranks.items()},
                    chunk_rankings=ranks,details=details,error=None,
                    reference_coverage_top3=evidence_coverage(case,ranks['Reranker'][:3],chunks,records))
                # Top3 refers to three actual chunks, not three parents collected from a longer list.
                row['actual_chunk_top3_parent_hit']=int(bool(set(parent_ranking(ranks['Reranker'][:3],chunks)) & set(case['relevant_record_indices'])))
            except Exception as error:
                row.update(rankings={},error=safe_error(error))
            return row
        pending=[item for item in zip(cases,queries,query_vectors) if item[0]['id'] not in done]
        with ThreadPoolExecutor(max_workers=3) as executor:
            for row in executor.map(run,pending):
                rows.append(row)
                save(result_path,{'golden_sha256':fingerprint,'chunk_size':size,'overlap':overlap,'chunk_count':len(chunks),'cases':rows})
                print(f'{size}/{overlap} {row["id"]}: {"OK" if not row["error"] else "ERROR"}',flush=True)
        good=[r for r in rows if r.get('error') is None]
        summary={'chunk_size':size,'overlap':overlap,'unit':'Unicode characters','chunk_count':len(chunks),
          'parent_record_proxy':aggregate(rows,'Reranker'),
          'actual_chunk_top3_parent_hit':mean(r['actual_chunk_top3_parent_hit'] for r in good) if good else None,
          'reference_coverage_top3':mean(r['reference_coverage_top3'] for r in good) if good else None,
          'successful':len(good),'failed':len(rows)-len(good)}
        slot_scores=[chunk_slot_metrics(r['chunk_rankings']['Reranker'],chunks,r['relevant_record_indices']) for r in good]
        summary['actual_chunk_slot_metrics']={k:mean(s[k] for s in slot_scores) for k in slot_scores[0]} if slot_scores else {}
        all_configs.append(summary)
        save(RESULTS/'chunk_ablation_v1.json',{'golden_sha256':fingerprint,
          'warning':'Record-projected Hit/Recall are proxies: partial chunks may not contain full answer. Actual chunk Top3 evidence coverage is reported separately. No production optimum claim.',
          'configs':all_configs})
    if frozen_files()!=before:
        raise RuntimeError('Frozen files changed')
    print(json.dumps(all_configs,ensure_ascii=False))

if __name__=='__main__':
    main()
