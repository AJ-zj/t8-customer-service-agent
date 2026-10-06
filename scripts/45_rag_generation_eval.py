"""Draft-only evaluation. Uses Grounded QA generator/audit without any service write."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
from concurrent.futures import ThreadPoolExecutor
from rag_eval_common import *
from app.services import grounded_qa as qa


def evaluate_draft(case, answer, evidence):
    citations=citation_check(answer,{e[0] for e in evidence})
    audit=qa.audit_answer(case['question'],answer,evidence)
    exact_refusal=strict_abstention(answer)
    valid=citations['number_valid'] and (citations['has_citation'] or exact_refusal)
    return {'citations':citations,'citation_validity':valid,
        'audit_status':audit['status'],'semantic_support':'NOT_VERIFIED_BY_DETERMINISTIC_CHECK',
        'strict_abstention_correct':exact_refusal if not case['answerable'] else None,
        'abstention_semantic_review_needed':not case['answerable'] and not exact_refusal,
        'bad_cases':(['missing citation'] if not citations['has_citation'] and not exact_refusal else [])+
          (['unsupported claim'] if audit['status'].startswith('PROGRAM_BLOCKED_UNSUPPORTED') else [])+
          (['unnecessary abstention'] if case['answerable'] and exact_refusal else [])}


def main():
    dataset,records,fingerprint=load_golden()
    before=frozen_files()
    retrieval_results=json.loads((RESULTS/'retrieval_ablation_v1.json').read_text(encoding='utf-8'))
    if retrieval_results['golden_sha256']!=fingerprint:
        raise ValueError('Retrieval Golden mismatch')
    by_id={r['id']:r for r in retrieval_results['cases']}
    output=RESULTS/'generation_eval_v1.json'
    rows=[]
    if output.exists():
        old=json.loads(output.read_text(encoding='utf-8'))
        if old['golden_sha256']!=fingerprint:
            raise ValueError('Generation Golden mismatch')
        rows=old['cases']
    done={r['id'] for r in rows}
    key=retrieval.get_siliconflow_api_key()
    def run(case):
        row=dict(case)
        r=by_id[case['id']]
        ids=r.get('rankings',{}).get('Reranker')
        if ids is None:
            return dict(row,error={'status':'failure','type':'RetrievalFailure'},bad_cases=['retrieval/API failure'])
        evidence=[(f'E{rank}',i,records[i]) for rank,i in enumerate(ids[:3],1)]
        row['retrieved_records']=[{'evidence_id':e,'record_index':i,**rec} for e,i,rec in evidence]
        row['contexts']=[rec['text'] for _,_,rec in evidence]
        row['rankings']=r['rankings']
        start=time.monotonic()
        try:
            answer=qa.generate_answer(case['question'],evidence,key)
            row.update(answer=answer,elapsed_seconds=time.monotonic()-start,error=None,
                       **evaluate_draft(case,answer,evidence))
            if case['answerable'] and not ranking_metrics(ids,case['relevant_record_indices'])['hit@3']:
                row['bad_cases'].append('retrieval miss')
            if case['answerable'] and set(r['rankings']['RRF'][:3]) & set(case['relevant_record_indices']) and not set(ids[:3]) & set(case['relevant_record_indices']):
                row['bad_cases'].append('reranker regression')
        except Exception as error:
            row.update(answer=None,elapsed_seconds=time.monotonic()-start,error=safe_error(error),bad_cases=['generation/API failure'])
        return row
    with ThreadPoolExecutor(max_workers=3) as executor:
        for row in executor.map(run,[c for c in dataset['cases'] if c['id'] not in done]):
            rows.append(row)
            good=[r for r in rows if r.get('error') is None]
            negatives=[r for r in good if not r['answerable']]
            save(output,{'golden_sha256':fingerprint,'model':qa.CHAT_MODEL,'system_prompt':qa.SYSTEM_PROMPT,
               'parameters':{'temperature':0.1,'max_tokens':qa.MAX_TOKENS,'timeout_seconds':120,'generator_max_attempts':2},
               'cases':rows,'summary':{'successful':len(good),'failed':len(rows)-len(good),
                'citation_validity':mean(r['citation_validity'] for r in good) if good else None,
                'strict_abstention_accuracy':mean(r['strict_abstention_correct'] for r in negatives) if negatives else None,
                'abstention_requires_semantic_judge':sum(r['abstention_semantic_review_needed'] for r in negatives)}})
            print(f'{row["id"]}: {"GENERATED" if not row["error"] else "ERROR"}',flush=True)
    if frozen_files()!=before:
        raise RuntimeError('Frozen files changed')

if __name__=='__main__':
    main()
