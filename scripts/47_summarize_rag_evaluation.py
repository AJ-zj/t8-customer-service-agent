"""Deterministic report from saved observations. Does not call any API."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag_eval_common import *
from collections import Counter


def main():
    dataset,records,fingerprint=load_golden()
    loaded={name:json.loads((RESULTS/f'{name}_v1.json').read_text(encoding='utf-8'))
            for name in ('retrieval_ablation','chunk_ablation','generation_eval','ragas_eval')}
    if any(data['golden_sha256']!=fingerprint for data in loaded.values()):
        raise ValueError('Results are not from one Golden')
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('judge_summary',Path(__file__).with_name('46_ragas_judge_eval.py'))
    judge_module=module_from_spec(spec)
    spec.loader.exec_module(judge_module)
    generation=loaded['generation_eval']['cases']
    judge=loaded['ragas_eval']['cases']
    good_gen=[r for r in generation if r.get('error') is None]
    positives=[r for r in judge if r['answerable']]
    bad={}
    for r in generation+judge:
        bad.setdefault(r['id'],set()).update(r.get('bad_cases',[]))
    rows=[dict(id=r['id'],question=r['question'],bad_case_tags=sorted(bad.get(r['id'],set())))
          for r in generation if bad.get(r['id'])]
    probe=RESULTS/'ragas_eval_v1_adapter_failure.json'
    diagnostic=json.loads(probe.read_text(encoding='utf-8'))['summary'] if probe.exists() else None
    report={'golden_sha256':fingerprint,'cases':len(dataset['cases']),
        'answerable':sum(c['answerable'] for c in dataset['cases']),
        'unanswerable':sum(not c['answerable'] for c in dataset['cases']),
        'retrieval':loaded['retrieval_ablation']['summary'],
        'chunk_configs':loaded['chunk_ablation']['configs'],
        'generation_successful':len(good_gen),'generation_failed':len(generation)-len(good_gen),
        'citation_number_validity':mean(r['citations']['number_valid'] for r in good_gen),
        'strict_citation_requirement_validity':mean(r['citation_validity'] for r in good_gen),
        'answerable_citation_presence':mean(r['citations']['has_citation'] for r in good_gen if r['answerable']),
        'generation_judge_all_samples':loaded['ragas_eval']['summary'],
        'generation_judge_answerable_only':judge_module.metric_summary(positives),
        'ragas_version':loaded['ragas_eval']['ragas_version'],
        'judge_model':loaded['ragas_eval']['judge_model'],
        'initial_adapter_failure_excluded_from_semantic_scores':diagnostic,
        'bad_cases':rows,
        'limitations':['Single-record atomic questions; Recall equals Hit in V1.',
          'RAGAS is a model estimate, not human ground truth; manually inspect known contradictory low-score samples.',
          'Top5 measures the diagnostic reranker ranking, not production evidence count.',
          'Chunk record hit is a proxy; inspect exact source coverage.',
          'Production chunk optimality cannot be established from these five settings.'],
        'production_changed':False,'database_writes':0}
    save(RESULTS/'rag_evaluation_summary_v1.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='bad_cases'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
