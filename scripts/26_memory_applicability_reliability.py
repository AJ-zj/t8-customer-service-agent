"""50 real, independent Gate calls only; no Agent, embedding or database."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.memory.applicability import MemoryApplicabilityGate, MemoryApplicabilityError

SOURCE = ROOT / 'data/evaluation/memory_applicability_gate_results_v1.json'
OUTPUT = ROOT / 'data/evaluation/memory_applicability_reliability_v1.json'


def main():
    if OUTPUT.exists():
        raise RuntimeError('Refusing to overwrite observations.')
    previous = json.loads(SOURCE.read_text(encoding='utf-8'))
    oracle = {r['case_id']: r for r in previous['arms']['oracle']}
    real = {r['case_id']: r for r in previous['arms']['real']}
    extra = {r['case_id']: r for r in previous['supplements']}
    inputs = [oracle['M26'], oracle['M27'], real['M04'], real['N24'],
              real['N19'], real['M01'], extra['H05'], extra['INJECTION'],
              extra['H04'], real['M28']]
    fixed = []
    for row in inputs:
        candidates = row['candidates']
        fixed.append({'case_id': row['case_id'], 'question': row['query'],
                      'facts': [c['text'] for c in candidates],
                      'expected_local_indices': [i for i, c in enumerate(candidates)
                                                 if c['index'] in row['relevant_fact_indices']]})
    report = {'inputs_fixed_before_calls': fixed, 'attempts_per_case': 5,
              'no_retry': True, 'observations': []}
    gate = MemoryApplicabilityGate()
    for case in fixed:
        for attempt in range(1, 6):
            started = time.perf_counter()
            row = {'case_id': case['case_id'], 'attempt': attempt,
                   'status': 'SUCCESS', 'selected_local_indices': None, 'semantic_correct': False}
            try:
                decision = gate.classify(case['question'], case['facts'])
                row['selected_local_indices'] = decision.applicable_fact_indices
                row['semantic_correct'] = set(decision.applicable_fact_indices) == set(case['expected_local_indices'])
            except MemoryApplicabilityError as error:
                row['status'] = error.code
            row['elapsed_seconds'] = round(time.perf_counter() - started, 3)
            report['observations'].append(row)
            OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(row, ensure_ascii=False), flush=True)
    counts = Counter(r['status'] for r in report['observations'])
    report['summary'] = {'total': 50, 'counts': dict(counts),
                         'contract_success_rate': counts['SUCCESS'] / 50,
                         'semantic_correct': sum(r['semantic_correct'] for r in report['observations'])}
    new = set(counts) - {'SUCCESS', 'INVALID_INDEX', 'TIMEOUT', 'SCHEMA_ERROR', 'JSON_ERROR',
                         'HTTP_ERROR', 'TRANSPORT_ERROR'}
    report['conclusion'] = ('APPLICABILITY_GATE_RELIABILITY: NEW_BAD_CASE' if new else
                            'APPLICABILITY_GATE_RELIABILITY: PASS' if counts['SUCCESS'] >= 49 else
                            'APPLICABILITY_GATE_RELIABILITY: STILL_UNSTABLE')
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), report['conclusion'], flush=True)


if __name__ == '__main__':
    main()
