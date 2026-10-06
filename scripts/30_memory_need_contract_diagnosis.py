"""Zero-call diagnosis of saved strict JSON and exact bare-enum predictions.

This does not repair V2.5A or change a production parser. If semantic quality
is insufficient, no Plain Enum experiment is performed.
"""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/evaluation/memory_need_classification_results_v1.json'
DATASET = ROOT / 'data/evaluation/memory_need_classification_v1.json'
OUTPUT = ROOT / 'data/evaluation/memory_need_contract_diagnosis_v1.json'
spec = importlib.util.spec_from_file_location('memory_need_metrics', ROOT / 'scripts/29_memory_need_classification_experiment.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)  # No main() or API call.


def exact_bare_label(content):
    if not isinstance(content, str):
        return None
    value = content.strip()
    return value if value in baseline.MODES else None


def meets_semantic_threshold(summary):
    classes = summary['per_class']
    return (summary['overall_accuracy'] >= .95
            and classes['NONE']['precision'] >= .95
            and classes['NONE']['recall'] >= .95
            and classes['RECALL']['f1'] >= .9
            and classes['PERSONALIZE']['f1'] >= .9)


def main():
    if OUTPUT.exists():
        raise RuntimeError('Refusing to overwrite diagnostic results.')
    # Parser checks use fixtures only, not model calls.
    assert exact_bare_label(' NONE\n') == 'NONE'
    for invalid in ('答案是 NONE', 'NONE.', '```NONE```', '{"mode":"NONE"}',
                    'NONE RECALL', 'none', '"NONE"', None):
        assert exact_bare_label(invalid) is None
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    dataset_raw = DATASET.read_bytes()
    assert hashlib.sha256(dataset_raw).hexdigest() == source['dataset_sha256']
    annotations = {c['case_id']: c for c in json.loads(dataset_raw)['cases']}
    strict_rows = []
    semantic_rows = []
    observations = []
    for original in source['cases']:
        annotation = annotations[original['case_id']]
        assert original['current_question'] == annotation['current_question']
        assert original['expected_mode'] == annotation['expected_mode']
        strict_mode = None
        if original['schema_valid']:
            decision = baseline.MemoryNeedDecision.model_validate(
                json.loads(original['visible_output']))
            strict_mode = decision.mode
            assert strict_mode == original['predicted_mode']
        bare = exact_bare_label(original.get('visible_output'))
        # Timeouts/transport failures remain ERROR even if partial text exists.
        semantic_mode = strict_mode or (
            bare if original['status'] == 'MALFORMED_JSON' else None)
        strict_rows.append({**original, 'predicted_mode': strict_mode})
        semantic_rows.append({**original, 'predicted_mode': semantic_mode,
                              'schema_valid': semantic_mode is not None,
                              'status': 'SUCCESS' if semantic_mode else original['status']})
        observations.append({**annotation, 'raw_assistant_content': original.get('visible_output'),
                             'original_status': original['status'],
                             'strict_prediction': strict_mode, 'semantic_prediction': semantic_mode,
                             'prediction_source': 'STRICT_JSON' if strict_mode else
                             'EXACT_BARE_ENUM' if semantic_mode else 'ERROR',
                             'semantic_correct': semantic_mode == original['expected_mode']})
    assert len(observations) == 60
    strict = baseline.evaluate(strict_rows)
    semantic = baseline.evaluate(semantic_rows)
    # semantic summary acceptance is diagnostic only, never JSON schema success.
    semantic['diagnostic_label_success'] = semantic.pop('schema_success')
    passed = meets_semantic_threshold(semantic)
    assert not passed, 'Semantic threshold met: stop and review before a separate real Plain Enum run.'
    result = {'source_sha256': hashlib.sha256(raw).hexdigest(),
              'dataset_sha256': source['dataset_sha256'], 'new_model_calls': 0,
              'strict_json_summary': strict, 'semantic_label_summary': semantic,
              'prediction_sources': dict(Counter(o['prediction_source'] for o in observations)),
              'semantic_threshold_met': passed, 'plain_enum_retest_performed': False,
              'cases': observations, 'conclusion': 'MEMORY_NEED_SEMANTIC_INSUFFICIENT'}
    assert SOURCE.read_bytes() == raw and DATASET.read_bytes() == dataset_raw
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('strict_json_summary', 'semantic_label_summary',
          'prediction_sources', 'semantic_threshold_met', 'new_model_calls', 'conclusion')},
          ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
