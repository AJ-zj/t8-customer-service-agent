"""Read saved decisions; two deterministic offline guards, no network or DB."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data/evaluation/memory_applicability_semantic_v1.json'
SOURCE = ROOT / 'data/evaluation/memory_applicability_semantic_ab_results_v1.json'
OUTPUT = ROOT / 'data/evaluation/memory_applicability_guardrail_results_v1.json'
SPEC = importlib.util.spec_from_file_location('semantic_metrics', ROOT / 'scripts/27_memory_applicability_semantic_ab.py')
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)  # Import only: main/network calls never executed.

PREFERENCE_FACT = re.compile(r'喜欢|偏好|习惯|更关心|倾向')
PREFERENCE_QUERY = re.compile(r'喜欢|偏好|习惯|希望|怕吵|噪音|噪声|安静|静音|声音|更关心|倾向|常用|平常')
HOME_FACT = re.compile(
    r'(?:我家(?:卧室|厨房|楼上|客厅)?(?:有|是|铺了|铺的是|同时有)(?:地毯|木地板|瓷砖|地毯和木地板)'
    r'|(?:我家|家里|家中)养(?:猫|狗)(?:，地上(?:猫|狗)毛比较多)?)[。]?'
)
# Use a deliberately narrow complete-fact grammar, not a loose keyword match.
ENVIRONMENT_CONTEXT = re.compile(r'我家|家里|家中|这种地面|这种环境|环境下|卧室|客厅|厨房|宠物环境|猫毛|狗毛')
CAPABILITY = re.compile(r'支持哪些|有哪些功能|支持什么|能不能支持|产品有什么|功能有哪些')
PERSONAL_NEED = re.compile(r'我家|我的|适合我|按我|根据我')
GROUND_REFERENCE = re.compile(r'这种地面|地板材质|地面|两种地面')
ANCHORS = {
    'carpet': re.compile(r'地毯'), 'wood': re.compile(r'木地板'),
    'floor': re.compile(r'地板'), 'tile': re.compile(r'瓷砖'),
    'cat': re.compile(r'猫'), 'dog': re.compile(r'狗'),
    'pet': re.compile(r'宠物|毛发'),
    'bedroom': re.compile(r'卧室'), 'living_room': re.compile(r'客厅'),
    'kitchen': re.compile(r'厨房'),
}


def is_home_fact(fact):
    return HOME_FACT.fullmatch(fact.strip()) is not None


def home_corresponds(question, fact):
    # A shared noun alone is insufficient: caller must first require user/environment context.
    for kind, pattern in ANCHORS.items():
        if pattern.search(question):
            if kind == 'pet' and re.search(r'养猫|养狗|猫毛|狗毛', fact):
                return True
            if pattern.search(fact):
                return True
    return False


def guardrail(question, facts, selected):
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < len(facts) for i in selected):
        raise ValueError('Invalid saved contract: no partial acceptance.')
    chosen = set(selected)
    events = []
    if not PREFERENCE_QUERY.search(question):
        for i in sorted(chosen):
            if PREFERENCE_FACT.search(facts[i]):
                chosen.remove(i)
                events.append({'rule_id': 'REMOVE_UNASKED_PREFERENCE', 'index': i})
    if ENVIRONMENT_CONTEXT.search(question) and not (
            CAPABILITY.search(question) and not PERSONAL_NEED.search(question)):
        home = [i for i, fact in enumerate(facts) if is_home_fact(fact)]
        grounds = [i for i in home if re.search(r'地毯|木地板|瓷砖', facts[i])]
        for i in home:
            direct = home_corresponds(question, facts[i])
            # "这种地面" can refer to the one available concrete ground fact.
            # Ambiguous multiple ground facts are deliberately not inferred.
            unique_ground = bool(GROUND_REFERENCE.search(question) and grounds == [i])
            if i not in chosen and (direct or unique_ground):
                chosen.add(i)
                events.append({'rule_id': 'ADD_EXPLICIT_HOME_CONTEXT', 'index': i})
    final = [i for i in range(len(facts)) if i in chosen]
    return {'final_selected': final, 'added': sorted(chosen-set(selected)),
            'removed': sorted(set(selected)-chosen), 'triggered_rules': events}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise RuntimeError('Refusing to overwrite result.')
    data=json.loads(DATASET.read_text(encoding='utf-8'))
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert sha(DATASET)==source['dataset_sha256']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (DATASET,SOURCE,ROOT/'app/memory/applicability.py')}
    cases=data['cases']; by_id={c['case_id']:c for c in cases}
    report={'source_hashes':hashes,'llm_calls':0,
            'policy':'Core48 first attempt metrics; all58 and repeated hard cases reported separately. Failed LLM calls remain fail-closed without deterministic additions.',
            'rules':{'REMOVE_UNASKED_PREFERENCE':PREFERENCE_QUERY.pattern,
                     'ADD_EXPLICIT_HOME_CONTEXT':{'context':ENVIRONMENT_CONTEXT.pattern,
                       'fact_grammar':HOME_FACT.pattern,'capability_exclusion':CAPABILITY.pattern,
                       'generic_ground':'Only one concrete ground candidate; no ambiguous guessing.'}}, 'arms':{}}
    for arm in ('Qwen/Qwen2.5-7B-Instruct|B','deepseek-ai/DeepSeek-V4-Flash|A'):
        rows=[]; adapted=[]; original=source['arms'][arm]
        for row in original:
            case=by_id[row['case_id']]
            change=guardrail(case['question'],case['candidate_facts'],row['selected']) if row['status']=='SUCCESS' else {
                'final_selected':[],'added':[],'removed':[],'triggered_rules':[]}
            exact=row['status']=='SUCCESS' and set(change['final_selected'])==set(case['expected_applicable_indices'])
            rows.append({'case_id':row['case_id'],'attempt':row['attempt'],'question':case['question'],
                         'candidate_facts':case['candidate_facts'],'original_llm_selected':row['selected'],
                         **change,'expected':case['expected_applicable_indices'],'original_correct':row['exact'],
                         'correct':exact,'status':row['status']})
            adapted.append({**row,'selected':change['final_selected'],'exact':exact})
        core_ids={c['case_id'] for c in cases if c['evaluation_split']=='core48'}
        first=[r for r in adapted if r['attempt']==1]
        core_rows=[r for r in first if r['case_id'] in core_ids]
        core_original=[r for r in original if r['attempt']==1 and r['case_id'] in core_ids]
        summary={'original_core48':BASE.metrics(cases,core_original),'guardrail_core48':BASE.metrics(cases,core_rows),
                 'original_all58':BASE.metrics(cases,[r for r in original if r['attempt']==1]),
                 'guardrail_all58':BASE.metrics(cases,first),
                 'categories':{cat:{'before':BASE.metrics(cases,[r for r in core_original if by_id[r['case_id']]['category']==cat]),
                                    'after':BASE.metrics(cases,[r for r in core_rows if by_id[r['case_id']]['category']==cat])}
                               for cat in ('device_profile','home_environment','usage_preference')},
                 'hard_cases':{cid:{'before_exact':sum(r['original_correct'] for r in rows if r['case_id']==cid),
                                   'after_exact':sum(r['correct'] for r in rows if r['case_id']==cid),
                                   'attempts':sum(r['case_id']==cid for r in rows)} for cid in BASE.HARD},
                 'added_all_attempts':sum(len(r['added']) for r in rows),
                 'removed_all_attempts':sum(len(r['removed']) for r in rows),
                 'added_core48':sum(len(r['added']) for r in rows if r['attempt']==1 and r['case_id'] in core_ids),
                 'removed_core48':sum(len(r['removed']) for r in rows if r['attempt']==1 and r['case_id'] in core_ids),
                 'regressions':[{'case_id':r['case_id'],'attempt':r['attempt']} for r in rows if r['original_correct'] and not r['correct']],
                 'improvements':[{'case_id':r['case_id'],'attempt':r['attempt']} for r in rows if not r['original_correct'] and r['correct']]}
        met=summary['guardrail_core48']
        stable=all(h['after_exact']==h['attempts'] for h in summary['hard_cases'].values())
        summary['candidate']=met['false_injection']<=1 and met['precision']>=.85 and met['recall']>=.8 and met['exact_accuracy']>=.8 and stable and not summary['regressions']
        report['arms'][arm]={'rows':rows,'summary':summary}
    q=report['arms']['Qwen/Qwen2.5-7B-Instruct|B']['summary']['candidate']
    d=report['arms']['deepseek-ai/DeepSeek-V4-Flash|A']['summary']['candidate']
    report['conclusion']='QWEN_GUARDRAIL_CANDIDATE' if q else 'DEEPSEEK_GUARDRAIL_CANDIDATE' if d else 'DETERMINISTIC_GUARDRAIL_INSUFFICIENT'
    assert all(sha(ROOT/p)==value for p,value in hashes.items())
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({name:a['summary'] for name,a in report['arms'].items()},ensure_ascii=False,indent=2))
    print(report['conclusion'])


if __name__=='__main__':
    main()
