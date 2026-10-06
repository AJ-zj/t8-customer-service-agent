import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from rag_eval_common import (load_golden,validate_golden,ranking_metrics,aggregate,
    evaluate_rankings,citation_check,strict_abstention,frozen_files)

def module(filename):
    spec=importlib.util.spec_from_file_location(filename,SCRIPTS/filename)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class RAGEvaluationTests(unittest.TestCase):
    def test_golden_schema_source_proofs_and_counts(self):
        data,records,_=load_golden()
        self.assertEqual(len(data['cases']),60)
        self.assertEqual(sum(c['answerable'] for c in data['cases']),50)
        self.assertTrue(validate_golden(data,records))

    def test_golden_sha_frozen(self):
        from rag_eval_common import GOLDEN,sha
        self.assertEqual(sha(GOLDEN.read_bytes()),GOLDEN.with_suffix('.sha256').read_text().strip())

    def test_invalid_relevant_index_rejected(self):
        data,records,_=load_golden()
        data['cases'][0]['relevant_record_indices']=[999]
        with self.assertRaises(ValueError): validate_golden(data,records)

    def test_unanswerable_invented_reference_rejected(self):
        data,records,_=load_golden()
        data['cases'][-1]['reference_answer']='invented'
        with self.assertRaises(ValueError): validate_golden(data,records)

    def test_reference_cannot_add_unsourced_claim(self):
        data,records,_=load_golden()
        data['cases'][0]['reference_answer']+='价格是999元'
        with self.assertRaises(ValueError): validate_golden(data,records)

    def test_ndcg_binary_standard_formula(self):
        s=ranking_metrics([9,2,1],[1,2])
        expected=(1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3))
        self.assertAlmostEqual(s['ndcg@3'],expected)
        self.assertEqual(ranking_metrics([2,1],[1,2])['ndcg@3'],1)

    def test_mrr_first_relevant(self):
        self.assertEqual(ranking_metrics([0,1,2],[1,2])['mrr'],.5)
        self.assertEqual(ranking_metrics([0],[1])['mrr'],0)

    def test_recall_multiple_relevant(self):
        self.assertEqual(ranking_metrics([0,1,2],[1,4])['recall@3'],.5)

    def test_duplicate_rank_rejected(self):
        with self.assertRaises(ValueError): ranking_metrics([0,0],[0])

    def test_unanswerable_excluded_and_failure_separate(self):
        rows=[dict(answerable=True,relevant_record_indices=[1],rankings={'x':[1]}),
              dict(answerable=False,relevant_record_indices=[],rankings={'x':[2]}),
              dict(answerable=True,relevant_record_indices=[2],rankings={})]
        r=aggregate(rows,'x')
        self.assertEqual((r['eligible'],r['successful'],r['failed']),(2,1,1))
        self.assertEqual(r['hit@3'],1)

    def _run_channels(self):
        from rag_eval_common import retrieval
        records=[{'text':str(i)} for i in range(12)]
        store=Mock()
        store.search_by_vector.return_value=[{'record_index':i,'dense_score':1-i/20} for i in reversed(range(12))]
        scores=[5,4,3]+[0]*9
        ranks={i:i+1 for i in range(12)}
        with patch.object(retrieval,'rank_bm25',return_value=(scores,ranks,[(0,1),(1,2),(2,3)])), \
             patch.object(retrieval,'request_rerank',return_value=[(2,.9),(1,.8),(0,.7)]) as rerank, \
             patch.object(retrieval,'reciprocal_rank_fusion',wraps=retrieval.reciprocal_rank_fusion) as rrf:
            before=copy.deepcopy(records)
            result,details=evaluate_rankings('q',records,Mock(),store,'fake',[1,0])
            self.assertEqual(records,before)
            self.assertLessEqual(len(rerank.call_args.args[1]),10)
            rrf.assert_called_once()
        return result

    def test_bm25_only_threshold(self): self.assertEqual(self._run_channels()['BM25'],[0,1,2])
    def test_dense_only_full_ranking(self): self.assertEqual(self._run_channels()['Dense'],list(reversed(range(12))))
    def test_rrf_evaluation_called(self): self.assertTrue(self._run_channels()['RRF'])
    def test_reranker_evaluation(self): self.assertEqual(self._run_channels()['Reranker'],[2,1,0])

    def test_citation_validity(self):
        self.assertTrue(citation_check('事实[E1]。事实【E2】',{'E1','E2'})['number_valid'])
        self.assertEqual(citation_check('事实[E9]',{'E1'})['invalid'],['E9'])
        self.assertTrue(citation_check('[E 1]',{'E1'})['malformed'])
        self.assertTrue(citation_check('[E1】',{'E1'})['malformed'])

    def test_abstention_not_substring_match(self):
        self.assertTrue(strict_abstention('根据当前提供的说明书资料无法确认。'))
        self.assertFalse(strict_abstention('根据当前提供的说明书资料无法确认。但售价999元。'))

    def test_generation_dry_run_uses_no_service_persistence(self):
        m=module('45_rag_generation_eval.py')
        e=[('E1',1,{'text':'清理主刷','pdf_page':15})]
        r=m.evaluate_draft({'question':'主刷清理','answerable':True},'清理主刷[E1]',e)
        self.assertEqual(r['semantic_support'],'NOT_VERIFIED_BY_DETERMINISTIC_CHECK')
        self.assertTrue(r['citation_validity'])

    def test_chunk_window_and_mapping(self):
        m=module('44_chunk_ablation.py')
        records=[{'text':'abcdefghij','pdf_page':1}]
        chunks=m.split_records(records,6,2)
        self.assertEqual([c['text'] for c in chunks],['abcdef','efghij'])
        self.assertEqual(m.parent_ranking([1,0],chunks),[0])

    def test_chunk_reference_coverage(self):
        m=module('44_chunk_ablation.py')
        records=[{'text':'abcdefghij','pdf_page':1}]
        chunks=m.split_records(records,6,2)
        case={'source_evidence':[{'record_index':0,'quote':'abcdefghij'}]}
        self.assertEqual(m.evidence_coverage(case,[0],chunks,records),.6)
        self.assertEqual(m.evidence_coverage(case,[0,1],chunks,records),1)

    def test_chunk_metrics_preserve_duplicate_parent_rank_slots(self):
        m=module('44_chunk_ablation.py')
        chunks=[{'parent_record_index':0},{'parent_record_index':0},{'parent_record_index':1}]
        score=m.chunk_slot_metrics([0,1,2],chunks,[1])
        self.assertEqual(score['mrr'],1/3)
        self.assertEqual(score['ndcg@3'],.5)

    def test_no_production_files_mutated_by_metrics(self):
        before=frozen_files()
        ranking_metrics([0,1],[1])
        self.assertEqual(before,frozen_files())

    def test_judge_failure_not_semantic_zero(self):
        # Aggregation excludes null failure scores; it must not invent 0 for unavailable judges.
        m=module('46_ragas_judge_eval.py')
        r=m.metric_summary([{'faithfulness':1.0,'answer_relevance':None},
                            {'faithfulness':None,'answer_relevance':.8}])
        self.assertEqual(r['faithfulness'],{'mean':1.0,'successful':1,'failed':1})
        self.assertEqual(r['answer_relevance']['mean'],.8)

    def test_judge_adapter_does_not_duplicate_chat_path(self):
        m=module('46_ragas_judge_eval.py')
        base=m.api_base_url('https://api.siliconflow.cn/v1/chat/completions')
        self.assertEqual(base+'/chat/completions','https://api.siliconflow.cn/v1/chat/completions')
        with self.assertRaises(ValueError): m.api_base_url('https://api.siliconflow.cn/v1/chat')
