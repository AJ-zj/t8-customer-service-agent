"""只读诊断 T8 主刷清理问题在现有检索流水线中的排名。"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
QUERY = "请问石头 T8 的主刷怎么清理？"
TARGET = 14
TOP_N = 10


def load_script(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载现有模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label(record):
    title = record.get("title") or record.get("section") or "（无标题）"
    return f"第{record.get('pdf_page')}页《{title}》"


def print_rows(stage, ranked, records, score_name):
    print(f"\n{stage} Top-{min(TOP_N, len(ranked))}：")
    for rank, (index, score) in enumerate(ranked[:TOP_N], 1):
        print(f"  {rank:>2}. record_index={index} | {label(records[index])} | "
              f"{score_name}={score:.8f}")


def main():
    try:
        hybrid = load_script("t8_debug_hybrid", "07_hybrid_search.py")
        reranker = load_script("t8_debug_reranker", "11_rerank_experiment.py")
        bm_records, bm25_index = hybrid.bm25_module.build_index()
        records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm_records:
            raise RuntimeError("BM25 与 Embedding 知识记录顺序不一致。")
        if len(records) <= TARGET or records[TARGET].get("pdf_page") != 15:
            raise RuntimeError("record_index=14 未对应到预期的说明书第15页。")
        if "软胶主刷" not in records[TARGET].get("text", ""):
            raise RuntimeError("record_index=14 不包含预期的软胶主刷内容。")

        embedding = hybrid.embedding_module
        document_vectors = reranker.load_cached_vectors(
            embedding, len(records), corpus_hash
        )
        api_key = embedding.get_api_key()
        query_vectors = embedding.request_embeddings([QUERY], api_key)
        if len(query_vectors) != 1 or len(query_vectors[0]) != len(document_vectors[0]):
            raise RuntimeError("查询向量与文档向量数量或维度不匹配。")

        bm_scores, bm_ranks, bm_candidates = hybrid.rank_bm25(
            QUERY, records, bm25_index
        )
        bm_order = sorted(range(len(records)), key=lambda i: bm_scores[i], reverse=True)
        bm_full = [(i, float(bm_scores[i])) for i in bm_order]
        em_full = embedding.rank_records(query_vectors[0], records, document_vectors)
        em_ranks = {i: rank for rank, (i, _) in enumerate(em_full, 1)}
        _, em_candidates = hybrid.rank_embedding(
            query_vectors[0], records, document_vectors
        )
        fused = hybrid.reciprocal_rank_fusion(
            bm_candidates, em_candidates, k=hybrid.RRF_K
        )
        union_ids = list(dict.fromkeys(
            [i for i, _ in bm_candidates] + [i for i, _ in em_candidates]
        ))
        reranked = reranker.request_rerank(
            QUERY, [(i, records[i]) for i in union_ids], api_key
        )

        print(f"问题：{QUERY}\n目标：record_index={TARGET} | {label(records[TARGET])}")
        print(f"候选规则：BM25 分数≥{hybrid.bm25_module.MIN_SCORE} 后取前{hybrid.CANDIDATE_LIMIT}条；"
              f"Embedding 取前{hybrid.CANDIDATE_LIMIT}条；RRF k={hybrid.RRF_K}；"
              f"Reranker 联合候选={len(union_ids)}条。")

        print_rows("BM25 全库排序", bm_full, records, "score")
        print_rows("Embedding 全库排序", em_full, records, "cosine")
        rrf_rows = [(i, score) for i, score, _, _ in fused]
        print_rows("RRF 融合排序", rrf_rows, records, "rrf_score")
        print_rows("Reranker 最终排序", reranked, records, "rerank_score")

        rrf_ranks = {i: rank for rank, (i, _, _, _) in enumerate(fused, 1)}
        rerank_ranks = {i: rank for rank, (i, _) in enumerate(reranked, 1)}
        bm_in = any(i == TARGET for i, _ in bm_candidates)
        em_in = any(i == TARGET for i, _ in em_candidates)
        print("\n目标记录逐阶段结果：")
        print(f"  BM25：第{bm_ranks[TARGET]}名，score={bm_scores[TARGET]:.8f}，"
              f"进入 RRF 候选={'是' if bm_in else '否'}")
        em_score = next(score for i, score in em_full if i == TARGET)
        print(f"  Embedding：第{em_ranks[TARGET]}名，cosine={em_score:.8f}，"
              f"进入 RRF 候选={'是' if em_in else '否'}")
        rrf_rank = rrf_ranks.get(TARGET)
        rrf_score = next((score for i, score, _, _ in fused if i == TARGET), None)
        rrf_score_text = f"{rrf_score:.8f}" if rrf_score is not None else "N/A"
        print(f"  RRF：{f'第{rrf_rank}名' if rrf_rank else '未进入候选'}，"
              f"rrf_score={rrf_score_text}")
        rerank_rank = rerank_ranks.get(TARGET)
        rerank_score = next((score for i, score in reranked if i == TARGET), None)
        rerank_score_text = f"{rerank_score:.8f}" if rerank_score is not None else "N/A"
        print(f"  Reranker：{f'第{rerank_rank}名' if rerank_rank else '未返回'}，"
              f"rerank_score={rerank_score_text}")

        if not bm_in and not em_in:
            diagnosis = "候选召回：目标记录未进入 BM25/Embedding 的联合候选池。"
        elif bm_ranks[TARGET] > 3 and em_ranks[TARGET] > 3:
            diagnosis = ("BM25/Embedding 初排：目标虽进入联合候选池，但两路都已排在 Top-3 外；"
                         f"RRF第{rrf_rank}名、Reranker第{rerank_rank}名，后续未能提升。")
        elif rrf_rank is not None and rrf_rank > 3:
            diagnosis = "RRF：目标已召回，但首次在 RRF 排序中落到 Top-3 之外。"
        elif rerank_rank is not None and rerank_rank > 3:
            diagnosis = "Reranker：目标进入 RRF Top-3，但被重排到最终 Top-3 之外。"
        elif rerank_rank is not None:
            diagnosis = "当前完整流水线中目标仍在最终 Top-3；本次未复现 Top-3 失败。"
        else:
            diagnosis = "Reranker：目标进入融合候选，但未出现在其返回排序中。"
        print(f"\n首次问题阶段：{diagnosis}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, StopIteration) as error:
        print(f"诊断失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
