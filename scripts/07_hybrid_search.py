"""BM25 与 Embedding 的 Reciprocal Rank Fusion 混合检索。"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_LIMIT = 10
RRF_K = 60
TOP_K = 3


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载检索模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bm25_module = load_script("t8_bm25", "05_bm25_search.py")
embedding_module = load_script("t8_embedding", "06_embedding_search.py")


def rank_bm25(query, records, index):
    """完整 BM25 排序供查名次；仅保留达基线阈值的前 10 条供融合。"""
    tokens = bm25_module.tokenize(query)
    scores = index.get_scores(tokens) if tokens else [0.0] * len(records)
    ordered = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    all_ranks = {record_id: rank for rank, record_id in enumerate(ordered, 1)}
    candidates = [(i, all_ranks[i]) for i in ordered
                  if scores[i] >= bm25_module.MIN_SCORE][:CANDIDATE_LIMIT]
    return scores, all_ranks, candidates


def rank_embedding(query_vector, records, document_vectors):
    """复用 Embedding 排序；保留全库名次，截取前 10 条参与融合。"""
    ordered = embedding_module.rank_records(query_vector, records, document_vectors)
    all_ranks = {record_id: rank for rank, (record_id, _) in enumerate(ordered, 1)}
    candidates = [(record_id, all_ranks[record_id])
                  for record_id, _ in ordered[:CANDIDATE_LIMIT]]
    return all_ranks, candidates


def reciprocal_rank_fusion(bm25_candidates, embedding_candidates, k=RRF_K):
    """按标准 RRF 求分；只融合候选名次，不混加两路原始分数。"""
    rank_lists = [{record_id: rank for record_id, rank in candidates}
                  for candidates in (bm25_candidates, embedding_candidates)]
    candidate_ids = set().union(*(set(ranks) for ranks in rank_lists))
    fused = []
    for record_id in candidate_ids:
        score = sum(1 / (k + ranks[record_id])
                    for ranks in rank_lists if record_id in ranks)
        fused.append((record_id, score, rank_lists[0].get(record_id),
                      rank_lists[1].get(record_id)))
    return sorted(fused, key=lambda item: (-item[1], item[0]))


def rank_label(rank):
    return f"第{rank}名" if rank is not None else "未进候选"


def show_top3(query, fused, records, bm25_ranks, embedding_ranks):
    print(f"问题：{query} | RRF k={RRF_K}")
    if not fused:
        print("没有通过 BM25 阈值或 Embedding Top-10 截断的候选。")
        return
    for position, (record_id, score, _, _) in enumerate(fused[:TOP_K], 1):
        record = records[record_id]
        title = record.get("title") or record.get("section") or "（无标题）"
        print(f"Top {position} | 第{record.get('pdf_page')}页《{title}》 | RRF={score:.6f}"
              f" | BM25={rank_label(bm25_ranks.get(record_id))}"
              f" | Embedding={rank_label(embedding_ranks.get(record_id))}")


def evaluate_query(query, query_vector, case, records, bm25_index, document_vectors):
    bm25_scores, bm25_ranks, bm25_candidates = rank_bm25(query, records, bm25_index)
    embedding_ranks, embedding_candidates = rank_embedding(
        query_vector, records, document_vectors
    )
    fused = reciprocal_rank_fusion(bm25_candidates, embedding_candidates)
    show_top3(query, fused, records, bm25_ranks, embedding_ranks)
    if case is None:
        return True

    targets = [i for i, record in enumerate(records)
               if bm25_module.is_test_target(record, case)]
    if len(targets) != 1:
        print(f"[未通过] 目标知识匹配到 {len(targets)} 条，无法核验目标排名。")
        return False

    target = targets[0]
    fused_ranks = {record_id: rank for rank, (record_id, _, _, _) in enumerate(fused, 1)}
    bm25_hit = bm25_scores[target] >= bm25_module.MIN_SCORE and bm25_ranks[target] <= TOP_K
    embedding_hit = embedding_ranks[target] <= TOP_K
    rrf_rank = fused_ranks.get(target)
    rrf_hit = rrf_rank is not None and rrf_rank <= TOP_K
    print(f"目标排名：BM25 第{bm25_ranks[target]}名（Top-3={'是' if bm25_hit else '否'}）；"
          f"Embedding 第{embedding_ranks[target]}名（Top-3={'是' if embedding_hit else '否'}）；"
          f"RRF={rank_label(rrf_rank)}（Top-3={'是' if rrf_hit else '否'}）。")
    print(f"[{'通过' if rrf_hit else '未通过'}] RRF 目标知识是否进入 Top-3。")
    return rrf_hit


def main():
    parser = argparse.ArgumentParser(description="T8 说明书 BM25 + Embedding RRF 混合检索")
    parser.add_argument("query", nargs="?", help="要检索的问题")
    parser.add_argument("--test", action="store_true", help="运行原有 4 道核验题")
    args = parser.parse_args()
    if args.test and args.query is not None:
        parser.error("--test 不需要额外查询文本")
    if not args.test and (args.query is None or not args.query.strip()):
        parser.error("请提供非空查询文本，或使用 --test")

    try:
        records, corpus_hash = embedding_module.load_corpus()
        bm25_records, bm25_index = bm25_module.build_index()
        if records != bm25_records:
            raise RuntimeError("两路检索读取的知识记录或顺序不一致，已停止融合。")
        api_key = embedding_module.get_api_key()
        document_vectors = embedding_module.load_document_vectors(
            records, corpus_hash, api_key
        )
        cases = bm25_module.TEST_CASES if args.test else [None]
        queries = [case["query"] for case in cases] if args.test else [args.query]
        query_vectors = embedding_module.request_embeddings(queries, api_key)
        if len(query_vectors) != len(queries):
            raise RuntimeError("查询向量数量与问题数量不一致。")
        results = [evaluate_query(query, vector, case, records, bm25_index, document_vectors)
                   for query, vector, case in zip(queries, query_vectors, cases)]
        return 0 if all(results) else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
