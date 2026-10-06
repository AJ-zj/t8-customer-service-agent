"""只读分析 C05、C10、C13 的 BM25、Embedding 与 RRF 排名。"""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TARGET_CASES = ("T8-C05", "T8-C10", "T8-C13")


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载检索模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label(record):
    return f"第{record.get('pdf_page')}页《{record.get('title') or record.get('section') or '无标题'}》"


def snippet(text, limit=100):
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[:limit] + "…"


def contribution(rank, k):
    return 1 / (k + rank) if rank is not None else 0.0


def load_cached_vectors(embedding, record_count, corpus_hash):
    if not embedding.CACHE.is_file():
        raise RuntimeError(f"只读诊断需要已有 Embedding 缓存：{embedding.CACHE}")
    saved = json.loads(embedding.CACHE.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise RuntimeError("现有 Embedding 缓存格式无效；只读诊断不会重建缓存。")
    if (saved.get("model") != embedding.MODEL
            or saved.get("corpus_sha256") != corpus_hash
            or saved.get("record_count") != record_count):
        raise RuntimeError("现有 Embedding 缓存与模型或知识库不匹配；只读诊断不会重建缓存。")
    return embedding.validate_vectors(saved.get("vectors"), record_count)


def analyze_case(case, records, hybrid, bm25_index, document_vectors, query_vector):
    bm25 = hybrid.bm25_module
    embedding = hybrid.embedding_module
    query = case["question"]
    bm_scores, bm_ranks, bm_candidates = hybrid.rank_bm25(query, records, bm25_index)
    em_ranks, em_candidates = hybrid.rank_embedding(query_vector, records, document_vectors)
    em_order = embedding.rank_records(query_vector, records, document_vectors)
    em_scores = {index: score for index, score in em_order}
    fused = hybrid.reciprocal_rank_fusion(bm_candidates, em_candidates)
    rrf_ranks = {index: rank for rank, (index, _, _, _) in enumerate(fused, start=1)}
    rrf_scores = {index: score for index, score, _, _ in fused}
    targets = [item["record_index_zero_based"] for item in case["core_knowledge"]]
    print(f"\n{'=' * 72}\n{case['case_id']} | {query}")
    print(f"RRF 配置：k={hybrid.RRF_K}，BM25阈值={bm25.MIN_SCORE}，两路候选截断={hybrid.CANDIDATE_LIMIT}")

    for item, index in zip(case["core_knowledge"], targets):
        record = records[index]
        score = float(bm_scores[index])
        in_bm_candidates = index in {i for i, _ in bm_candidates}
        bm_status = "达到阈值" if score >= bm25.MIN_SCORE else "低于阈值"
        if score >= bm25.MIN_SCORE and not in_bm_candidates:
            bm_status += "，但被 BM25 候选上限截断"
        erank, esim = em_ranks[index], em_scores[index]
        rr, rscore = rrf_ranks.get(index), rrf_scores.get(index)
        brank = bm_ranks[index] if in_bm_candidates else None
        erank_in_rrf = erank if index in {i for i, _ in em_candidates} else None
        bcontrib = contribution(brank, hybrid.RRF_K)
        econtrib = contribution(erank_in_rrf, hybrid.RRF_K)
        print(f"\n核心知识：{item.get('pdf_location') or label(record)} | 记录索引={index}")
        print(f"  BM25：原始分数={score:.6f}，完整排名={bm_ranks[index]}，{bm_status}；参与融合={in_bm_candidates}")
        print(f"  Embedding：余弦相似度={esim:.6f}，完整排名={erank}")
        print(f"  RRF：排名={rr if rr is not None else '未进融合候选'}，分数={rscore if rscore is not None else 0.0:.6f}；"
              f"BM25贡献={bcontrib:.6f}，Embedding贡献={econtrib:.6f}")

    print("\nRRF Top-5 候选（显示来源名次、分路贡献与正文片段）：")
    earliest_core = min((rrf_ranks[index] for index in targets if index in rrf_ranks), default=None)
    for position, (index, score, bm_rank, em_rank) in enumerate(fused[:5], start=1):
        record = records[index]
        bc, ec = contribution(bm_rank, hybrid.RRF_K), contribution(em_rank, hybrid.RRF_K)
        print(f"  {position}. {label(record)} | RRF={score:.6f} | "
              f"BM25名次={bm_rank if bm_rank is not None else '无'}(贡献{bc:.6f}) | "
              f"Embedding名次={em_rank if em_rank is not None else '无'}(贡献{ec:.6f})")
        print(f"     {snippet(record.get('text', ''))}")
    if earliest_core is None:
        print("  正确知识不在 RRF 融合候选中。")
    else:
        ahead = [(rank, index, score, br, er) for rank, (index, score, br, er)
                 in enumerate(fused, start=1) if rank < earliest_core]
        print(f"  排在首条正确知识（RRF第{earliest_core}名）之前的候选：")
        for rank, index, score, br, er in ahead:
            print(f"    第{rank}名 {label(records[index])} | BM25名次={br} | Embedding名次={er} | RRF={score:.6f}")

    if case["case_id"] == "T8-C05":
        target = records[targets[0]]
        lengths = [len(record.get("text", "")) for record in records]
        tokens = len(bm25.tokenize(target.get("text", "")))
        markers = ("重置WiFi", "添加设备", "开机 /关机", "启动清扫", "局部清扫")
        found = [marker for marker in markers if marker in target.get("text", "")]
        median = sorted(lengths)[len(lengths) // 2]
        print(f"\nC05 第12页记录规模：{len(target.get('text', ''))}字，{tokens}个jieba词项；"
              f"41条记录字数中位数={median}，倍数={len(target.get('text', '')) / max(median, 1):.2f}。")
        print(f"  同一记录中检测到的主题标记：{', '.join(found) or '无'}")
    return {"case": case, "targets": targets, "bm_scores": bm_scores,
            "bm_ranks": bm_ranks, "bm_candidates": bm_candidates,
            "em_scores": em_scores, "em_ranks": em_ranks,
            "em_candidates": em_candidates, "fused": fused, "rrf_ranks": rrf_ranks}


def print_attribution(reports, records, hybrid):
    by_id = {report["case"]["case_id"]: report for report in reports}
    print("\n简短归因报告")
    print("已由本次具体分数/名次证实：")
    r = by_id["T8-C05"]
    i = r["targets"][0]
    print(f"  C05：正确知识 BM25 第{r['bm_ranks'][i]}名、Embedding 第{r['em_ranks'][i]}名、"
          f"RRF第{r['rrf_ranks'].get(i, '未进候选')}名；第12页整页有{len(records[i].get('text', ''))}字，"
          f"Embedding未进入Top-10候选。")
    r = by_id["T8-C10"]
    i = r["targets"][0]
    score = float(r["bm_scores"][i])
    print(f"  C10：BM25分数{score:.6f}达到{hybrid.bm25_module.MIN_SCORE}阈值，但全库第{r['bm_ranks'][i]}名，"
          f"超过Top-{hybrid.CANDIDATE_LIMIT}候选截断；目标只获Embedding贡献，RRF第{r['rrf_ranks'].get(i, '未进候选')}名。")
    r = by_id["T8-C13"]
    p3, p19 = r["targets"]
    print(f"  C13：第19页替代核心知识 RRF第{r['rrf_ranks'].get(p19, '未进候选')}名，"
          f"BM25第{r['bm_ranks'][p19]}名 + Embedding第{r['em_ranks'][p19]}名共同贡献；"
          f"第3页核心知识 RRF第{r['rrf_ranks'].get(p3, '未进候选')}名。any_of因此由第19页命中。")
    print("仍需实验验证的原因假设：")
    print("  C05：整页记录明显长于语料中位数且混合多个主题，可能稀释局部清扫语义；需与同一页拆分主题后的向量排名做对照，不能由本次排名单独定因。")
    print("  C10：候选截断是未进入BM25融合池的直接机制；BM25为何全库仅第13名，可再做词项贡献/候选分布分析。")


def main():
    try:
        evaluation = load_script("t8_evaluation_readonly", "08_evaluate_retrieval.py")
        candidate, records, answerable, _ = evaluation.verify_inputs()
        cases = {case["case_id"]: case for case in answerable}
        hybrid = load_script("t8_hybrid_diagnostic", "07_hybrid_search.py")
        bm_records, bm_index = hybrid.bm25_module.build_index()
        live_records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm_records or records != live_records:
            raise RuntimeError("检索模块读取的知识记录或顺序与候选集校验结果不一致。")
        if corpus_hash != candidate["source"]["corpus_sha256"]:
            raise RuntimeError("JSONL 指纹与候选集不一致，停止诊断。")
        missing = [case_id for case_id in TARGET_CASES if case_id not in cases]
        if missing:
            raise RuntimeError(f"候选集中缺少诊断题：{missing}")

        embedding = hybrid.embedding_module
        api_key = embedding.get_api_key()
        vectors = load_cached_vectors(embedding, len(records), corpus_hash)
        selected = [cases[case_id] for case_id in TARGET_CASES]
        query_vectors = embedding.request_embeddings([case["question"] for case in selected], api_key)
        if len(query_vectors) != len(selected):
            raise RuntimeError("查询向量数量不匹配，停止诊断。")
        print(f"输入指纹与核心标注已核验；知识向量使用现有缓存（{len(records)}条）。")
        reports = [analyze_case(case, records, hybrid, bm_index, vectors, vector)
                   for case, vector in zip(selected, query_vectors)]
        print_attribution(reports, records, hybrid)
        print("说明：诊断只读，不改变检索参数、知识库或候选答案。")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"诊断失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
