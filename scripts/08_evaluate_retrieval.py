"""统一评测 T8 候选集中的 BM25、Embedding 与 RRF 检索结果。"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/evaluation/t8_retrieval_candidates_v1.json"
TOP_K = 3


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载检索模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inputs():
    candidate = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    source = candidate["source"]
    raw_by_name = {}
    for field, path_field in (("pdf_sha256", "official_pdf_path"),
                              ("corpus_sha256", "corpus_path")):
        path = ROOT / source[path_field]
        if not path.is_file():
            raise RuntimeError(f"输入文件不存在：{path}")
        raw = path.read_bytes()
        raw_by_name[path_field] = raw
        actual = hashlib.sha256(raw).hexdigest()
        if actual != source[field]:
            raise RuntimeError(
                f"{path_field} SHA-256 不匹配：候选集记录={source[field]}，当前文件={actual}。停止评测。"
            )

    records = [json.loads(line) for line in raw_by_name["corpus_path"].decode("utf-8").splitlines()
               if line.strip()]
    if len(records) != source["corpus_records"]:
        raise RuntimeError(f"知识记录数不匹配：候选集={source['corpus_records']}，当前={len(records)}。")
    cases = candidate.get("cases", [])
    if len(cases) != 15:
        raise RuntimeError(f"候选题数量应为15，当前为{len(cases)}。")
    answerable = [case for case in cases if case.get("answerability") == "answerable"]
    unanswerable = [case for case in cases if case.get("answerability") == "unanswerable"]
    if len(answerable) != 13 or len(unanswerable) != 2:
        raise RuntimeError(f"题目分组异常：有答案={len(answerable)}，无答案={len(unanswerable)}。")

    for case in cases:
        core = case.get("core_knowledge", [])
        if case.get("answerability") == "unanswerable":
            if core or case.get("core_match_policy") != "not_applicable":
                raise RuntimeError(f"{case['case_id']} 的无答案标注不符合约定。")
            continue
        if not core:
            raise RuntimeError(f"{case['case_id']} 没有核心知识标注。")
        if case.get("core_match_policy") not in {"all_of", "any_of"}:
            raise RuntimeError(f"{case['case_id']} 的核心匹配规则无效。")
        for item in core:
            index = item.get("record_index_zero_based")
            if not isinstance(index, int) or not 0 <= index < len(records):
                raise RuntimeError(f"{case['case_id']} 的知识记录索引无效：{index!r}。")
            record, locator = records[index], item.get("record_locator", {})
            for field in ("model", "pdf_page", "section", "title"):
                if record.get(field) != locator.get(field):
                    raise RuntimeError(
                        f"{case['case_id']} 索引{index}的{field}不匹配："
                        f"候选集={locator.get(field)!r}，JSONL={record.get(field)!r}。"
                    )
            text_hash = hashlib.sha256(record.get("text", "").encode("utf-8")).hexdigest()
            if text_hash != locator.get("text_sha256"):
                raise RuntimeError(f"{case['case_id']} 索引{index}的文本 SHA-256 不匹配。")
    return candidate, records, answerable, unanswerable


def record_label(record):
    title = record.get("title") or record.get("section") or "（无标题）"
    return f"第{record.get('pdf_page')}页《{title}》"


def core_hit_status(case, top_ids):
    targets = [item["record_index_zero_based"] for item in case["core_knowledge"]]
    hits = [index in set(top_ids) for index in targets]
    passed = all(hits) if case["core_match_policy"] == "all_of" else any(hits)
    return passed, hits, targets


def evaluate_case(case, records, hybrid, bm25_index, document_vectors, query_vector):
    query = case["question"]
    bm25_scores, bm25_ranks, bm25_candidates = hybrid.rank_bm25(query, records, bm25_index)
    embedding_ranks, embedding_candidates = hybrid.rank_embedding(
        query_vector, records, document_vectors
    )
    fused = hybrid.reciprocal_rank_fusion(bm25_candidates, embedding_candidates)
    rrf_ranks = {index: rank for rank, (index, _, _, _) in enumerate(fused, start=1)}
    ranked_ids = {
        "BM25": [index for index, _ in bm25_candidates[:TOP_K]],
        "Embedding": [index for index, _ in embedding_candidates[:TOP_K]],
        "RRF": [index for index, _, _, _ in fused[:TOP_K]],
    }
    rank_maps = {"BM25": bm25_ranks, "Embedding": embedding_ranks, "RRF": rrf_ranks}
    print(f"\n{case['case_id']} | {query} | 核心规则={case['core_match_policy']}")
    for method, ids in ranked_ids.items():
        labels = [record_label(records[index]) for index in ids] or ["（无结果）"]
        print(f"  {method} Top-3：{'；'.join(labels)}")

    _, _, targets = core_hit_status(case, ranked_ids["BM25"])
    for item, index in zip(case["core_knowledge"], targets):
        record = records[index]
        label = item.get("pdf_location") or record_label(record)
        ranks = []
        for method, rank_map in rank_maps.items():
            rank = rank_map.get(index)
            text = f"第{rank}名" if rank is not None else "未进融合候选"
            if method == "BM25" and index not in {i for i, _ in bm25_candidates}:
                text += "（未进入阈值候选）"
            ranks.append(f"{method}={text}")
        print(f"  核心知识：{label} | " + "；".join(ranks))

    passed = {}
    for method, ids in ranked_ids.items():
        passed[method], hits, _ = core_hit_status(case, ids)
        status = "通过" if passed[method] else "未通过"
        print(f"  {method}：{status}（核心知识 Top-3 命中：{hits}）")
    return passed


def main():
    try:
        candidate, records, answerable, unanswerable = verify_inputs()
        print(f"输入校验通过：PDF SHA-256、JSONL SHA-256、{len(records)}条记录及全部核心标注一致。")
        hybrid = load_script("t8_hybrid_eval", "07_hybrid_search.py")
        bm25_records, bm25_index = hybrid.bm25_module.build_index()
        live_records, corpus_hash = hybrid.embedding_module.load_corpus()
        if records != bm25_records or records != live_records:
            raise RuntimeError("校验后检索模块读取的知识记录或顺序发生变化，停止评测。")
        if corpus_hash != candidate["source"]["corpus_sha256"]:
            raise RuntimeError("评测期间 JSONL 文件发生变化，停止评测。")

        embedding = hybrid.embedding_module
        api_key = embedding.get_api_key()
        document_vectors = embedding.load_document_vectors(records, corpus_hash, api_key)
        queries = [case["question"] for case in answerable]
        query_vectors = embedding.request_embeddings(queries, api_key)
        if len(query_vectors) != len(answerable):
            raise RuntimeError("查询向量数量与有答案题数不匹配，停止评测。")

        totals = {method: 0 for method in ("BM25", "Embedding", "RRF")}
        for case, query_vector in zip(answerable, query_vectors):
            outcomes = evaluate_case(
                case, records, hybrid, bm25_index, document_vectors, query_vector
            )
            for method, passed in outcomes.items():
                totals[method] += int(passed)

        print("\n题目级 Hit@3（仅本候选集的13道有答案题）：")
        for method, passed_count in totals.items():
            print(f"  {method}: {passed_count} / {len(answerable)}")
        print("\n无答案题（不纳入 Hit@3；当前不判定拒答成败）：")
        for case in unanswerable:
            print(f"  {case['case_id']} | {case['question']} | 暂不评估拒答能力。")
        print("说明：以上是当前小型候选测试集结果，不代表整体系统准确率。")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
