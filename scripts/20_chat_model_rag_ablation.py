"""Read-only main-answer-model race using one fixed production Top-3 per case."""
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULT_PATH = ROOT / "data/evaluation/t8_chat_model_rag_ablation_v1_results.json"
TIMEOUT_SECONDS = 25
MODELS = [
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.6-27B",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Pro",
]
CASE_SPECS = [
    ("RACE-01", "T8-C01", "主刷怎么清理？"),
    ("RACE-02", "T8-C05", None),
    ("RACE-03", "T8-C10", None),
    ("RACE-04", "T8-C08", None),
    ("RACE-05", "T8-C01", "请问石头 T8 的主刷怎么清理？"),
]


def load_script(name, filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"项目模块不可用：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_cases(evaluator):
    candidate, records, _, _ = evaluator.verify_inputs()
    golden = {case["case_id"]: case for case in candidate["cases"]}
    cases = []
    for case_id, golden_id, override_question in CASE_SPECS:
        source = golden[golden_id]
        if source.get("answerability") != "answerable":
            raise RuntimeError(f"{golden_id} 不是有答案候选题。")
        cases.append({
            "case_id": case_id,
            "golden_case_id": golden_id,
            "question": override_question or source["question"],
            "expected_answer_points": source["expected_answer_points"],
            "core_knowledge": source["core_knowledge"],
        })
    return candidate, records, cases


def evidence_snapshot(top_records):
    return [
        {
            "evidence_id": evidence_id,
            "record_index": record_index,
            "pdf_page": record.get("pdf_page"),
            "title": record.get("title"),
            "section": record.get("section"),
            "text": record.get("text", ""),
        }
        for evidence_id, record_index, record in top_records
    ]


def build_user_prompt(qa, question, top_records):
    return (
        f"用户问题：\n{question}\n\n本轮唯一可用的说明书证据如下：\n"
        f"<evidence>\n{qa.render_evidence(top_records)}\n</evidence>\n\n"
        "请按系统规则回答；如果证据未明确回答问题的某个部分，直接说明无法确认，不要推测。"
    )


class GenerationFailure(RuntimeError):
    def __init__(self, status, safe_code):
        self.status = status
        self.safe_code = safe_code


def is_timeout_error(error):
    reason = getattr(error, "reason", None)
    return isinstance(error, (TimeoutError, socket.timeout)) or isinstance(
        reason, (TimeoutError, socket.timeout)
    )


def generate_once(qa, question, top_records, model, api_key):
    from app.config import CHAT_URL

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": qa.SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(qa, question, top_records)},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": qa.MAX_TOKENS,
    }, ensure_ascii=False).encode("utf-8")
    request = Request(CHAT_URL, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise GenerationFailure("error", f"http_{error.code}") from None
    except (URLError, TimeoutError, socket.timeout, OSError) as error:
        status = "timeout" if is_timeout_error(error) else "error"
        raise GenerationFailure(status, type(error).__name__) from None
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        raise GenerationFailure("error", "invalid_response") from None

    try:
        answer = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise GenerationFailure("error", "missing_text") from None
    if not isinstance(answer, str) or not answer.strip():
        raise GenerationFailure("error", "empty_text")
    return answer.strip()


def deterministic_audit(qa, question, answer, top_records):
    result = qa.audit_answer(question, answer, top_records)
    return {
        "status": result["status"],
        "citation_status": result["citation_status"],
        "citation_ids": result["citation_ids"],
        "invalid_citations": result["invalid_citations"],
        "semantic_support": result["semantic_support"],
        "reasons": result["reasons"],
    }


def record_label(record):
    return record.get("title") or record.get("section") or "（无标题）"


def retrieve_fixed_evidence(qa, cases, pipeline, records):
    from app.services.qa_service import normalize_retrieval_query

    fixed = []
    for case in cases:
        retrieval_query = normalize_retrieval_query(case["question"])
        top_records = qa.retrieve(retrieval_query, pipeline)
        if len(top_records) != qa.RERANK_TOP_K:
            raise RuntimeError(f"{case['case_id']} 没有得到生产 Top-3，停止赛马。")
        fixed.append({
            **case,
            "retrieval_query": retrieval_query,
            "top_records": top_records,
            "evidence": evidence_snapshot(top_records),
        })
        labels = [
            f"E{rank}:第{record.get('pdf_page')}页《{record_label(record)}》"
            for rank, (_, _, record) in enumerate(top_records, 1)
        ]
        print(f"固定证据 {case['case_id']} | {case['question']} | " + "；".join(labels))
    return fixed


def run_race(qa, cases, api_key, candidate):
    results = []
    for model in MODELS:
        for case in cases:
            for attempt in (1, 2):
                started = time.perf_counter()
                try:
                    answer = generate_once(
                        qa, case["question"], case["top_records"], model, api_key
                    )
                    audit = deterministic_audit(
                        qa, case["question"], answer, case["top_records"]
                    )
                    row = {
                        "model": model,
                        "case_id": case["case_id"],
                        "question": case["question"],
                        "attempt": attempt,
                        "status": "success",
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                        "answer": answer,
                        "has_valid_text": True,
                        "deterministic_audit": audit,
                    }
                except GenerationFailure as error:
                    row = {
                        "model": model,
                        "case_id": case["case_id"],
                        "question": case["question"],
                        "attempt": attempt,
                        "status": error.status,
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                        "answer": None,
                        "has_valid_text": False,
                        "safe_error_code": error.safe_code,
                    }
                results.append(row)
                save_report(candidate, cases, results, complete=False, announce=False)
                print(
                    f"\n{model} | {case['case_id']} | attempt={attempt} | "
                    f"{row['status']} | {row['elapsed_seconds']:.2f}s"
                )
                if row["status"] == "success":
                    print(f"自动审查：{row['deterministic_audit']['status']}；"
                          f"语义支持={row['deterministic_audit']['semantic_support']}")
                if row["status"] != "timeout" or attempt == 2:
                    break
    return results


def final_rows_for_model(model, results):
    rows = {}
    for row in results:
        if row["model"] == model:
            rows[row["case_id"]] = row
    return rows


def report_results(cases, results):
    print("\n模型汇总（成功按5道题去重；延迟为成功请求的平均耗时）：")
    print("Model | Success | Timeout | Avg Latency | 程序拦截 | JSON/文本")
    for model in MODELS:
        attempts = [row for row in results if row["model"] == model]
        final = final_rows_for_model(model, results)
        successes = [row for row in final.values() if row["status"] == "success"]
        successful_attempts = [row for row in attempts if row["status"] == "success"]
        latency = (sum(row["elapsed_seconds"] for row in successful_attempts)
                   / len(successful_attempts)) if successful_attempts else None
        blocked = sum(
            row.get("deterministic_audit", {}).get("status", "").startswith("PROGRAM_BLOCKED_")
            for row in successes
        )
        average = f"{latency:.2f}s" if latency is not None else "N/A"
        print(f"{model} | {len(successes)}/5 | "
              f"{sum(row['status'] == 'timeout' for row in attempts)} | "
              f"{average} | {blocked}/{len(successes)} | {len(successes)}/{len(successes)}")

    print("\n逐题四模型回答对照（原始模型输出；质量结论需结合证据人工核验）：")
    for case in cases:
        print(f"\n{case['case_id']} | {case['question']}")
        for model in MODELS:
            row = final_rows_for_model(model, results).get(case["case_id"])
            if row is None or row["answer"] is None:
                state = row["status"] if row else "未运行"
                print(f"\n{model} | {state} | 无有效文本")
            else:
                print(f"\n{model} | {row['elapsed_seconds']:.2f}s | "
                      f"{row['deterministic_audit']['status']}\n{row['answer']}")


def save_report(candidate, cases, results, complete=False, announce=True):
    report = {
        "experiment": "t8_chat_model_rag_ablation_v1",
        "run_status": "complete" if complete else "partial",
        "dataset_id": candidate["dataset_id"],
        "generation_parameters": {
            "temperature": 0.1,
            "max_tokens": 1200,
            "timeout_seconds": TIMEOUT_SECONDS,
            "retry_policy": "timeout only; one retry maximum",
        },
        "cases": [
            {key: value for key, value in case.items() if key != "top_records"}
            for case in cases
        ],
        "attempts": results,
        "quality_note": (
            "audit_answer provides deterministic risk checks only; citation validity does not "
            "prove semantic support. Groundedness, relevance, and completeness require manual review."
        ),
    }
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if announce:
        print(f"\n实验答案与固定 Evidence 已保存：{RESULT_PATH}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    try:
        evaluator = load_script("t8_chat_race_evaluator", "08_evaluate_retrieval.py")
        qa = load_script("t8_chat_race_qa", "12_grounded_qa.py")
        candidate, records, cases = prepare_cases(evaluator)
        pipeline = qa.create_pipeline()
        if records != pipeline[2]:
            raise RuntimeError("评测校验与生产检索读取的知识库顺序不一致。")

        print("候选集 PDF/JSONL 指纹和标注校验通过；读取现有 Embedding 缓存，不重建。")
        print(f"回答 Prompt=生产 Grounded QA Prompt；固定参数 temperature=0.1，"
              f"max_tokens={qa.MAX_TOKENS}，单次超时={TIMEOUT_SECONDS}s。")
        cases = retrieve_fixed_evidence(qa, cases, pipeline, records)
        results = run_race(qa, cases, pipeline[-1], candidate)
        report_results(cases, results)
        save_report(candidate, cases, results, complete=True)
        return 0
    except Exception as error:
        # Never print exception text: upstream errors may contain request details.
        print(f"实验停止：{type(error).__name__}。未输出敏感请求或密钥。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
