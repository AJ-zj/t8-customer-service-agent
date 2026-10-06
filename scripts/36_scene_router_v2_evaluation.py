"""Frozen prompt-only Router regression with transport/schema failure observation."""

import hashlib
import argparse
import json
from pathlib import Path
import runpy
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.agent.llm_router import IntentDecision, LLMIntentRouter, LLMRouterError, ROUTER_SYSTEM_PROMPT
from app.config import ROUTER_MODEL
from pydantic import ValidationError
from app.agent.ambiguity import needs_router_clarification

summarize = runpy.run_path(str(ROOT / "scripts/35_scene_router_evaluation.py"))["summarize"]


class ObservedTransport:
    def __init__(self):
        self.diagnostics = {}

    def __call__(self, request, *, timeout):
        try:
            with urlopen(request, timeout=timeout) as response:
                status, body = response.status, response.read()
            self.diagnostics["http_status"] = status
        except HTTPError as error:
            self.diagnostics.update(failure_type="HTTP_ERROR", http_status=error.code)
            raise
        except (TimeoutError, socket.timeout):
            self.diagnostics["failure_type"] = "TIMEOUT"
            raise
        except URLError as error:
            self.diagnostics["failure_type"] = (
                "TIMEOUT" if isinstance(error.reason, TimeoutError) else "NETWORK_ERROR")
            raise
        except OSError:
            self.diagnostics["failure_type"] = "NETWORK_ERROR"
            raise
        try:
            envelope = json.loads(body)
            content = envelope["choices"][0]["message"]["content"]
        except json.JSONDecodeError:
            self.diagnostics["failure_type"] = "UPSTREAM_JSON_ERROR"
        except (KeyError, IndexError, TypeError):
            self.diagnostics["failure_type"] = "RESPONSE_SCHEMA_ERROR"
        else:
            if not isinstance(content, str) or not content.strip():
                self.diagnostics["failure_type"] = "EMPTY_OR_NON_STRING_CONTENT"
            else:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    self.diagnostics["failure_type"] = "CONTENT_JSON_ERROR"
                    self.diagnostics["markdown_fence"] = content.strip().startswith("```")
                else:
                    try:
                        IntentDecision.model_validate(parsed)
                    except ValidationError as error:
                        # Only error kinds/field locations, never input values or response body.
                        self.diagnostics.update(failure_type="SCHEMA_ERROR", schema_errors=[
                            {"type": e["type"], "location": e["loc"]} for e in error.errors()])

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return body

        result = Response()
        result.status = status
        return result


def evaluate(dataset, output, guard=False):
    source = dataset.read_bytes()
    rows = []
    transport = ObservedTransport()
    # Keep model-only outputs visible; apply the same production guard explicitly below.
    classifier = LLMIntentRouter(transport=transport, ambiguity_guard=False)
    for case in json.loads(source)["cases"]:
        transport.diagnostics = {}
        row, start = dict(case), time.perf_counter()
        try:
            decision = classifier.classify(case["question"])
            row["model_intent"] = decision.intent
            if guard and needs_router_clarification(case["question"]):
                decision = IntentDecision(intent="clarify", reason="问题是极短泛化描述，缺少明确业务目标或具体症状。")
                row["guard_applied"] = True
            row.update(status="SUCCESS", intent=decision.intent, reason=decision.reason)
        except LLMRouterError:
            row.update(status="ERROR")
        row.update(elapsed_seconds=time.perf_counter() - start, diagnostics=transport.diagnostics)
        rows.append(row)
        output.write_text(json.dumps({
            "model": ROUTER_MODEL, "variant": "guard" if guard else "prompt_only", "temperature": 0,
            "dataset_sha256": hashlib.sha256(source).hexdigest(),
            "prompt_sha256": hashlib.sha256(ROUTER_SYSTEM_PROMPT.encode()).hexdigest(),
            "attempts": rows, "summary": summarize(rows),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(case["case_id"], row["status"], row.get("intent", "ERROR"),
              round(row["elapsed_seconds"], 3), flush=True)
    if dataset.read_bytes() != source:
        raise RuntimeError("Frozen dataset changed during evaluation.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", action="store_true")
    args = parser.parse_args()
    base = ROOT / "data/evaluation"
    old = json.loads((base / "scene_router_v1_results.json").read_text(encoding="utf-8"))
    rows = evaluate(base / "scene_router_v1.json", base / (
        "scene_router_v2_guard_results.json" if args.guard else "scene_router_v2_prompt_only_results.json"), args.guard)
    old_rows = {row["case_id"]: row for row in old["attempts"]}
    fixes, regressions, remaining = [], [], []
    for row in rows:
        previous = old_rows[row["case_id"]]
        old_ok = previous.get("intent") == previous["expected_intent"]
        new_ok = row.get("intent") == row["expected_intent"]
        detail = {"case_id": row["case_id"], "question": row["question"],
                  "gold": row["expected_intent"], "v1": previous.get("intent", "ERROR"),
                  "v2": row.get("intent", "ERROR")}
        if not old_ok and new_ok:
            fixes.append(detail)
        if old_ok and not new_ok:
            regressions.append(detail)
        if not new_ok:
            remaining.append(detail)
    summary = {"v1": old["summary"], "v2": summarize(rows),
               "fixed": fixes, "regressions": regressions, "remaining_errors": remaining,
               "v1_contract_failure_evidence": {
                   "case_id": "S20", "elapsed_seconds": 25.058410800003912,
                   "failure_type": "UNKNOWN_NOT_RECORDED",
                   "note": "Historical result retained only sanitized ERROR; latency alone cannot prove timeout."}}
    (base / ("scene_router_v2_guard_comparison.json" if args.guard else "scene_router_v2_comparison.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    evaluate(base / "scene_router_hard_v1.json", base / (
        "scene_router_hard_v1_guard_results.json" if args.guard else "scene_router_hard_v1_results.json"), args.guard)


if __name__ == "__main__":
    main()
