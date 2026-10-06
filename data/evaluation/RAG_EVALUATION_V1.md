# T8 RAG Evaluation V1

This is an offline engineering evaluation, not a production publication path.
No Agent/API/Review/Publication or database persistence functions are called.

## Frozen inputs and labels

`t8_rag_golden_v1.json` has 60 manually authored questions: 50 answerable,
10 unanswerable. Every answerable reference consists of verbatim source excerpts,
with the record index, physical PDF page and text SHA-256 retained. The labels
were frozen before any API evaluation. Its separate `.sha256` file rejects later
edits. Existing 13+2 candidates and 56 expression regressions are not overwritten.
This is not an independently adjudicated multi-annotator dataset.

V1 deliberately tests atomic, single-core-record questions. Recall therefore equals
Hit in this dataset; it must not be presented as a multi-document recall study.

## Metrics

Hit@K: at least one gold record among K results. Recall@K: number of retrieved gold
records divided by number of gold records. MRR: reciprocal rank of the first gold
record, zero if absent. RRF/reranker use their actual bounded candidate lists.
NDCG uses binary relevance, `1/log2(rank+1)` discount and ideal ordering of golds.
Only answerable cases enter retrieval metrics. Missing API results are failures,
excluded from semantic means with their denominators explicitly retained.

Production evidence remains Top3. Hit@5/Recall@5/NDCG@5 are diagnostic statistics
of the full reranker output before the existing Top3 slice; they do not describe
five evidence records being sent to production generation.

Production channel limits: BM25 10 (minimum score 1), Chroma Dense 10, RRF K=60,
RRF candidate cap 10, Reranker Qwen/Qwen3-Reranker-0.6B, final Top3. All methods
use the same production Conservative normalization. Numeric metric computation
from saved rankings is deterministic; external embedding/reranker/generation/
Judge responses are not guaranteed bit-identical on future API calls.

## Chunk experiment interpretation

Isolated fixed character windows: 256/32, 256/64, 512/32, 512/96, 768/96.
Splitting occurs within each already parsed record, preserving page/FAQ boundaries
and all source text. This measures splitting the current extraction, not re-parsing
PDF layouts or optimizing token chunks. Indexes are under `chunk_ablation/`.
Model, BM25, normalization, RRF and reranker parameters are held constant.

Record-projected rankings deduplicate parent records and are **proxy** metrics:
retrieving a fragment of a page does not prove it contains an answer. We additionally
report parent Hit from the **actual three chunks** and character-union coverage
of each frozen reference excerpt within those three chunks. Do not claim a globally
optimal production chunk configuration from this small experiment.
`actual_chunk_slot_metrics` preserves actual chunk positions: duplicate parent
fragments occupy their original slots but receive no repeated relevance credit.

## Generation and Judge

`45_rag_generation_eval.py` reuses the production Grounded QA system prompt,
evidence renderer and generator directly, with saved Top3 ranking snapshots.
It uses original questions for generation and calls only pure `audit_answer()`;
it never calls `process_question()` or `finalize_draft()`. Production generation
parameters and its existing limited retry behavior are unchanged. Final API
failures are recorded; transient failures hidden by that generator's successful
retry cannot be reconstructed from these results.

`46_ragas_judge_eval.py` actually executes official RAGAS 0.4.3 collections
Faithfulness and AnswerRelevancy. SiliconFlow is configured as an OpenAI-compatible
endpoint; no OpenAI provider service is called. Judge model is the existing
deepseek-ai/DeepSeek-V4-Flash, temperature 0, max_tokens 4096, timeout 60s,
OpenAI retries 0 and Instructor max_retries 1. Relevance uses the existing Qwen
embedding model. RAGAS built-in prompts and actual request counts are saved in
per-metric traces, excluding all headers. `reference` is retained for audit;
these two official RAGAS metrics do not take reference as input.

Citation IDs are checked deterministically, including nonexistent, missing and
malformed citations; number validity does not prove semantic support. A full-string
refusal rule is a deliberately strict diagnostic and does not grade paraphrases.
Unanswerable responses receive a separate strict-boolean Judge using question,
answer and actual contexts. This is not misrepresented as a built-in RAGAS metric.
Judge labels are estimates requiring human review, not absolute ground truth.

An initial evaluation-adapter URL error is retained separately in
`results/ragas_eval_v1_adapter_failure.json`; these failures receive no semantic
score. The context-aware abstention recheck preserves previous decisions/traces.

## Run from the project root (PowerShell)

```powershell
.\.venv\Scripts\python.exe scripts\42_validate_rag_golden.py
.\.venv\Scripts\python.exe scripts\43_retrieval_ablation.py
.\.venv\Scripts\python.exe scripts\44_chunk_ablation.py
.\.venv\Scripts\python.exe scripts\45_rag_generation_eval.py
.\.venv-ragas\Scripts\python.exe scripts\46_ragas_judge_eval.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Only for creating the independent Judge environment on another machine:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-ragas
.\.venv-ragas\Scripts\python.exe -m pip install -r data\evaluation\requirements-ragas.txt
```

The full installed evaluation environment is recorded in
`requirements-ragas-lock.txt`. API Key is loaded through existing root `.env`
configuration and never saved in results. Completed cases are checkpointed; rerunning
does not silently replace labels or repeat successful cases. For a new API run,
archive the prior results explicitly and record a new run/version.
