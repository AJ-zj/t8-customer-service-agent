# FINAL PROJECT TECHNICAL INVENTORY V1

以石头T8官方说明书为知识源的多场景Agent/RAG客服，以可追溯证据、安全即时交付和独立人工审核/发布提供有依据的回答。

源码/配置/正式JSON盘点及测试复跑。本轮模型调用0、业务库写入0、生产逻辑修改0。不是README、架构图或简历稿。

## 技术栈与模型

{
  "python": {
    "version": "3.12.3",
    "source": "当前venv；Docker python:3.12-slim"
  },
  "backend": {
    "fastapi": {
      "version": "0.142.2",
      "source": "requirements-runtime.txt/current venv"
    },
    "pydantic": {
      "version": "2.13.5",
      "source": "requirements-runtime.txt/current venv"
    },
    "langgraph": {
      "version": "1.2.12",
      "source": "requirements-runtime.txt/current venv"
    },
    "SQLAlchemy": {
      "version": "2.1.3",
      "source": "requirements-runtime.txt/current venv"
    },
    "psycopg": {
      "version": "3.3.6",
      "source": "requirements-runtime.txt/current venv"
    },
    "chromadb": {
      "version": "1.5.9",
      "source": "requirements-runtime.txt/current venv"
    },
    "jieba": {
      "version": "0.42.1",
      "source": "requirements-runtime.txt/current venv"
    },
    "rank-bm25": {
      "version": "0.2.2",
      "source": "requirements-runtime.txt/current venv"
    },
    "uvicorn": {
      "version": "0.54.0",
      "source": "requirements-runtime.txt/current venv"
    },
    "python-dotenv": {
      "version": "1.2.4",
      "source": "requirements-runtime.txt/current venv"
    },
    "langchain-core": {
      "version": "1.6.6",
      "source": "requirements-runtime.txt/current venv"
    }
  },
  "frontend": {
    "react": {
      "version": "19.3.0",
      "source": "frontend/package-lock.json"
    },
    "react-dom": {
      "version": "19.3.0",
      "source": "frontend/package-lock.json"
    },
    "react-router-dom": {
      "version": "7.18.4",
      "source": "frontend/package-lock.json"
    },
    "vite": {
      "version": "7.3.7",
      "source": "frontend/package-lock.json"
    },
    "vitest": {
      "version": "3.2.7",
      "source": "frontend/package-lock.json"
    }
  },
  "language": "JavaScript/JSX",
  "deployment": {
    "DockerEngine": "29.6.2 current read-only",
    "Compose": "5.3.1 current read-only",
    "Postgres": "postgres:16 tag; patch UNVERIFIED",
    "Nginx": "nginx:alpine; patch UNVERIFIED",
    "NodeBuild": "node:20-alpine"
  }
}

{
  "router": [
    "Qwen/Qwen2.5-7B-Instruct",
    "app/config.py"
  ],
  "generation": [
    "deepseek-ai/DeepSeek-V4-Flash",
    "app/config.py/app/agent/tool_calling.py"
  ],
  "backup": [
    "Qwen/Qwen3.8-27B",
    "record only, no fallback"
  ],
  "embedding": [
    "Qwen/Qwen3-Embedding-0.6B",
    "app/retrieval/t8_retrieval.py"
  ],
  "reranker": [
    "Qwen/Qwen3-Reranker-0.6B",
    "app/retrieval/t8_retrieval.py"
  ],
  "semantic_gate": [
    "Qwen/Qwen2.5-7B-Instruct",
    "app/config.py"
  ]
}

备用Qwen3.8只记录，不fallback。Frontend确切版本来自package-lock，不是package.json范围。

## 知识库与Golden

{
  "name": "Roborock T8 Chinese official manual",
  "pdf": "data/raw/roborock/Roborock_T8_User_Manual_ZH.pdf",
  "pages": 26,
  "records": 41,
  "pdf_sha256": "308f4e710ee18e1b670bf3b1448f854eb2edbe29d1f119b027fcc4880ea92b1f",
  "corpus_sha256": "a8be9a479c1bfc7438d79cbc03c954b18fd52b4863468ff0bbfcbf40eaad7d12",
  "golden": {
    "total": 60,
    "answerable": 50,
    "unanswerable": 10,
    "sha256": "547fd0f48197bc150640100fb07ba6287fee5d0f4439c1e96c2950c2167e9386",
    "historical_expected_matches": true,
    "source_file": "data/evaluation/t8_rag_golden_v1.json"
  }
}

## 27模块清单

| 模块 | 实现 | 文件 | 职责 | 默认启用 | 限制 |
| --- | --- | --- | --- | --- | --- |
| Scene Router | True | app/agent/llm_router.py | 五业务场景及clarify/handoff严格分类，上下文和fallback | 按请求 | 25s模型；契约成功≠准确率 |
| Scene Policy | True | app/agent/scene_policy.py | 固定场景权限与输出模式 | 是 | 仅五个固定业务场景 |
| Troubleshooting Intake | True | app/agent/troubleshooting.py | ready/needs clarification及错误码fast path | 故障时 | ready不说明故障码含义有据 |
| ReAct Agent | True | app/agent/react_graph.py | 有界Tool循环、单Evidence Snapshot、finalize一次 | 是 | Tool最多两次，非通用自治 |
| Manual Search Tool | True | app/agent/tools.py | 只读T8说明书Top3证据 | 是 | 需外部Embedding/Reranker |
| Query Normalization | True | app/services/qa_service.py | 删除开头请问/石头/T8及空格 | 是 | 不删除“的”，不语义改写 |
| BM25 Retrieval | True | app/retrieval/t8_retrieval.py | jieba+BM25阈值Top10 | 是 | 最低分1.0 |
| Chroma Dense Retrieval | True | app/retrieval/t8_vector_store.py | 外部query向量、本地cosine索引 | 是 | 校验已有索引不自动重建 |
| RRF Fusion | True | app/retrieval/t8_retrieval.py | RRF K60候选Top10 | 是 | 单独RRF未优于BM25 |
| Qwen Reranker | True | app/retrieval/t8_retrieval.py | 校验映射后排序Top3 | 是 | 分数非正确概率 |
| Grounded Generation | True | app/agent/tool_calling.py | DeepSeek依据当轮Tool Evidence生成 | 是 | provider stream false |
| Citation Audit | True | app/services/grounded_qa.py | 引用编号和程序风险判定 | 是 | 编号合法不等于语义支持 |
| Program Safety Audit | True | app/services/grounded_qa.py | 金额/个人运行数据等规则 | 是 | 确定性规则非事实裁判 |
| Static Boundary | True | app/services/static_knowledge_boundary.py | 静态说明书与个人实时信息边界 | 是 | 小型规则，不是中文parser |
| Semantic Grounding Gate | True | app/services/semantic_grounding_gate.py | 核验事实和核心问题，失败关闭 | 条件 | LLM Judge，有限schema retry |
| Answer Release Policy | True | app/services/answer_release_policy.py | 决定即时auto/review/abstain | 是 | 仅部分scene可自动交付 |
| Review Workflow | True | app/services/review_service.py | 只读审核，pending原子决策 | 是 | 共享token非正式IAM |
| Explicit Publication | True | app/services/review_service.py | approved后单独显式发布 | 是 | 没有自动发布 |
| Session Memory | True | app/agent/memory.py | 6消息/100会话，按会话串行 | 带session | 进程内 |
| Long-term Extraction | True | app/memory/extractor.py | 严格抽取原文候选 | 有owner时 | 失败安全降级 |
| Long-term Dedup | True | app/memory/service.py | owner/hash+semantic去重、原子计数 | 写事实时 | 不同hash并发近重复不保证 |
| Memory Lifecycle | True | app/memory/lifecycle.py | importance/TTL/decay/显式清理 | 写入/清理时 | 无后台scheduler |
| React Chat Frontend | True | frontend/src/pages/ChatPage.jsx | 聊天、SSE和会话交互 | 是 | Abort不保证取消后端 |
| Review Admin Frontend | True | frontend/src/pages/ReviewPage.jsx | 草稿/证据核对与决策/发布 | 是 | token仅state |
| Published Page | True | frontend/src/pages/PublishedPage.jsx | 公开显示已发布答案 | 是 | 未发布404 |
| Safe SSE | True | app/main.py | 共用编排、安全后再分块 | 是 | 不是provider token流 |
| Docker/Nginx | True | docker-compose.yml | Postgres/db-init/API/Nginx UI | Compose | 中文路径BuildKit受限 |

## 实际数据流

React JSON/SSE→validate→session lock/context → Router→Scene Policy→conditional troubleshooting intake → bounded ReAct→search_t8_manual → contextual query from program→normalization → BM25 Top10+Chroma dense Top10→RRF60 Top10→Reranker Top3 → fixed Tool Evidence observation→Draft → finalize_draft deterministic Audit→one Review commit → Static Boundary on original question→preliminary Release → Semantic Gate only when required→final Release → safe live response/SSE buffer_then_release→visible session exchange → optional owner extraction/filter/original quote persistence

Audit+Review happen inside Graph before Boundary/Gate/live release in API; no double process_question/retrieval.

Audit和Review入库发生在Graph；Boundary/Gate/Release在API orchestration后。Boundary不是检索前拦截。四层职责分别为知识能力范围、确定性风险、Evidence语义支持、即时交付决策。Program Block优先，Gate错误失败关闭。clarify/handoff/Intake澄清旁路生成和Review。

## Memory

{
  "source_file": [
    "app/agent/memory.py",
    "app/memory/runtime.py",
    "app/memory/service.py",
    "app/memory/lifecycle.py",
    "app/main.py",
    "app/config.py"
  ],
  "working": "internal graph messages/Snapshot/Draft not session transcript",
  "session": {
    "messages": 6,
    "sessions": 100,
    "process_local": true,
    "same_session_serialized": true,
    "different_sessions_concurrent": true,
    "visible_user_exchange_only": true,
    "client_id_not_auth": true
  },
  "long_term": {
    "categories": [
      "device_profile",
      "home_environment",
      "usage_preference"
    ],
    "injection_default": false,
    "storage": "when memory_owner_id supplied; extraction/filter/save",
    "persisted": "validated source_quote.strip() only",
    "dedup": "owner+hash uniqueness; cosine semantic .97; atomic SQL seen_count",
    "similarity_threshold": 0.75,
    "top_k": 3,
    "explicit_injection_role": "user data; not system/evidence; excluded from Router/query",
    "lifecycle": {
      "category_importance_ttl_days": {
        "device_profile": [
          0.9,
          180
        ],
        "home_environment": [
          0.75,
          90
        ],
        "usage_preference": [
          0.65,
          60
        ],
        "uncategorized": [
          0.5,
          30
        ]
      },
      "repeat_bonus_step": 0.02,
      "max": 0.1,
      "decay": "importance*0.5**(age/(TTL/2))*(1+min(.20,.02*(seen-1))) cap 0..1, expired 0",
      "refresh": "atomic seen_count/last_seen/expiry",
      "search": "expired excluded; cosine only",
      "cleanup": "explicit, no scheduler"
    }
  },
  "summary": "长期事实抽取/存储/生命周期已实现；自动注入Agent默认关闭。"
}

长期事实抽取/存储/生命周期已实现；自动注入Agent默认关闭。

## Review/Publication

generated、pending_review、approved/rejected、published是不同状态。auto_released≠approved≠published；approved≠published。即时答复可能有answer但published=false，review_required时answer=null。仅显式publish后公共GET可读，未发布返回404。来源app/main.py、review_service.py、models.py。

## Retrieval配置

{
  "source_file": [
    "app/retrieval/t8_retrieval.py",
    "app/retrieval/t8_vector_store.py",
    "app/services/readiness.py"
  ],
  "bm25_top10": 10,
  "bm25_min_score": 1.0,
  "dense_top10": 10,
  "rrf_k": 60,
  "rrf_output_top10": 10,
  "reranker_top3": 3,
  "embedding_dimension": 1024,
  "collection": "t8_manual_qwen3_embedding_0_6b",
  "distance": "cosine",
  "records": 41,
  "persist": "data/vector_store/chroma",
  "production_chunk": "41 source/FAQ records, separate from char chunk ablations"
}

黄金60题=50 answerable/10 unanswerable；SHA 547fd0f48197bc150640100fb07ba6287fee5d0f4439c1e96c2950c2167e9386。正式retrieval source data/evaluation/results/retrieval_ablation_v1.json。

| 方法 | Hit1 | Hit3 | Recall3 | MRR | NDCG3 | Hit5 | NDCG5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.74 | 0.9 | 0.9 | 0.8312 | 0.8357 | 0.94 | 0.8529 |
| Dense | 0.66 | 0.84 | 0.84 | 0.7607 | 0.7683 | 0.86 | 0.7769 |
| RRF | 0.74 | 0.88 | 0.88 | 0.8131 | 0.8179 | 0.88 | 0.8179 |
| Reranker | 0.88 | 0.98 | 0.98 | 0.9307 | 0.9405 | 1 | 0.9482 |

RRF单独Hit@3 .88低于BM25 .90，不能宣称RRF显著提升。Reranker结果Hit@1 44/50，Hit@3 49/50，Hit@5 50/50；生产Evidence仍Top3。

### Chunk ablation

| size/overlap chars | count | parent Hit3 | parent MRR | actual slot MRR | coverage |
| --- | --- | --- | --- | --- | --- |
| 256/32 | 63 | 0.98 | 0.944 | 0.94333 | 0.98 |
| 256/64 | 66 | 0.98 | 0.94167 | 0.94067 | 0.97995 |
| 512/32 | 48 | 0.98 | 0.944 | 0.944 | 0.98 |
| 512/96 | 48 | 0.98 | 0.945 | 0.944 | 0.98 |
| 768/96 | 44 | 1 | 0.93667 | 0.93667 | 1.0 |

来源chunk_ablation_v1.json；父记录Hit是proxy，slot和reference coverage另算。768/96命中100%但MRR非最高，不能称全局最优；生产仍41条记录。

## Generation/RAGAS

来源generation_eval_v1/ragas_eval_v1/rag_evaluation_summary_v1.json；RAGAS0.4.3，Judge DeepSeek-V4-Flash。

| 范围 | Faithfulness均值 | 有效/失败 | Relevance均值 | 有效/失败 |
| --- | --- | --- | --- | --- |
| Full60 | 0.86666 | 59/1 | 0.73299 | 60/0 |
| Answerable50 | 0.9009 | 49/1 | 0.81055 | 50/0 |

有1个Faithfulness APITimeout排除均值。历史拒答Judge10/10；引用编号60/60，answerable引用presence50/50，严格引用要求56/60=.9333。RAGAS/LLM Judge不是人工准确率。

## Router / Troubleshooting

Router：65/70 accuracy=92.86%，Macro F1=0.9388；contract=69/70=98.57%。Clarify F1=1。hard集19/20，1 timeout，不能把contract称accuracy。来源scene_router_v2_guard_results和scene_router_hard_v1_guard_results。

Troubleshooting：30/30原集、20/20 hard；两组false-ready/false-clarify为0；multi-turn 5/5。错误码fast path只判ready，不证明码义。来源troubleshooting_intake_v1_2_results.json。

## Answer Release回放

本轮当前Audit/Boundary/Policy复用V1.1.3保存Gate，不调用模型：

| 集合 | auto | review | safe |
| --- | --- | --- | --- |
| answerable | 49 | 1 | 0 |
| unanswerable | 0 | 0 | 10 |

answerable自动交付49/50=98%，不是准确率；false_boundary=0，unanswerable incorrect_auto_release=0；G054当前重放safe_abstention。没有独立V1.1.5结果文件，标注为本地策略重放。V1.1.3真实E2E的Case B为review_required；其他版本/replay的草稿不同，不覆盖该真实记录。

## SSE

{
  "source_file": [
    "app/main.py",
    "frontend/src/api/sse.js",
    "frontend/src/pages/ChatPage.jsx",
    "frontend/nginx.conf",
    "data/evaluation/results/safe_sse_streaming_v1.json"
  ],
  "endpoint": "POST /api/agent/chat/stream",
  "events": [
    "status",
    "answer_start",
    "answer_delta",
    "review_required",
    "done",
    "error",
    "heartbeat_comment"
  ],
  "heartbeat_s": 9,
  "chunk_chars": 14,
  "provider_streaming": false,
  "client_sse_streaming": true,
  "mode": "buffer_then_release",
  "nginx": {
    "loaded": true,
    "exact_sse_location": true,
    "http11": true,
    "buffering_off": true,
    "cache_off": true,
    "read_timeout_180": true,
    "accel_buffering_no": true
  },
  "case_A_first_ms": 110.0,
  "case_A_done_ms": 9407.0,
  "precheck_delta": 0,
  "review_required_delta": 0,
  "max_review_delta": 1,
  "memory_writes_per_request": 1,
  "nginx_e2e_pass": true,
  "limitation": "first event is status; abort may leave worker running",
  "legacy_summary_conflict": false
}

provider_streaming=false、client_sse=true、buffer_then_release。首事件110ms是状态，Case A完成9407ms。安全链完成前delta=0，待审核delta=0。Nginx关闭该endpoint buffering。Browser abort可能留下后台worker。结果JSON有旧observer字段矛盾，见metric_conflicts。

## Tests 与真实E2E

{
  "python": {
    "total": 470,
    "passed": 470,
    "failed": 0,
    "skipped": 0,
    "duration_seconds": 18.948,
    "command": ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v"
  },
  "frontend": {
    "total": 48,
    "passed": 48,
    "failed": 0,
    "skipped": 0,
    "duration_seconds": 36.64,
    "command": "node node_modules/vitest/vitest.mjs run --configLoader runner --pool=threads --maxWorkers=1 --minWorkers=1",
    "execution": "完整ASCII临时副本含node_modules，未重装；运行后清除",
    "note": "首个复制命令误排除依赖dist无法启动；修正复制后全测试通过"
  },
  "classes": "unit/integration tests、离线benchmark、真实E2E分别统计"
}

Frontend在未重装依赖的ASCII临时副本跑；首次测试副本排除node_modules内dist导致启动错误，修正后48项全通过。

历史Final System E2E：12/12首次通过，retry 0，external failure 0，business failure 0。涵盖多场景、多轮、审核与发布。本轮未重跑。

## Deployment / Frontend

{
  "source_file": [
    "Dockerfile",
    "docker-compose.yml",
    "frontend/Dockerfile",
    "frontend/nginx.conf",
    "scripts/50_initialize_database.py",
    "data/evaluation/results/deployment_v1_1_e2e.json"
  ],
  "services": [
    "postgres",
    "db-init",
    "api",
    "frontend"
  ],
  "binding": "127.0.0.1",
  "ports": {
    "frontend": 8080,
    "api": 8000,
    "postgres": 5433
  },
  "db": "roborock; shop_demo untouched",
  "db_init": [
    "14_create_review_table.py",
    "21_create_memory_facts_table.py",
    "32_upgrade_review_publication.py",
    "40_upgrade_memory_lifecycle.py"
  ],
  "init": "additive/idempotent sequential; asserts actual DB name",
  "volumes": "Postgres named; Chroma bind",
  "uid_gid": 10001,
  "health": "liveness only",
  "ready": "DB/schema+41 corpus+same fingerprint Chroma+1024 dimension; no model/rebuild",
  "historical_image_sizes": {
    "api": 212971706,
    "frontend": 26420107
  },
  "image_size_note": "historical Deployment V1.1, not current exact",
  "build_limitation": "This Docker Desktop still fails BuildKit session setup for the non-ASCII project path. Identical Dockerfiles passed with an ASCII temporary build context; no application workaround or user manual-copy startup step was introduced.",
  "current_containers": "read-only: 9 running total, 8 acceptance containers healthy, one db-init exited, pg-crud-demo unchanged",
  "review_token": "React state only, request header, not storage/URL/bundle"
}

health只作liveness，ready核验roborock、schema、41条corpus和1024维索引。Review Token仅React state并经header发送，不入浏览器持久存储/URL/bundle。部署结果的镜像大小为历史值。构建记录有3项漏洞告警(1 moderate/2 critical)，本轮未审计解决状态。

## Safety code map

| 控制 | 文件 | 边界 |
| --- | --- | --- |
| Citation/Money/Personal Runtime Audit | app/services/grounded_qa.py | valid citation not entailment |
| Static source boundary | app/services/static_knowledge_boundary.py | limited rules |
| Semantic grounding fail closed | app/services/semantic_grounding_gate.py | LLM judge, failure review |
| Safe abstention/program priority | app/services/answer_release_policy.py | block precedes gate |
| Explicit publication | ['app/services/review_service.py', 'app/models.py'] | approved!=published; unpublished GET 404 |
| Admin token | app/main.py | X-Review-Token and compare_digest |
| Frontend token | ['frontend/src/pages/ReviewPage.jsx', 'frontend/src/api/client.js'] | state/header only |
| Untrusted Memory/tool query | ['app/agent/react_graph.py', 'app/agent/tool_calling.py'] | user data, model arg not query control |
| SSE safe allowlist | app/main.py | answer deltas only after full safety processing |

## 指标冲突

| 项目 | 说明 | 来源 |
| --- | --- | --- |
| 引用指标口径 | 编号合法60/60、answerable presence50/50、严格引用要求56/60=.9333是三种检查，不合并成100%。 | ['data/evaluation/results/generation_eval_v1.json', 'data/evaluation/results/rag_evaluation_summary_v1.json'] |
| Release V1.1.5 | 没有单独结果JSON；本次当前Audit/Boundary/Policy重放V1.1.3缓存Gate，非真实模型复测。 | data/evaluation/results/answer_release_policy_v1_1_3.json |
| Case B | 真实V1.1.3 Case B是review_required；早期修复验收/replay用不同草稿，auto不能覆盖现场结果。 | ['data/evaluation/results/answer_release_policy_v1_1_3.json', 'data/evaluation/results/answer_release_diagnostics_v1.json'] |
| SSE汇总字段 | overall_pass=true但all_cases_have_received_processing_done=false；同文件后续mock/frame-order验证通过，保留旧observer冲突。 | data/evaluation/results/safe_sse_streaming_v1.json |
| 测试数量 | 旧407/29为历史，本次实跑470/48。 | ['tests', 'frontend/tests'] |
| npm版本 | package.json semver范围不是安装版本；按package-lock报告。 | ['frontend/package.json', 'frontend/package-lock.json'] |

## UNVERIFIED

- 逐模块冻结批准: 缺少独立仓库批准记录，不能因PASS标COMPLETE/FROZEN。
- 当前容器源码/镜像尺寸一致性: 未重建并对照；镜像大小用历史正式结果。
- Nginx/Postgres精确patch: nginx:alpine、postgres:16为浮动标签。
- V1.1.5独立60题真实Gate评测: 无结果JSON；本次仅本地policy replay。
- 依赖漏洞修复状态: 有历史构建告警，本轮未联网审计。

## 阶段证据

| Component | Stage | Evidence | status |
| --- | --- | --- | --- |
| Vector Store | V1 | data/evaluation/t8_chroma_dense_ab_v1_results.json | UNVERIFIED: no standalone repository approval artifact |
| Scene Router | V2.1 | data/evaluation/scene_router_v2_guard_results.json | UNVERIFIED: no standalone repository approval artifact |
| Troubleshooting | V1.2 | data/evaluation/troubleshooting_intake_v1_2_results.json | UNVERIFIED: no standalone repository approval artifact |
| Memory V1 | V1 | app/agent/memory.py | UNVERIFIED: no standalone repository approval artifact |
| Memory V2 | V2 | app/config.py | UNVERIFIED: no standalone repository approval artifact |
| Memory Lifecycle | V3 | app/memory/lifecycle.py | UNVERIFIED: no standalone repository approval artifact |
| RAG Evaluation | V1 | data/evaluation/results/rag_evaluation_summary_v1.json | UNVERIFIED: no standalone repository approval artifact |
| Final System E2E | V1 | data/evaluation/results/final_system_e2e_v1.json | UNVERIFIED: no standalone repository approval artifact |
| Deployment | V1.1 | data/evaluation/results/deployment_v1_1_e2e.json | UNVERIFIED: no standalone repository approval artifact |
| Frontend | V1 | data/evaluation/results/frontend_v1_e2e.json | UNVERIFIED: no standalone repository approval artifact |
| Answer Release | V1.1.5 | tests/test_static_knowledge_boundary.py | UNVERIFIED: no standalone repository approval artifact |
| Safe SSE | V1 | data/evaluation/results/safe_sse_streaming_v1.json | UNVERIFIED: no standalone repository approval artifact |

## Resume-Safe Claims

| Claim | Evidence | 指标含义 | 安全表述 |
| --- | --- | --- | --- |
| T8知识库 | ['data/processed/t8_pages.jsonl', 'data/raw/roborock/Roborock_T8_User_Manual_ZH.pdf'] | 26页/41记录由当前文件计数 | 构建基于26页官方说明书的41条可追溯知识记录。 |
| Hybrid retrieval | ['data/evaluation/results/retrieval_ablation_v1.json', 'app/retrieval/t8_retrieval.py'] | 50 answerable Hit@3 49/50 MRR .9307 | BM25+Chroma Dense+RRF+Reranker在50条可回答Golden上Hit@3为98%，MRR0.9307。 |
| Retrieval ablation | ['data/evaluation/results/retrieval_ablation_v1.json'] | RRF .88到Reranker .98 | 对BM25/Dense/RRF/Reranker做固定集消融，重排后Hit@3为98%。 |
| Router eval | ['data/evaluation/scene_router_v2_guard_results.json'] | 65/70分类；contract另69/70 | Router在70题分类准确率92.86%，Macro F1 93.88%。 |
| Troubleshooting intake | ['data/evaluation/troubleshooting_intake_v1_2_results.json'] | 原集30/30+hard20/20+multi5 | 故障信息Workflow在30条原集和20条hard无false-ready/false-clarify，5组多轮通过。 |
| Bounded ReAct | ['app/agent/react_graph.py', 'app/agent/tool_calling.py'] | 代码+单测支持架构声明 | 实现有界LangGraph ReAct和只读说明书Tool，单请求固定Evidence Snapshot。 |
| Layered safety | ['app/services/grounded_qa.py', 'app/services/static_knowledge_boundary.py', 'app/services/semantic_grounding_gate.py', 'app/services/answer_release_policy.py'] | 层有独立职责 | 实现确定性Audit、Static Boundary、Semantic Gate及Release决策分层。 |
| Live delivery replay | ['data/evaluation/results/answer_release_policy_v1_1_3.json', 'app/services/answer_release_policy.py'] | 保存Gate replay 49/50，不是accuracy | 缓存Gate结果策略重放中，可回答自动交付49/50，无答案10题零错误自动放行。 |
| Review/publication | ['app/services/review_service.py', 'app/models.py', 'data/evaluation/results/frontend_v1_e2e.json'] | approved和published分离 | 实现Postgres草稿审核及显式发布分离，未发布公共GET 404。 |
| Session memory | ['app/agent/memory.py'] | 6消息/100 session进程内 | 实现按会话串行的短期滑动记忆与会话隔离。 |
| Long-term facts | ['app/memory/service.py', 'app/memory/filtering.py', 'app/memory/lifecycle.py'] | auto injection default false | 实现用户原文事实过滤、去重及TTL生命周期，自动注入默认关闭。 |
| SSE | ['data/evaluation/results/safe_sse_streaming_v1.json', 'app/main.py'] | 110ms first status / 9407ms done | 实现安全完成后SSE分块，真实Case A首状态110ms，done约9.4s。 |
| Tests | ['tests', 'frontend/tests'] | 470 backend +48 frontend, not model accuracy | 维护470项Python及48项前端测试，当前全套通过。 |
| Real E2E | ['data/evaluation/results/final_system_e2e_v1.json'] | historical 12 first-pass | 历史12个真实系统验收案例首次通过，含审核/发布和多场景。 |
| Deployment | ['data/evaluation/results/deployment_v1_1_e2e.json', 'Dockerfile', 'docker-compose.yml'] | ASCII context deployment evidence | 验收Postgres、幂等db-init、非root API和Nginx Compose结构。 |

## Claims To Avoid

| 避免 | 原因 |
| --- | --- |
| 准确率98% | Hit@3或自动交付率不是答案准确率。 |
| RAG准确率98% | 检索命中不证明生成正确。 |
| Router 98.57% accuracy | 98.57%为69/70契约，分类65/70。 |
| 100%无幻觉 | 小样本Judge/规则不能证明。 |
| Production-grade SaaS | 缺正式身份、租户、扩缩容及生产负载验收。 |
| 毫秒级回答 | 110ms首状态，Case A done 9407ms。 |
| 真实LLM Token Streaming | provider stream=false，服务端安全后分块。 |
| 默认长期记忆个性化 | 注入默认false。 |
| 全自动审核 | Gate是模型Judge，仍有人工Review/Publication。 |
| RRF显著提升 | RRF Hit3 .88低于BM25 .90。 |
| 当前chunk全局最优 | 有限实验且MRR/Hit不一致。 |
| 测试数量等于模型准确率 | 程序测试和模型benchmark是不同指标。 |
| auto release就是published | 即时交付不改变pending/unpublished。 |
| 任意Windows一键部署 | 中文路径BuildKit限制仍存在。 |

## 面试工程主题（非答案）

- read-only ReAct tool vs side-effect QA
- program-controlled retrieval query
- snapshot consistency and finalize once
- same-session lock/LRU
- untrusted user memory and default-off injection
- Audit/Boundary/Gate/Release layering
- instant delivery vs approval vs publication
- SSE buffer then release and abort
- Chroma fingerprint/dimension/index mapping
- conditional SQL updates
- atomic seen_count and semantic race limit
- Golden/Judge metric scope

## 已知限制

- Static Boundary是有限规则，不是通用中文parser。
- Semantic Gate/RAGAS依赖模型Judge，不是人工事实准确率且不保证零幻觉。
- 长期Memory注入默认false。
- Session Memory单进程，重启丢失，多实例不共享。
- client session_id及memory_owner_id不是身份认证。
- SSE Abort不能保证取消后台同步任务。
- buffer_then_release不降低模型完成和安全核验耗时。
- Windows中文路径BuildKit shared-key限制未解决，ASCII context已验。
- 知识源仅T8说明书，无价格/设备遥测/用户历史记录。
- 第12页部分图标纯文本缺失，不能猜按键。
- 60题Golden及50 answerable为小样本，不代表跨设备/开放用户泛化。
- Top5只是诊断，production evidence Top3。
- 不同hash新语义近重复事实并发不保证幂等。
- Memory cleanup显式调用，无后台调度。
- Review共享口令无正式登录/RBAC/多租户。
- QAService close异常仍可能覆盖commit结果；ReviewService已best effort。
- 历史构建日志报告3依赖漏洞(1中2高)，当前处理未核验。
- 本轮没有重建容器，healthy不证明容器源码等于当前仓库。

## Sources / Secret Scan

JSON source_files保存当前源码/结果SHA；每组指标带source_file。输出没有API密钥、Review Token、数据库口令/连接串、草稿/Evidence正文或CoT。secret值比对与凭据pattern扫描PASS，输入源码及历史JSON SHA未改变。

FINAL_PROJECT_TECHNICAL_INVENTORY_V1: PASS
