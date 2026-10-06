# 智扫通——石头 T8 智能客服系统

面向石头 T8 官方说明书的可审计 AI Agent 客服：以可追溯证据回答产品问题，并把即时交付、人工审核和显式发布分开处理。

基于 **React + FastAPI + LangGraph**，知识源为 **26 页中文官方说明书**，构建 **BM25 + Chroma Dense → RRF → Qwen Reranker** 检索链；使用 **DeepSeek-V4-Flash** 生成回答，并组合场景 Router、故障 Intake、短期会话记忆、Audit、Semantic Gate、Review / Publication 与 Safe SSE。

这是一个本地单机工程项目，聚焦受控 Agent 与回答交付安全；不是通用智能体平台或生产 SaaS。

| 指标 | 结果 | 口径 |
| --- | ---: | --- |
| Router Accuracy | 92.86% | 70 条分类评测，65/70 |
| Retrieval Hit@3 | 98% | 50 条可回答 Golden Query，49/50 |
| Retrieval MRR | 0.9307 | 50 条可回答 Golden Query |
| Answerable Faithfulness | 0.9009 | RAGAS Judge，有效 49 条，1 次超时 |
| Answerable auto-delivery rate | 98% | 缓存 Gate 结果的本地策略重放，49/50；不是答案准确率 |
| Unanswerable safe abstention | 10/10 | 10 条不可回答样本的策略重放 |
| Python tests | 470/470 | 单元与集成测试 |
| Frontend tests | 48/48 | 在 ASCII 临时副本运行 |

> 所有指标都是各自评测口径下的结果。检索命中、Judge 分数、交付决策和软件测试不能互相替代，也不代表开放用户或跨设备泛化能力。

## 核心亮点

1. **Hybrid Retrieval**：BM25 与 Chroma Dense 候选经 RRF 融合，再由 Reranker 排序；保留阶段消融结果，避免把融合本身说成性能提升来源。
2. **受控 Agent 编排**：LangGraph 场景路由、Troubleshooting Intake 和有界 ReAct；Agent 只开放一个只读说明书检索 Tool，最多两次调用且同一请求复用 Evidence Snapshot。
3. **分层回答安全**：Static Knowledge Boundary、程序可判定的 Deterministic Audit、条件式 Semantic Grounding Gate、Answer Release Policy 各自负责不同问题。
4. **审核与发布分离**：生成记录进入审核流程；批准后仍需显式发布，公开读取只返回已发布记录。
5. **三层记忆**：Graph 工作状态、进程内 Session Sliding Memory、持久化长期事实分层设计；长期事实自动注入默认关闭。
6. **Safe SSE**：共用 JSON API 编排，在完整生成和安全检查后才分块释放，避免待审核草稿提前流出。
7. **可核对评测与部署**：固定 Golden、Router / Troubleshooting / RAGAS 评测、历史真实 E2E、Docker Compose 与 readiness 检查相互区分。

## 系统架构

```mermaid
flowchart TD
    U[Browser / React] -->|JSON 或 SSE| API[FastAPI API 编排]
    API --> SM[Session Memory<br/>最近 6 条消息 / 同会话串行]
    SM --> ROUTER[Scene Router<br/>Qwen2.5-7B-Instruct]
    ROUTER --> POLICY[Scene Policy]
    POLICY --> INTAKE[Troubleshooting Intake<br/>仅故障场景]
    INTAKE -->|needs clarification| CLARIFY[澄清响应]
    POLICY -->|clarify / handoff| FLOW[流程响应]
    INTAKE -->|ready| AGENT[LangGraph 有界 ReAct Agent]
    POLICY -->|知识场景| AGENT
    AGENT --> TOOL[只读 search_t8_manual]
    TOOL --> NORM[程序化 Query Normalization]
    NORM --> RET[BM25 Top10 + Chroma Dense Top10]
    RET --> RRF[RRF K=60 → Top10]
    RRF --> RERANK[Qwen Reranker → Top3]
    RERANK --> SNAP[固定 Tool Evidence Snapshot]
    SNAP --> GEN[DeepSeek-V4-Flash Draft]
    GEN --> AUDIT[Deterministic Audit]
    AUDIT --> REVIEWDB[(PostgreSQL<br/>Review Record：每次最多一条)]
    AUDIT --> API
    API --> BOUNDARY[Static Knowledge Boundary]
    BOUNDARY --> PRE[初步 Release Policy]
    PRE -->|需要语义核验| GATE[Semantic Grounding Gate<br/>Qwen Judge，失败关闭]
    PRE -->|无需 Gate| RELEASE[最终 Release Policy]
    GATE --> RELEASE
    RELEASE -->|auto_released / safe_abstention / review_required| API
    API -->|完整安全处理后| U
    API -.用户可见问答交换.-> SM
    API -.有 owner 时：抽取 / 过滤 / 持久化.-> LTM[(PostgreSQL<br/>Long-term Facts)]
    LTM -.默认不注入 Agent.-> AGENT
    REVIEWDB --> ADMIN[内部 Review UI<br/>X-Review-Token]
    ADMIN -->|approved 后显式 publish| PUB[(Published Answer)]
    PUB -->|GET /api/published/{id}| U
    CLARIFY --> API
    FLOW --> API
```

**顺序说明**：知识问答在 Agent Graph 中先检索、生成 Draft、执行 Deterministic Audit，并保存一次内部 Review Record；回到 API 编排后，才检查 Static Boundary、进行初步交付决策，并在需要时调用 Semantic Gate。Clarify、handoff 和 Intake 补充信息路径不调用知识生成，也不创建问答审核记录。

### 一次知识问答的处理顺序

1. FastAPI 校验输入；有 `session_id` 时读取该会话近期的用户可见消息，并串行化同会话请求。
2. Router 将请求分到五类业务 Scene，或引导 `clarify` / `human_handoff` 流程；固定 Scene Policy 控制可用能力。
3. 故障场景先由 Troubleshooting Intake 判断信息是否足够；不足时追问，足够时才继续 Agent / Manual 路径。
4. 有界 ReAct Agent 调用只读说明书 Tool。程序控制真实检索 query，模型 Tool 参数不控制 query；两次 Tool 请求最多执行一次 Retrieval，并复用相同 Evidence Snapshot。
5. DeepSeek 根据本轮 Top3 Evidence 生成 Draft；Audit 检查引用编号和可确定的风险，再由服务一次性保存审核记录。
6. API 对原始问题做 Static Boundary 与 Release Policy 判断；仅在需要时调用 Semantic Grounding Gate。失败关闭或要求审核的结果不会作为已核验正式答案自动释放。
7. JSON / SSE 返回公开允许的结果；Session Memory 只追加用户看到的问题与响应。长期事实抽取仅在具备 owner 时执行，且自动注入 Agent 默认关闭。

## RAG 检索链

生产检索链固定为：

```text
Conservative Query Normalization
→ BM25 Top10 + Chroma Dense Top10
→ Reciprocal Rank Fusion (K=60)，融合候选 Top10
→ Qwen Reranker
→ Top3 Evidence
```

- 知识源：石头 T8 中文官方说明书，26 页；当前语料为 41 条可追溯记录。
- Dense store：ChromaDB，cosine 距离；Embedding 维度 1024。Readiness 校验已有索引指纹和维度，不会自动重建。
- Embedding：`Qwen/Qwen3-Embedding-0.6B`；Reranker：`Qwen/Qwen3-Reranker-0.6B`。
- Reranker 分数用于排序，不代表答案正确概率。Evidence 编号对应本次固定 Snapshot，进入生成与审核快照。

| 检索阶段 | Hit@3 | MRR | 说明 |
| --- | ---: | ---: | --- |
| BM25 | 90% | 0.8312 | 稀疏检索基线 |
| Dense | 84% | 0.7607 | Chroma 向量检索基线 |
| RRF | 88% | 0.8131 | 融合候选后、Reranker 前 |
| RRF + Reranker | 98% | 0.9307 | 最终生产检索顺序；Hit@1 为 44/50，Hit@5 为 50/50 |

评测集包含 50 条可回答 Golden Query。RRF 的 Hit@3 为 88%，低于 BM25 的 90%；RRF 用于组合稀疏与稠密候选，最终 Hit@3 提升主要体现在后续重排，不能据此说 RRF 单独显著提升检索。生产仅向生成模型提供 Top3。

## Agent 与场景路由

Router 使用 `Qwen/Qwen2.5-7B-Instruct`；主回答模型为 `deepseek-ai/DeepSeek-V4-Flash`。备用配置 `Qwen/Qwen3.8-27B` 仅作记录，不参与自动 fallback。

Router 评测共 70 条：分类准确率 65/70（92.86%），Macro F1 为 93.88%；严格输出契约成功 69/70（98.57%）。**契约成功率不等于分类准确率。** 另一个 hard set 为 19/20，包含一次 timeout。

五类业务 Scene：

| Scene | 用途 |
| --- | --- |
| `consult_qa` | 说明书产品咨询 |
| `troubleshooting` | 故障排查 |
| `maintenance` | 维护保养 |
| `maintenance_report` | 清扫 / 维护记录类交互 |
| `configuration_recommendation` | 配置建议 |

`clarify` 与 `human_handoff` 是流程意图，不是额外业务 Scene。ReAct 采用受控 Tool Surface：只读查询 T8 官方手册，不联网、不控制机器人、不读取实时设备状态，也不开放任意工具自治。

### 故障 Intake

Troubleshooting Intake 在故障场景中判断信息是否已足以进入处理：`ready` 继续到 Agent / Manual，`needs_clarification` 先询问必要信息。已验证错误码快速路径和多轮上下文，但“信息足以继续”不等于说明书已经支持错误码解释。

- 原始评测：30/30；hard set：20/20。
- 两组 false-ready / false-clarify 均为 0；多轮用例 5/5。
- 以上是冻结样本集结果，不代表开放故障分布上的效果。

## 安全回答与交付

系统把“能否回答”“回答是否有依据”和“是否向用户交付”拆成独立检查：

| 层 | 主要职责 | 不代表什么 |
| --- | --- | --- |
| Static Knowledge Boundary | 判断静态说明书是否具备回答某类问题的能力，识别个人实时设备状态等边界 | 不是通用中文语义解析器 |
| Deterministic Audit | 检查引用编号、金额、个人运行数据、缺失引用等可由程序判定的风险 | 引用编号合法不等于 Evidence 语义支持答案 |
| Semantic Grounding Gate | 核对 Evidence 是否支持 Draft 核心回答和具体事实；失败关闭 | 是模型 Judge，不是人工事实裁判 |
| Answer Release Policy | 根据前序结果决定 `auto_released`、`safe_abstention` 或 `review_required` | `auto_released` 不等于审核批准或发布 |

程序拦截优先于 Semantic Gate。Semantic Gate 只在策略需要时调用；对未回答、缺少依据或 Gate 失败的内容，系统按安全策略拒答或转人工审核。回答策略与生成式 Audit 并非一个“AI 审核模型”包办。

### 评测结果与口径

**RAGAS / Generation**：冻结集 60/60 生成成功。RAGAS 使用 `deepseek-ai/DeepSeek-V4-Flash` 作为 Judge：

| 范围 | Faithfulness | Answer Relevance |
| --- | ---: | ---: |
| 全部 60 题 | 0.8667（59 条有效，1 次超时） | 0.7330（60 条有效） |
| 50 条可回答题 | 0.9009（49 条有效，1 次超时） | 0.8105（50 条有效） |

10 条不可回答题的拒答 Judge 结果为 10/10。引用编号合法性为 60/60；可回答题引用出现率为 50/50；严格引用要求为 56/60（93.33%）。这些是模型 / Judge 评测，不是人工事实准确率；一次 Faithfulness 超时从该项均值样本中排除。

**Answer Release 本地策略重放**：复用已保存的 Semantic Gate 输出，没有重新进行全量外部模型调用。

| 冻结集 | `auto_released` | `review_required` | `safe_abstention` |
| --- | ---: | ---: | ---: |
| 50 条可回答 | 49 | 1 | 0 |
| 10 条不可回答 | 0 | 0 | 10 |

可回答题自动交付率为 49/50（98%），它描述交付策略分布，**不是答案准确率**。不可回答题 incorrect auto release 为 0，`false_boundary` 为 0。V1.1.5 的这项最终验证是本地策略重放，不是 60 题新一轮 Gate 调用。

## Review 与 Publication

内部审核和用户交付是两条有关联但不同的状态线：

```text
Draft generated → pending_review → approved / rejected → explicit publish
```

- `auto_released`：当前响应经过交付策略后可以即时返回；不等于 `approved`，也不等于 `published`。
- `approved`：内部人员完成审核；仍需显式执行发布操作。
- `published`：显式发布后的可公开记录；只有此时 `GET /api/published/{record_id}` 才可读取，未发布记录返回 404。
- 审核列表、决策和发布接口由 `X-Review-Token` 保护。项目采用共享口令方案，没有正式用户身份和 RBAC。

## Memory

| 层 | 内容与边界 |
| --- | --- |
| Request / Graph Working State | 单次运行的消息、Tool Snapshot 和 Draft；内部状态不作为会话转录暴露 |
| Session Sliding Memory | 最近最多 6 条用户可见消息、最多 100 个会话；同一会话串行，不同会话可并行 |
| Persistent Long-term Facts | `device_profile`、`home_environment`、`usage_preference`；原文片段过滤、精确 / 语义去重、`seen_count`、重要度、TTL 与衰减、显式清理 |

Session Memory 保存在进程内，服务重启后丢失，多实例之间不共享。客户端提供的 session id 和 owner id 不是身份认证。长期事实抽取与存储可在提供 owner 时使用，但 `ENABLE_LONG_TERM_MEMORY_INJECTION=false`：默认不把长期事实自动注入 Agent，也不宣称系统会默认自动个性化。长期记忆作为用户来源数据，不是 System Instruction 或手册 Evidence。

## Safe SSE

流式接口为 `POST /api/agent/chat/stream`，事件包括 `status`、`answer_start`、`answer_delta`、`review_required`、`done`、`error`；心跳使用 SSE comment。

当前是 **`provider_streaming=false`、`client_sse_streaming=true`、`buffer_then_release`**：服务端先完成模型生成、审核和交付检查，再把允许公开的最终答案分块发送。核验完成前不发送 `answer_delta`；待审核结果没有答案分块。Nginx 对该 SSE 路径关闭缓冲。

冻结 Case A 的首个状态事件约 110ms 到达，`done` 约 9407ms。110ms 是首个状态事件时间，不是完整回答延迟；缓冲释放也不会缩短模型完整生成和安全核验时间。浏览器 `AbortController` 中止读取，不保证终止后端已开始的同步处理。

## 技术栈

| 层 | 技术与已核实版本 |
| --- | --- |
| Runtime | Python 3.12.3 |
| Backend | FastAPI 0.142.2、Pydantic 2.13.5、LangGraph 1.2.12、SQLAlchemy 2.1.3、psycopg 3.3.6、Uvicorn 0.54.0 |
| Retrieval | ChromaDB 1.5.9、jieba 0.42.1、rank-bm25 0.2.2 |
| Tool adapter | `langchain-core` 1.6.6（用于结构化只读 Tool）；Agent 图由 LangGraph 编排 |
| Frontend | React / React DOM 19.3.0、React Router DOM 7.18.4、Vite 7.3.7、Vitest 3.2.7；JavaScript / JSX |
| Models | Router / Semantic Gate：`Qwen/Qwen2.5-7B-Instruct`；生成：`deepseek-ai/DeepSeek-V4-Flash`；Embedding：`Qwen/Qwen3-Embedding-0.6B`；Reranker：`Qwen/Qwen3-Reranker-0.6B` |
| Storage | PostgreSQL（Compose `postgres:16` tag）、ChromaDB 本地持久化索引 |
| Deployment | Docker、Docker Compose、Nginx（镜像 `nginx:alpine` tag） |

PostgreSQL 与 Nginx 的具体 patch 版本未确认，不在此推断。后备模型 `Qwen/Qwen3.8-27B` 只记录，不自动切换。

## 项目结构

```text
01-智扫通-扫地机器人智能客服系统/
├── app/
│   ├── agent/       # LangGraph、Router、ReAct、Session Memory、Tools
│   ├── memory/      # 长期事实抽取、过滤、存储与生命周期
│   ├── retrieval/   # BM25、Chroma Dense、RRF、Reranker
│   ├── services/    # Grounded QA、Audit、Boundary、Gate、Release、Review
│   ├── main.py      # FastAPI、JSON/SSE 编排、API 安全边界
│   ├── models.py    # SQLAlchemy ORM
│   └── schemas.py   # API / 服务数据结构
├── frontend/
│   ├── src/         # React 页面、API client、SSE client
│   ├── tests/       # Vitest 测试
│   ├── Dockerfile
│   └── nginx.conf
├── data/
│   ├── raw/         # 原始官方说明书
│   ├── processed/   # 处理后的知识记录
│   ├── vector_store/# Chroma 持久化数据
│   └── evaluation/  # Golden 与冻结评测结果
├── tests/           # Python 单元 / 集成测试
├── scripts/         # 初始化、数据准备与离线评测脚本
├── docs/            # 项目最终技术盘点
├── Dockerfile
├── docker-compose.yml
├── requirements-runtime.txt
├── requirements.txt
└── .env.example
```

## 本地启动

已验证入口为 Docker Compose。需要 Docker Desktop / Docker Compose，以及本地 `.env` 中的模型与数据库配置。

首次运行时在仓库根目录执行：

```powershell
Copy-Item .env.example .env
```

然后用本地编辑器填写 `.env` 中的私有配置；Compose 内部数据库地址应使用服务名 `postgres` 和容器端口 `5432`。不要把 `.env`、真实口令或 API Key 提交到仓库。

```powershell
docker compose up --build
```

服务启动后访问：

- 前端：<http://127.0.0.1:8080>
- API：<http://127.0.0.1:8000>
- API liveness：<http://127.0.0.1:8000/health>
- API readiness：<http://127.0.0.1:8000/ready>

Compose 包含 PostgreSQL、`db-init`、API 与 Frontend/Nginx；初始化脚本按顺序幂等执行并校验目标数据库名。默认端口分别是 PostgreSQL `5433`、API `8000`、Frontend `8080`，均绑定 `127.0.0.1`。PostgreSQL 使用命名卷，Chroma 使用仓库内持久化目录。

停止服务：

```powershell
docker compose down
```

`/health` 只检查 API 进程是否存活。`/ready` 检查数据库连接与 schema、41 条知识语料、已有 Chroma 索引指纹及 1024 维向量；不会调用外部模型，也不会重建索引。

> **Windows 路径说明**：当前 Docker Desktop / BuildKit 在本项目的中文路径下曾出现 build context / shared-key 错误。相同 Dockerfile 已在 ASCII 临时构建路径完成验收；不要据此认为中文路径下一键构建已解决。

## 使用示例

可以在聊天页输入以下真实场景问题；下表说明的是处理路径，不是预先编造的答案：

| 用户问题 | 处理路径示例 |
| --- | --- |
| 主刷怎么清理？ | 进入维护类 Scene，由说明书检索提供证据 |
| 错误码 401 是什么意思？ | 进入故障 Intake；快速路径只判断信息是否足以继续，错误码含义仍需手册证据 |
| 我家机器人现在电量是多少？ | 静态知识边界标记为设备实时信息不可由说明书确认，安全拒答 |
| 我要人工客服 | 进入 `human_handoff` 流程，不调用 RAG |
| 现在 T8 官方卖多少钱？ | 当前知识源不含实时售价，不能声称查询到当前价格 |

如请求进入人工审核，用户侧响应不会暴露内部 Draft、完整 Evidence 或 Tool 消息。

## 重要 API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/agent/chat` | Agent JSON 问答 |
| `POST` | `/api/agent/chat/stream` | 安全检查后的 SSE 响应 |
| `GET` | `/health` | 进程存活检查 |
| `GET` | `/ready` | 依赖与索引 readiness 检查 |
| `GET` | `/internal/reviews` | 内部审核列表；请求头 `X-Review-Token` |
| `GET` | `/internal/reviews/{record_id}` | 内部审核详情；请求头 `X-Review-Token` |
| `POST` | `/internal/reviews/{record_id}/decision` | 审核批准 / 驳回；请求头 `X-Review-Token` |
| `POST` | `/internal/reviews/{record_id}/publish` | 审核通过后的显式发布；请求头 `X-Review-Token` |
| `GET` | `/api/published/{record_id}` | 公开读取已发布答案；未发布时 404 |

Review Token 只通过内部请求头传递，不放在 URL 中；当前共享口令不等于正式登录或权限系统。

## 测试与验收

当前冻结盘点中的测试结果：

```text
Python:   470/470 passed
Frontend: 48/48 passed
```

Python 测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Frontend 48 项是在 ASCII 临时副本中执行的，因为原中文路径下 Node 访问存在限制；副本已清理。测试数说明程序测试覆盖状态，不代表模型回答准确率。

历史 Final System E2E 为 12/12 first-pass success、0 retries、0 external failures、0 business failures；覆盖多场景、多轮、审核与发布等流程。**README 编写阶段没有重新运行该 E2E。**

## 已知限制

1. 当前知识源仅包含石头 T8 官方说明书；不包含实时售价、设备遥测和用户清扫历史。说明书第 12 页部分图标在纯文本中缺失，系统不能凭空命名按键。
2. Static Knowledge Boundary 是有限、保守的规则集合，不是通用中文语义解析器。
3. Semantic Grounding Gate 和 RAGAS 依赖 LLM Judge；Judge 分数不等同人工事实审核，也不能证明零幻觉。
4. 检索 / Router / RAGAS Golden 规模有限；50 条可回答检索题和 60 条问答评测不能代表跨设备或开放用户泛化。
5. 长期事实自动注入默认关闭；session memory 是进程内状态，重启丢失且多实例不共享。当前 session id / memory owner id 不是身份认证。
6. SSE 采用 `buffer_then_release`，安全之前不发答案分块，但不会缩短完整模型生成时间；浏览器中止请求不保证取消后台任务。
7. Review 使用共享口令，没有正式账号、RBAC 或多租户权限管理；批准和发布必须分开操作。
8. Docker Desktop / BuildKit 在当前 Windows 中文项目路径下存在构建限制；ASCII 临时路径已验证相同 Dockerfile。
9. 最终 Release 指标复用了已保存 Gate 输出进行本地策略重放；它不是新一轮全量 Gate 模型调用。历史构建曾报告依赖漏洞，当前处理状态未核实。

## 项目状态

本 README 基于 `docs/project_final_inventory_v1.md` 与 `data/evaluation/results/project_final_inventory_v1.json` 编写。数字、模型、版本、链路和限制以该冻结盘点为准。项目定位为面向 T8 官方说明书的可审计 Agent / RAG 工程实践，不宣称已具备对外生产 SaaS 的身份、租户、扩缩容和负载验收能力。