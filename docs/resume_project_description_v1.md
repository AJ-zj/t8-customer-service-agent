# 推荐主版本

**项目名称：智扫通——石头 T8 智能客服系统**  
**副标题：AI Agent / RAG 智能客服系统（个人项目）**

基于石头 T8 官方说明书知识库，构建 Agent/RAG 多场景客服，通过可追溯证据生成回答，结合安全交付、人工审核与显式发布控制回答出口。

**技术栈：** Python / FastAPI / LangGraph / LangChain Core（Tool Adapter）/ React / PostgreSQL / ChromaDB / BM25 / RRF / Docker / Nginx / DeepSeek / Qwen

- **A1｜Agent 编排：** 基于 LangGraph 编排五类客服场景，串联 Scene Router、故障 Intake 与受控 ReAct，仅开放只读手册 Tool；70 条评测中 Router Accuracy 为 92.86%。

  > 可追问点：为什么限制 Tool 范围？Intake 的 ready 与“有依据回答”有什么区别？（内部准备，不复制进简历）

- **A2｜Hybrid RAG：** 构建 BM25 与 Chroma Dense 双路召回，经 RRF、Qwen Reranker 生成 Top3 证据；在 50 条可回答 Golden Query 上 Hit@3 达 98%，MRR 为 0.9307。

  > 可追问点：为什么融合后仍需重排？Hit@3 与回答质量分别衡量什么？（内部准备，不复制进简历）

- **A3｜安全交付：** 组合静态知识边界、程序 Audit、条件式 Grounding Gate 与交付策略，区分自动交付、拒答和待审；冻结策略回放中可回答样本自动交付率 98%，10/10 不可回答样本安全拒答。

  > 可追问点：哪些检查可由程序确定？Gate 失败时如何处理？（内部准备，不复制进简历）

- **A4｜工程闭环：** 实现人工审核与显式发布分离，贯通 React、PostgreSQL 和 Docker Compose/Nginx；Safe SSE 采用 buffer-then-release，安全检查完成前不输出答案增量。

  > 可追问点：Auto Release、Approve、Publish 为什么分开？SSE 为何不透传未核验 Draft？（内部准备，不复制进简历）

# RAG 偏重版本

**项目名称：智扫通——石头 T8 智能客服系统**

基于官方说明书构建可追溯的检索增强问答，以稀疏与向量检索、重排和证据核验支撑客服回答，并保留分阶段评测与人工审核路径。

**技术栈：** Python / FastAPI / ChromaDB / BM25 / RRF / Qwen Reranker / LangGraph / PostgreSQL / React / Docker / DeepSeek

- **B1｜知识检索：** 将官方手册整理为可追溯记录，构建 BM25 与 ChromaDB Vector Search 双路召回，经 RRF 融合、Qwen Reranker 重排输出 Top3；50 条可回答题 Hit@3 为 98%，MRR 为 0.9307。

  > 可追问点：原记录与实验切块如何区分？为什么不将 RRF 单独描述为效果提升来源？

- **B2｜检索约束：** 将只读检索封装为结构化 Tool，真实 Query 由程序上下文与保守规则构造；单请求复用固定 Evidence Snapshot，避免重复调用导致引用编号对应不同记录。

  > 可追问点：模型改写 Tool 参数时，如何保持检索输入与证据快照可控？

- **B3｜质量评测：** 分开评估检索、生成与证据支持，使用 RAGAS LLM Judge 得到可回答集 Faithfulness 0.9009、Answer Relevance 0.8105，并保留超时与有效样本口径。

  > 可追问点：Judge 分数与人工事实准确率为何不同？超时样本如何进入统计？

- **B4｜Grounding 交付：** 用程序 Audit、Static Boundary 和条件式 Semantic Gate 控制回答出口，证据不足或核验失败时拒答或待审；冻结策略回放中 10/10 不可回答样本安全拒答，错误自动放行 0。

  > 可追问点：引用编号合法为什么仍需要 Grounding？哪些请求不必调用 Gate？

# Agent 偏重版本

**项目名称：智扫通——石头 T8 智能客服系统**

围绕说明书客服构建受控 Agent，以场景路由、故障信息采集、只读工具和分层记忆组织上下文，通过独立安全策略决定回答交付方式。

**技术栈：** Python / FastAPI / LangGraph / LangChain Core（Tool Adapter）/ DeepSeek / Qwen / PostgreSQL / ChromaDB / React / Docker / SSE

- **C1｜场景编排：** 用 LangGraph 连接 Scene Router、故障 Intake 与知识问答，将澄清和转人工作为旁路流程；70 条路由评测 Accuracy 为 92.86%，Macro F1 为 93.88%。

  > 可追问点：五类业务 Scene 与两类 flow intent 为什么不混为一类？

- **C2｜受控 ReAct：** 基于标准 Tool Calling 只开放手册查询工具，限制最多两次调用、一次真实检索；复用同一 Observation 快照并统一 finalize，避免模型重复检索和重复保存审核记录。

  > 可追问点：多 Tool Call 如何共享快照并保留各自协议 ID？

- **C3｜分层 Memory：** 区分 Working State、Session 滑动上下文与 PostgreSQL 长期事实，支持同会话串行和事实去重、TTL/衰减；长期事实自动注入默认关闭，避免把存储能力等同可靠个性化。

  > 可追问点：哪些消息不能写入 Session？长期事实为什么是低权限用户数据？

- **C4｜安全出口：** 串联 Audit、条件式 Grounding 与 Release Policy，并分离人工审核和显式发布；通过 Safe SSE 先送状态、核验后释放答案，待审 Draft 不进入用户答案流。

  > 可追问点：状态流与模型生成流有什么区别？审核批准后是否立即公开？

# 超短版本

**智扫通——石头 T8 智能客服系统：** 基于官方说明书的多场景 Agent/RAG 客服，以手册证据、审核与交付策略控制回答出口。

- **S1：** 基于 LangGraph 实现五场景路由、故障 Intake 与受控 ReAct；70 题 Router Accuracy 92.86%。
- **S2：** BM25 + Chroma 经 RRF、Reranker 提供证据；50 条可回答题 Hit@3 98%、MRR 0.9307。
- **S3：** 分离 Grounding 交付、人工审核与显式发布；Safe SSE 在安全检查后释放答案，待审 Draft 不透传。

# 候选替换 Bullet

以下三条用于替换主版本对应条目，不建议全部追加。仍保持最终简历约四条主 Bullet。

- **R1｜Memory：** 实现 Working/Session/Persistent Facts 三层 Memory，长期事实支持去重、原子 seen_count、importance、TTL 与衰减；自动注入默认关闭，Session 仅保留用户可见对话。

  > 可追问点：如何避免 seen_count 并发丢失更新？哪些语义去重竞争仍未解决？

- **R2｜评测：** 建立检索与生成分层评测，采用 RAGAS LLM Judge 核验可回答集，Faithfulness 为 0.9009、Answer Relevance 为 0.8105；保留有效样本与超时记录。

  > 可追问点：为什么不能把 Faithfulness 写成人工准确率？

- **R3｜测试与交付：** 验证问答、审核和发布接口及前端交互边界，冻结 Python 470/470、Frontend 48/48 测试通过；结合 Docker Compose 和 readiness 检查支持本地部署验收。

  > 可追问点：单元测试、离线评测与真实 E2E 分别验证什么？

# Bullet → Evidence Mapping

## 事实源与使用范围

- 主事实源：[冻结 Inventory Markdown](project_final_inventory_v1.md)、[冻结 Inventory JSON](../data/evaluation/results/project_final_inventory_v1.json)。
- 交叉核对：[README](../README.md)、[系统架构 SVG](architecture/system_architecture_v1.svg)。
- 以下条目描述本项目已实现能力，不额外宣称团队管理、公司业务收益或独立原创算法；个人投入范围按实际经历陈述。
- 数字均引用冻结结果，本次未调用模型、数据库或重新运行测试。评测只支持对应固定样本上的结论，不外推开放用户或跨设备效果。
- LangChain Core 仅指 StructuredTool 适配；Agent 图编排使用 LangGraph。

| Bullet | 代码 / 实现证据（仓库根目录相对路径） | Inventory JSON 字段 / 评测来源 | 口径与限制 |
| --- | --- | --- | --- |
| A1、C1、S1 | `app/agent/llm_router.py`、`scene_policy.py`、`troubleshooting.py`、`react_graph.py` | `modules`、`metrics.router.summary`；`data/evaluation/scene_router_v2_guard_results.json` | 70 题，65 正确；Accuracy 92.86%，Macro F1 93.88%；69 次契约成功不等于分类正确 |
| A2、B1、S2 | `app/retrieval/t8_retrieval.py`、`t8_vector_store.py` | `retrieval_config`、`metrics.retrieval.summary.Reranker`；`data/evaluation/results/retrieval_ablation_v1.json` | BM25/Dense 各 Top10，RRF K=60 取 Top10，重排后 Top3；50 可回答题 Hit@3=49/50，MRR=0.930666… |
| B2、C2 | `app/agent/tools.py`、`react_graph.py`、`app/services/qa_service.py` | `data_flow`、`modules`；`tests/test_react_agent.py`、`tests/test_search_t8_manual_tool.py` | 一个只读手册 Tool，最多两次 Tool Call、一次真实检索；复用快照；正常 finalize 一次 |
| A3、B4 | `app/services/grounded_qa.py`、`static_knowledge_boundary.py`、`semantic_grounding_gate.py`、`answer_release_policy.py`、`app/main.py` | `metrics.release`；Inventory Markdown「Answer Release回放」 | 使用已保存 Gate 输出作本地策略回放；50 可回答题为49 auto/1 review；10 不可回答题全部 safe，incorrect auto release=0 |
| A4、C4、S3 | `app/services/review_service.py`、`app/main.py`、`frontend/src/api/sse.js`、`frontend/nginx.conf`、`docker-compose.yml` | `safety`、`sse`、`deployment`；`tests/test_publication.py`、`tests/test_agent_sse_api.py` | Auto Release ≠ Approved ≠ Published；SSE 为安全后分块交付，provider streaming 关闭 |
| B3、R2 | `scripts/46_ragas_judge_eval.py` | `metrics.generation_ragas`；`data/evaluation/results/ragas_eval_v1.json` | 可回答50题；Faithfulness 有效49题、1次超时，均值0.9009；Relevance有效50题，展示值0.8105沿用冻结README |
| C3、R1 | `app/agent/state.py`、`memory.py`、`app/memory/service.py`、`lifecycle.py`、`app/config.py` | `memory`、`modules`；`tests/test_memory_facts.py`、`tests/test_memory_lifecycle.py`、`tests/test_long_term_memory_feature_flag.py` | Session 为进程内；长期事实默认不注入。原子计数不代表不同hash语义近似首次并发插入完全幂等 |
| R3 | `tests/`、`frontend/tests/`、`docker-compose.yml`、`app/services/readiness.py` | `testing.python`、`testing.frontend`、`deployment` | 470/470与48/48是冻结测试记录；Frontend在ASCII临时副本执行；不代表模型准确率或高并发验收 |

## 数字核对表（内部材料，不整段放简历）

| 数字 | 含义 | 使用位置 |
| --- | --- | --- |
| 五类场景 | 咨询、故障、维护、维护记录、配置建议；clarify/handoff另作流程意图 | A1、C1说明、S1 |
| 70题 / 92.86% / 93.88% | Router固定集分类Accuracy与Macro F1 | A1、C1、S1 |
| 50题 / 49命中 / Hit@3 98% / MRR 0.9307 | 可回答Golden的最终检索评测 | A2、B1、S2 |
| 49 auto / 1 review；10 safe / 0错误放行 | 冻结策略回放分布，非新一轮全量模型调用 | A3、B4 |
| 0.9009 / 0.8105 | RAGAS LLM Judge的Faithfulness / Answer Relevance | B3、R2 |
| 两次调用 / 一次检索 | 受控Tool调用上限与单请求Snapshot复用 | C2 |
| 470/470 / 48/48 | 冻结Python / Frontend测试通过数 | R3 |

## Resume-Safe Check

- 已核实技术栈在实现中使用；未把备用模型写成运行中的自动fallback。
- A/B/C各四条，超短版三条，候选替换三条；推荐简介控制为一句话。每条只选择少量核心指标。
- 生产安全顺序：Graph内生成Draft → Deterministic Audit → Review Record；回API后Static Boundary → Preliminary Release → 条件Semantic Gate → Final Release。列表提及安全组件不表示任意重排执行顺序。
- 三种用户交付结果分别对应 `auto_released`、`safe_abstention`、`review_required`；只有策略允许的答案即时交付，不声称所有场景自动释放。
- RAGAS展示值沿用冻结口径，不转为百分比事实准确率；一次Faithfulness超时不隐去。
- 已知历史差异已保留：V1.1.3真实Case B是review_required，不能被其他草稿的策略回放覆盖；SSE旧observer汇总有冲突，不将其解释为全链路无故障证明。
- 无新增事实冲突；不使用未核实的公司职责、用户规模、业务收益或部署负载结论。主版本不把长期Memory作为默认个性化能力。
- Secret Scan：仅技术名称、路径和聚合指标；无密钥、口令、数据库连接串、用户敏感事实或内部Draft。
- 本轮仅新增本文；未修改生产代码、测试、README、部署、前端、评测集、Golden或架构图。

# 可追问点

本节仅记录问题，不展开面试教学。各版本 Bullet 后的问题也不属于简历正文。

| 对应条目 | 可追问点 |
| --- | --- |
| Agent | 为什么使用固定场景能力和有界ReAct，而不开放任意工具？错误码fast path为什么只判断ready？ |
| Hybrid RAG | RRF单独结果并未超过BM25，如何解释仍保留融合与重排链路？MRR用哪个候选列表计算？ |
| Tool / Snapshot | 模型参数与真实Query如何解耦？为什么重复Tool调用必须复用同一Evidence编号映射？ |
| Safety | 引用合法与语义支持怎样分工？Static Boundary为何在Audit/Review之后？Gate失败怎样关闭交付？ |
| Review / SSE | 即时交付、审核批准、公开发布如何区分？安全之前可以发送哪些SSE事件？ |
| Memory | Working State、Session与长期事实各保存什么？为何默认关闭长期事实注入？ |
| Evaluation | 固定样本结果、LLM Judge、单元测试、历史真实E2E之间如何避免口径混用？ |

历史补充证据：`metrics.final_e2e`记录冻结系统E2E用例 **12/12首次通过**；本轮未复跑，不写成系统稳定性承诺。SSE冻结Case A首状态约110ms仅是状态事件，不是完整回答耗时；未作为主Bullet性能卖点。

# 禁止表述

下列文字是反例，**不得复制为简历能力声明**。

| 禁止或高风险写法 | 本文采用的准确口径 |
| --- | --- |
| 把检索命中率或自动交付率称为整个RAG/回答的准确率 | 50条可回答Query的Hit@3；缓存Gate结果的策略回放自动交付率 |
| 将98.57%称为Router分类Accuracy | 分类Accuracy为92.86%；98.57%仅为输出契约成功率 |
| 零幻觉、100%准确或100%安全 | 固定评测样本上的结果，保留Judge和规则边界 |
| 生产级、企业级高可用、高并发或支持百万用户 | 本地单机工程项目，无此类负载或业务规模证据 |
| 毫秒级回答、实时模型Token流 | Safe SSE状态先行，安全检查后再释放答案增量 |
| 自动长期个性化、长期事实默认注入 | 长期存储已实现，自动注入默认关闭 |
| 全自动审核、人工审核后才能回答 | 条件安全交付与内部人工审核并行；批准后仍需显式发布 |
| RRF显著提升检索 | 保留融合与重排分阶段评测，不将最终结果归因于RRF单独效果 |
| 带领团队、公司业务增长等未经支持的职责和收益 | 个人项目的具体实现与验证结果 |

RESUME_PROJECT_DESCRIPTION_V1: PASS
