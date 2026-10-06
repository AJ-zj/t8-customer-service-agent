export const scenes = {
  consult_qa: '产品咨询', troubleshooting: '故障排查', maintenance: '维护建议',
  maintenance_report: '维护报告', configuration_recommendation: '配置推荐',
  clarify: '信息澄清', human_handoff: '人工客服'
};
export const statuses = { pending_review: '待审核', approved: '已通过', rejected: '已拒绝',
  unpublished: '未发布', published: '已发布' };
export const deliveries = { auto_released: '即时答复', review_required: '人工审核',
  safe_abstention: '资料不足', clarification: '需要补充信息', human_handoff: '转人工' };
export function SceneBadge({ intent }) { return intent ? <span className="badge scene">{scenes[intent] || '处理结果'}</span> : null; }
export function DeliveryBadge({ status }) { return status ? <span className={'badge delivery ' + status}>{deliveries[status] || '处理结果'}</span> : null; }
export function StatusBadge({ status }) { return <span className={'badge ' + status}>{statuses[status] || status}</span>; }
export function EmptyState({ children }) { return <div className="empty">{children}</div>; }
