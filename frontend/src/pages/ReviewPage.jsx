import { useState } from 'react';
import { Link } from 'react-router-dom';
import { listReviews, getReviewDetail, submitReviewDecision, publishReview } from '../api/client.js';
import { StatusBadge, EmptyState } from '../components/Badges.jsx';
const date = value => value ? new Date(value).toLocaleString('zh-CN') : '—';
export default function ReviewPage() {
  const [token, setToken] = useState('');
  const [authenticated, setAuthenticated] = useState(false);
  const [rows, setRows] = useState([]);
  const [detail, setDetail] = useState(null);
  const [filter, setFilter] = useState('pending_review');
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function work(action) {
    setBusy(true); setError('');
    try { await action(); } catch (e) { setError(e.message); if (e.status === 401 || e.status === 403) { setAuthenticated(false); setRows([]); setDetail(null); } }
    finally { setBusy(false); }
  }
  async function refresh(id = detail?.id) {
    setRows(await listReviews(token));
    if (id) setDetail(await getReviewDetail(id, token));
  }
  async function decide(decision) {
    await submitReviewDecision(detail.id, decision, note.trim(), token);
    await refresh(detail.id);
    setNote('');
  }
  const visible = rows.filter(row => filter === 'all' || (filter === 'published' ?
    row.publication_status === 'published' : row.review_status === filter));
  return <main className="review-page">
    <div className="page-heading"><div><span className="eyebrow">HUMAN REVIEW</span><h1>内部审核工作台</h1><p>核对说明书依据，审核与发布分开执行。</p></div>{authenticated && <button className="secondary" onClick={() => { setToken(''); setAuthenticated(false); setRows([]); setDetail(null); setNote(''); }}>退出审核</button>}</div>
    <div className="security-notice">♧ 管理员令牌仅保存在当前页面内存中。刷新或离开此页面后需重新输入。</div>
    {!authenticated && <form className="auth-card" onSubmit={e => { e.preventDefault(); work(async () => { await refresh(null); setAuthenticated(true); }); }}><span className="auth-icon">▤</span><h2>进入内部审核</h2><p>请使用管理员提供的 Review Token。</p><label>Review Token<input type="password" autoComplete="off" aria-label="Review Token" value={token} onChange={e => setToken(e.target.value)} /></label><button className="primary" disabled={busy || !token.trim()}>验证并进入</button></form>}
    {error && <p className="error-notice" role="alert">{error}</p>}
    {authenticated && <div className="review-columns"><section className="records-panel"><div className="panel-heading"><h2>审核记录</h2><button className="text-button" disabled={busy} onClick={() => work(() => refresh())}>刷新</button></div><div className="tabs">{[['pending_review','待审核'],['approved','已通过'],['rejected','已拒绝'],['published','已发布'],['all','全部']].map(([value,label]) => <button key={value} className={filter === value ? 'selected' : ''} onClick={() => setFilter(value)}>{label}</button>)}</div><p className="list-caption">最近 50 条记录 · {visible.length} 条符合筛选</p>{!visible.length && <EmptyState>暂无符合筛选的记录。</EmptyState>}{visible.map(row => <button className={'record-card ' + (detail?.id === row.id ? 'active' : '')} key={row.id} onClick={() => work(async () => { setDetail(await getReviewDetail(row.id, token)); setNote(''); })} disabled={busy}><span className="record-meta">#{row.id} <time>{date(row.created_at)}</time></span><strong>{row.question}</strong><div><StatusBadge status={row.review_status} /><StatusBadge status={row.publication_status} /></div><span className="audit-label">自动审查：{row.audit_status}</span></button>)}</section>
      <section className="detail-panel">{!detail ? <EmptyState>← 选择一条记录，核对草稿与证据</EmptyState> : <><div className="panel-heading"><h2>记录 #{detail.id}</h2><StatusBadge status={detail.publication_status} /></div><h3 className="detail-question">{detail.question}</h3><div className="detail-meta"><StatusBadge status={detail.review_status} /><span>创建于 {date(detail.created_at)}</span></div><div className="audit-info">自动审查：{detail.audit_status}<small>待质量审核或待正式发布审核；部分答复可能已即时提供给用户。即时答复不代表已经正式发布。</small><small>编号合法不等于事实受到证据支持，请逐项核对。</small></div><section className="draft-section"><h3>内部答案草稿</h3><div className="prose">{detail.draft_answer || '本条记录没有草稿。'}</div></section><section><h3>本轮说明书证据</h3>{Array.isArray(detail.evidence) ? detail.evidence.map((e,i) => <details className="evidence-card" key={i}><summary>{e.evidence_id} · PDF 第 {e.pdf_page} 页 · {e.title || e.section || '说明书'}</summary><p className="prose">{e.text}</p></details>) : <EmptyState>没有可展示的证据快照。</EmptyState>}</section>
      {detail.review_status === 'pending_review' && <div className="review-actions"><label>审核备注<textarea aria-label="审核备注" placeholder="说明审核依据或驳回原因…" value={note} maxLength={1000} onChange={e => setNote(e.target.value)} /></label><div><button className="secondary danger" disabled={busy || !note.trim()} onClick={() => work(() => decide('rejected'))}>驳回</button><button className="primary" disabled={busy || !note.trim()} onClick={() => work(() => decide('approved'))}>通过审核</button></div></div>}
      {detail.review_note && <p className="review-note">审核意见：{detail.review_note}</p>}
      {detail.review_status === 'approved' && detail.publication_status === 'unpublished' && <div className="publish-callout"><div><h3>已通过审核，尚未正式发布</h3><p>审核通过不会自动公开答案，请单独执行发布。</p></div><button className="primary" disabled={busy} onClick={() => work(async () => { await publishReview(detail.id, token); await refresh(detail.id); })}>正式发布</button></div>}
      {detail.publication_status === 'published' && <div className="published-callout"><p>发布于 {date(detail.published_at)}</p><Link className="primary link-button" to={'/published/' + detail.id}>查看正式答案 ↗</Link></div>}
      </>}</section></div>}
  </main>;
}
