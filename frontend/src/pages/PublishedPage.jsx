import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getPublishedAnswer } from '../api/client.js';
export default function PublishedPage() {
  const { recordId } = useParams();
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { let alive = true; setResult(null); setError('');
    getPublishedAnswer(recordId).then(value => { if (alive) setResult(value); }).catch(e => {
      if (alive) setError(e.status === 404 ? '该回答尚未发布或不存在。' : e.message);
    }); return () => { alive = false; }; }, [recordId]);
  return <main className="published-page"><span className="eyebrow">PUBLISHED ANSWER</span><h1>正式客服答复</h1><p className="page-subtitle">仅展示经过人工审核并显式发布的答案。</p>
    {error ? <div className="empty" role="alert">{error}</div> : !result ? <div className="empty" role="status">正在读取正式答复…</div> :
      <article className="answer-card"><span className="badge published">已审核 · 已发布</span><h2>{result.question}</h2><time>{new Date(result.published_at).toLocaleString('zh-CN')}</time><div className="prose public-answer">{result.answer}</div></article>}
  </main>;
}
