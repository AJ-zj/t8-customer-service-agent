import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { streamAgentMessage } from '../api/client.js';
import { DeliveryBadge, SceneBadge } from '../components/Badges.jsx';

export const newSessionId = () => crypto.randomUUID();
const starters = ['主刷多久清理一次？', '机器人无法正常工作怎么办？', '帮我生成一份日常维护清单。', '我要人工客服'];
export default function ChatPage() {
  const [session, setSession] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [sessionSlot, setSessionSlot] = useState(null);
  useEffect(() => { setSessionSlot(document.getElementById('new-session-slot')); }, []);
  const lock = useRef(false);
  const controller = useRef(null);
  const flow = useRef(null);
  const [statusMessage, setStatusMessage] = useState('');
  useEffect(() => { if (flow.current) flow.current.scrollTop = flow.current.scrollHeight; }, [messages, busy]);
  useEffect(() => () => { controller.current?.abort(); }, []);
  async function send(event) {
    event?.preventDefault();
    const value = question.trim();
    if (!value || value.length > 500 || lock.current) return;
    lock.current = true; setBusy(true); setStatusMessage('正在处理…'); setQuestion('');
    setMessages(old => [...old, { role: 'user', text: value }]);
    const activeController = new AbortController();
    controller.current = activeController;
    const assistantId = crypto.randomUUID();
    try {
      await streamAgentMessage(value, session, {
        signal: activeController.signal,
        onEvent: ({ event, data }) => {
          if (event === 'status') { setStatusMessage(data.message || '正在处理…'); return; }
          if (event === 'answer_start') {
            setMessages(old => [...old, { id: assistantId, role: 'assistant', text: '',
              intent: data.intent, deliveryStatus: data.delivery_status, pending: false, streaming: true }]);
            return;
          }
          if (event === 'answer_delta') {
            if (typeof data.text !== 'string') return;
            setMessages(old => old.map(item => item.id === assistantId ? { ...item, text: item.text + data.text } : item));
            return;
          }
          if (event === 'review_required') {
            setMessages(old => [...old, { id: assistantId, role: 'assistant',
              text: data.message || '回答已转入质量审核。', intent: data.intent,
              deliveryStatus: 'review_required', pending: true }]);
            return;
          }
          if (event === 'done') {
            setMessages(old => old.map(item => item.id === assistantId ? {
              ...item, intent: data.intent || item.intent,
              deliveryStatus: data.delivery_status || item.deliveryStatus, streaming: false,
            } : item));
            return;
          }
          if (event === 'error') {
            setMessages(old => [...old, { role: 'assistant', text: data.message || '问答服务暂不可用，请稍后重试。', error: true }]);
          }
        },
      });
    } catch (error) {
      if (error.name !== 'AbortError' && !activeController.signal.aborted) {
        setMessages(old => [...old, { role: 'assistant', text: error.message, error: true }]);
      }
    } finally {
      if (controller.current === activeController) {
        controller.current = null;
        lock.current = false;
        setBusy(false);
        setStatusMessage('');
      }
    }
  }
  function reset() {
    controller.current?.abort(); controller.current = null; lock.current = false;
    setBusy(false); setStatusMessage(''); setSession(newSessionId()); setMessages([]); setQuestion('');
  }
  const sessionButton = <button className="secondary" onClick={reset}>＋ 新建会话</button>;
  return <main className="chat-page">
    <div className="page-heading"><div><span className="eyebrow">SMART CUSTOMER SERVICE</span><h1>智能客服</h1><p>使用、维护与故障排查，从这里开始。</p></div>{sessionSlot ? createPortal(sessionButton, sessionSlot) : sessionButton}</div>
    <div className="chat-board">
      <div className="chat-flow" aria-live="polite" ref={flow}>
        {!messages.length && <div className="welcome"><div className="welcome-symbol">◉</div><span className="eyebrow">你好，欢迎使用智扫通</span><h2>今天，T8 遇到了什么问题？</h2><p>从说明书查找依据，经过审核再正式发布。<br />也可以先描述异常，我们会帮你补齐信息。</p>
          <div className="starter-grid">{starters.map((item, i) => <button key={item} onClick={() => setQuestion(item)}><span className="starter-number">0{i + 1}</span><span>{item}</span><b>↗</b></button>)}</div>
          <div className="workflow"><span>用户问题</span><i>→</i><span>资料检索</span><i>→</i><span>人工审核</span><i>→</i><span>正式发布</span></div>
        </div>}
        {messages.map((message, i) => <div className={'message ' + message.role} key={i}><span className="avatar">{message.role === 'user' ? '你' : '◉'}</span><div className="message-body"><div className="message-label">{message.role === 'user' ? '你' : '智扫通'} <SceneBadge intent={message.intent} /> <DeliveryBadge status={message.deliveryStatus} /></div><div className={'bubble ' + (message.error ? 'error-bubble' : '')}>{message.text}</div>{message.pending && <span className="pending-caption">◷ 待质量审核 / 待正式发布审核</span>}</div></div>)}
        {busy && <div className="working" role="status"><span className="spinner" />{statusMessage || '正在处理…'}</div>}
      </div>
      <form className="composer" onSubmit={send}><textarea aria-label="输入问题" placeholder="描述你的 T8 问题…（Enter 发送，Shift+Enter 换行）" value={question} maxLength={500} onChange={e => setQuestion(e.target.value)} onKeyDown={e => {
        if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
      }} /><div className="composer-footer"><span>本会话保留近期上下文 <b>{question.length}/500</b></span><button className="primary" disabled={busy || !question.trim()} type="submit">{busy ? '处理中' : '发送问题'} <span>↑</span></button></div></form>
    </div><p className="footnote">即时答复与正式发布分开管理；复杂或依据不足的回答会转入人工审核。</p>
  </main>;
}
