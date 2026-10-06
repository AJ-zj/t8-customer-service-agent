import { NavLink } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { checkHealth, checkReady } from '../api/client.js';

export default function AppShell({ children }) {
  const [ready, setReady] = useState(null);
  useEffect(() => { let alive = true; Promise.all([checkHealth(), checkReady()])
    .then(() => alive && setReady(true)).catch(() => alive && setReady(false));
    return () => { alive = false; }; }, []);
  return <div className="shell">
    <aside className="sidebar">
      <NavLink to="/" className="brand"><span className="brand-mark">◉</span><span>智扫通<small>石头 T8 智能客服系统</small></span></NavLink>
      <span className="eyebrow side-label">工作空间</span>
      <nav><NavLink to="/" end>◌ <span>智能客服</span></NavLink><NavLink to="/review">▤ <span>人工审核</span></NavLink></nav>
      <div id="new-session-slot" className="sidebar-session" />
      <div className="side-info"><span className="small-icon">↗</span><h3>有依据，才有答复</h3><p>智能场景路由<br />混合 RAG 检索<br />多轮故障排查<br />人工审核发布</p></div>
      <div className="sidebar-bottom"><span className={'dot ' + (ready ? 'green' : '')} />{ready === null ? '检查服务中' : ready ? '服务已就绪' : '服务暂不可用'}<small>官方说明书知识 · 人工审核保障</small></div>
    </aside>
    <div className="workspace"><header className="topbar"><span>客服工作台 <b>/ 石头 T8</b></span><span className="header-chip">官方说明书知识库</span></header>{children}</div>
  </div>;
}
