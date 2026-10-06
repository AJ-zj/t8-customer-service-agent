import { Routes, Route } from 'react-router-dom';
import AppShell from './components/AppShell.jsx';
import ChatPage from './pages/ChatPage.jsx';
import ReviewPage from './pages/ReviewPage.jsx';
import PublishedPage from './pages/PublishedPage.jsx';

export default function App() {
  return <AppShell><Routes>
    <Route path="/" element={<ChatPage />} />
    <Route path="/review" element={<ReviewPage />} />
    <Route path="/published/:recordId" element={<PublishedPage />} />
    <Route path="*" element={<p className="empty">页面不存在，请从左侧导航返回。</p>} />
  </Routes></AppShell>;
}
