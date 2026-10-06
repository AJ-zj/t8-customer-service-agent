import { consumeSseResponse } from './sse.js';
const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const errors = {
  400: '请求数据有误，请检查后重试。', 422: '请求数据有误，请检查后重试。',
  401: '管理员认证失败，请检查令牌。', 403: '管理员认证失败，请检查令牌。',
  404: '记录不存在或尚未发布。', 409: '当前状态不允许执行该操作。',
  503: '系统暂时不可用，请稍后重试。'
};
export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
async function request(path, { method = 'GET', body, token } = {}) {
  let response;
  try {
    response = await fetch(base + path, {
      method, headers: { 'Content-Type': 'application/json', ...(token ? { 'X-Review-Token': token } : {}) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) })
    });
  } catch { throw new ApiError(0, '无法连接服务，请检查网络。'); }
  if (!response.ok) throw new ApiError(response.status, errors[response.status] || '请求失败，请稍后重试。');
  try { return await response.json(); }
  catch { throw new ApiError(response.status, '服务返回无效结果，请稍后重试。'); }
}
export const sendAgentMessage = (question, session_id) => request('/api/agent/chat', {
  method: 'POST', body: { question, session_id }
});
export async function streamAgentMessage(question, session_id, { onEvent = () => {}, signal } = {}) {
  let response;
  try {
    response = await fetch(base + '/api/agent/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ question, session_id }),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(0, '无法连接服务，请检查网络。');
  }
  if (!response.ok) throw new ApiError(response.status, errors[response.status] || '请求失败，请稍后重试。');
  if (!response.headers?.get('content-type')?.toLowerCase().includes('text/event-stream')) {
    throw new ApiError(response.status, '服务返回无效结果，请稍后重试。');
  }
  try { await consumeSseResponse(response, onEvent); }
  catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(0, '流式响应中断，请稍后重试。');
  }
}
export const listReviews = token => request('/internal/reviews?scope=all&limit=50', { token });
export const getReviewDetail = (id, token) => request('/internal/reviews/' + id, { token });
export const submitReviewDecision = (id, decision, review_note, token) => request(
  '/internal/reviews/' + id + '/decision', { method: 'POST', token, body: { decision, review_note } });
export async function publishReview(id, token) {
  try { return await request('/internal/reviews/' + id + '/publish', { method: 'POST', token }); }
  catch (error) { if (error.status === 409) throw new ApiError(409, '当前记录不能发布。'); throw error; }
}
export const getPublishedAnswer = id => request('/api/published/' + id);
export const checkHealth = () => request('/health');
export const checkReady = () => request('/ready');
