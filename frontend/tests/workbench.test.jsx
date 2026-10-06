import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest';
import ChatPage from '../src/pages/ChatPage.jsx';
import ReviewPage from '../src/pages/ReviewPage.jsx';
import PublishedPage from '../src/pages/PublishedPage.jsx';
import { scenes, SceneBadge } from '../src/components/Badges.jsx';
import * as api from '../src/api/client.js';
const pending = { intent: 'maintenance', answer: null, published: false, safety_notice: '本轮真实审核提示' };
const summary = { id: 101, question: '主刷维护问题', audit_status: 'NEEDS_HUMAN_SEMANTIC_REVIEW', review_status: 'pending_review', publication_status: 'unpublished', created_at: '2026-10-06T00:00:00Z' };
const detail = { ...summary, draft_answer: '内部草稿内容', evidence: [{ evidence_id: 'E1', pdf_page: 15, text: '内部证据内容' }] };
let fetchMock;
function reply(data, status = 200) { return { ok: status < 400, status, json: async () => data }; }
const sseFrame = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
function streamReply(events) {
  const payload = events.map(item => typeof item === 'string' ? item : sseFrame(item.event, item.data)).join('');
  const bytes = new TextEncoder().encode(payload);
  return { ok: true, status: 200, headers: { get: name => name.toLowerCase() === 'content-type' ? 'text/event-stream; charset=utf-8' : null },
    body: new ReadableStream({ start(controller) { for (let i = 0; i < bytes.length; i += 17) controller.enqueue(bytes.slice(i, i + 17)); controller.close(); } }) };
}
const pendingStream = () => streamReply([
  { event: 'status', data: { stage: 'received', message: '已收到问题' } },
  { event: 'status', data: { stage: 'processing', message: '正在分析问题' } },
  { event: 'review_required', data: { intent: 'maintenance', delivery_status: 'review_required', message: pending.safety_notice, published: false } },
  { event: 'done', data: { intent: 'maintenance', delivery_status: 'review_required', published: false } },
]);
beforeEach(() => { fetchMock = vi.fn(async path => path.endsWith('/stream') ? pendingStream() : reply(pending)); vi.stubGlobal('fetch', fetchMock); });
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
function chat() { return render(<ChatPage />); }
async function send(text) { await userEvent.type(screen.getByLabelText('输入问题'), text); await userEvent.click(screen.getByRole('button', { name: /发送问题/ })); await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument()); }
function review() {
  fetchMock.mockImplementation(async (path, options) => {
    if (options?.method === 'POST') return reply({});
    return reply(path.includes('?') ? [summary] : detail);
  });
  return render(<MemoryRouter><ReviewPage /></MemoryRouter>);
}
async function login() { await userEvent.type(screen.getByLabelText('Review Token'), 'test-only-token'); await userEvent.click(screen.getByRole('button', { name: '验证并进入' })); await screen.findByRole('heading', { name: '审核记录' }); }
async function open() { await login(); await userEvent.click(screen.getByRole('button', { name: /#101/ })); await screen.findByText('内部草稿内容'); }
describe('chat', () => {
  it('maps every server scene to Chinese', () => { expect(Object.values(scenes)).toEqual(['产品咨询','故障排查','维护建议','维护报告','配置推荐','信息澄清','人工客服']); render(<SceneBadge intent="maintenance" />); expect(screen.getByText('维护建议')).toBeInTheDocument(); });
  it('sends the trimmed question to the stream endpoint with a signal', async () => { chat(); await send(' 主刷怎么清理？ '); const [url, options] = fetchMock.mock.calls[0]; const body = JSON.parse(options.body); expect(url).toBe('/api/agent/chat/stream'); expect(options.headers.Accept).toBe('text/event-stream'); expect(options.signal).toBeInstanceOf(AbortSignal); expect(body.question).toBe('主刷怎么清理？'); expect(body.session_id).toMatch(/^[A-Za-z0-9_-]{1,64}$/); expect(Object.keys(body)).toEqual(['question','session_id']); });
  it('reuses session for follow up', async () => { chat(); await send('主刷怎么清理？'); await send('那它多久清理？'); expect(JSON.parse(fetchMock.mock.calls[0][1].body).session_id).toBe(JSON.parse(fetchMock.mock.calls[1][1].body).session_id); });
  it('new session clears messages and replaces id', async () => { chat(); await send('第一问题'); const first = JSON.parse(fetchMock.mock.calls[0][1].body).session_id; await userEvent.click(screen.getByRole('button', { name: /新建会话/ })); expect(screen.queryByText('第一问题')).not.toBeInTheDocument(); await send('第二问题'); expect(JSON.parse(fetchMock.mock.calls[1][1].body).session_id).not.toBe(first); });
  it('renders troubleshooting clarification from streamed answer chunks', async () => { fetchMock.mockImplementation(async () => streamReply([
    { event: 'status', data: { stage: 'processing', message: '正在处理…' } },
    { event: 'answer_start', data: { intent: 'troubleshooting', delivery_status: 'clarification', published: false } },
    { event: 'answer_delta', data: { text: '请描述' } }, { event: 'answer_delta', data: { text: '具体异常。' } },
    { event: 'done', data: { intent: 'troubleshooting', delivery_status: 'clarification', published: false } },
  ])); chat(); await send('坏了'); expect(screen.getByText('请描述具体异常。')).toBeInTheDocument(); });
  it('renders pending notice without hidden draft or evidence', async () => { chat(); await send('主刷'); expect(screen.getByText(pending.safety_notice)).toBeInTheDocument(); expect(screen.queryByText('draft_answer')).not.toBeInTheDocument(); expect(screen.queryByText('evidence')).not.toBeInTheDocument(); });
  it('renders auto-released answer with its delivery badge', async () => { fetchMock.mockImplementation(async () => streamReply([
    { event: 'answer_start', data: { intent: 'maintenance', delivery_status: 'auto_released', published: false } },
    { event: 'answer_delta', data: { text: '按说明书清理主刷。' } }, { event: 'answer_delta', data: { text: '[E1]' } },
    { event: 'done', data: { intent: 'maintenance', delivery_status: 'auto_released', published: false } },
  ])); chat(); await send('主刷怎么清理？'); expect(screen.getByText('按说明书清理主刷。[E1]')).toBeInTheDocument(); expect(screen.getByText('即时答复')).toBeInTheDocument(); expect(screen.queryByText(/待质量审核/)).not.toBeInTheDocument(); });
  it('renders safe abstention from streamed fixed answer only', async () => { fetchMock.mockImplementation(async () => streamReply([
    { event: 'answer_start', data: { intent: 'consult_qa', delivery_status: 'safe_abstention', published: false } },
    { event: 'answer_delta', data: { text: '根据当前提供的说明书资料无法确认。' } },
    { event: 'done', data: { intent: 'consult_qa', delivery_status: 'safe_abstention', published: false } },
  ])); chat(); await send('错误码401是什么意思？'); expect(screen.getByText('根据当前提供的说明书资料无法确认。')).toBeInTheDocument(); expect(screen.getByText('资料不足')).toBeInTheDocument(); });
  it('renders review-required notice instead of any answer', async () => { fetchMock.mockImplementation(async () => pendingStream()); chat(); await send('主刷怎么清理？'); expect(screen.getByText(pending.safety_notice)).toBeInTheDocument(); expect(screen.getByText('人工审核')).toBeInTheDocument(); });
  it('shows auto-released answer even while published is false', async () => { fetchMock.mockImplementation(async () => streamReply([
    { event: 'answer_start', data: { intent: 'consult_qa', delivery_status: 'auto_released', published: false } },
    { event: 'answer_delta', data: { text: '公开前可实时显示的安全答案。' } },
    { event: 'done', data: { intent: 'consult_qa', delivery_status: 'auto_released', published: false } },
  ])); chat(); await send('拖地模式怎么开启？'); expect(screen.getByText('公开前可实时显示的安全答案。')).toBeInTheDocument(); expect(screen.getByText('即时答复')).toBeInTheDocument(); });
  it('renders real handoff text', async () => { fetchMock.mockImplementation(async () => streamReply([
    { event: 'answer_start', data: { intent: 'human_handoff', delivery_status: 'human_handoff', published: false } },
    { event: 'answer_delta', data: { text: '真实转人工提示' } },
    { event: 'done', data: { intent: 'human_handoff', delivery_status: 'human_handoff', published: false } },
  ])); chat(); await send('我要人工客服'); expect(screen.getByText('真实转人工提示')).toBeInTheDocument(); });
  it('Enter sends while Shift Enter does not', async () => { chat(); const input = screen.getByLabelText('输入问题'); fireEvent.change(input, { target: { value: '主刷' } }); fireEvent.keyDown(input, { key: 'Enter', shiftKey: true }); expect(fetchMock).not.toHaveBeenCalled(); fireEvent.keyDown(input, { key: 'Enter' }); await screen.findByText(pending.safety_notice); expect(fetchMock).toHaveBeenCalledTimes(1); });
  it('blocks duplicate sends while in flight', async () => { let finish; fetchMock.mockImplementation(() => new Promise(resolve => { finish = resolve; })); chat(); fireEvent.change(screen.getByLabelText('输入问题'), { target: { value: '主刷' } }); fireEvent.keyDown(screen.getByLabelText('输入问题'), { key: 'Enter' }); fireEvent.keyDown(screen.getByLabelText('输入问题'), { key: 'Enter' }); expect(fetchMock).toHaveBeenCalledTimes(1); finish(pendingStream()); await screen.findByText(pending.safety_notice); });
  it('shows processing status before the stream finishes', async () => {
    let streamController;
    fetchMock.mockResolvedValue({ ok: true, status: 200,
      headers: { get: () => 'text/event-stream' },
      body: new ReadableStream({ start(controller) { streamController = controller; controller.enqueue(new TextEncoder().encode(sseFrame('status', { stage: 'processing', message: '正在处理…' }))); } }) });
    chat(); await userEvent.type(screen.getByLabelText('输入问题'), '主刷');
    await userEvent.click(screen.getByRole('button', { name: /发送问题/ }));
    expect(await screen.findByText('正在处理…')).toBeInTheDocument();
    streamController.enqueue(new TextEncoder().encode(sseFrame('answer_start', { intent: 'maintenance', delivery_status: 'safe_abstention', published: false }) +
      sseFrame('answer_delta', { text: '安全回答' }) + sseFrame('done', { intent: 'maintenance', delivery_status: 'safe_abstention', published: false })));
    streamController.close(); await screen.findByText('安全回答');
  });
  it('appends two answer deltas to one assistant bubble', async () => {
    fetchMock.mockResolvedValue(streamReply([
      { event: 'answer_start', data: { intent: 'maintenance', delivery_status: 'auto_released', published: false } },
      { event: 'answer_delta', data: { text: '第一段' } }, { event: 'answer_delta', data: { text: '第二段' } },
      { event: 'done', data: { intent: 'maintenance', delivery_status: 'auto_released', published: false } },
    ])); chat(); await send('问题'); expect(screen.getByText('第一段第二段')).toBeInTheDocument();
  });
  it('does not append the done metadata as a second answer', async () => {
    fetchMock.mockResolvedValue(streamReply([
      { event: 'answer_start', data: { intent: 'maintenance', delivery_status: 'auto_released', published: false } },
      { event: 'answer_delta', data: { text: '唯一答案' } },
      { event: 'done', data: { intent: 'maintenance', delivery_status: 'auto_released', published: false, audit_status: 'NEEDS_HUMAN_SEMANTIC_REVIEW' } },
    ])); chat(); await send('问题'); expect(screen.getAllByText('唯一答案')).toHaveLength(1);
  });
  it('shows streamed safe abstention with its badge', async () => {
    fetchMock.mockResolvedValue(streamReply([
      { event: 'answer_start', data: { intent: 'consult_qa', delivery_status: 'safe_abstention', published: false } },
      { event: 'answer_delta', data: { text: '根据当前提供的说明书资料无法确认。' } },
      { event: 'done', data: { intent: 'consult_qa', delivery_status: 'safe_abstention', published: false } },
    ])); chat(); await send('问题'); expect(screen.getByText('资料不足')).toBeInTheDocument();
  });
  it('shows review notice without creating an empty answer delta bubble', async () => {
    fetchMock.mockResolvedValue(streamReply([
      { event: 'review_required', data: { intent: 'maintenance', delivery_status: 'review_required', message: '仅有审核提示', published: false } },
      { event: 'done', data: { intent: 'maintenance', delivery_status: 'review_required', published: false } },
    ])); chat(); await send('问题'); expect(screen.getByText('仅有审核提示')).toBeInTheDocument();
  });
  it('renders clarification chunks from the server', async () => {
    fetchMock.mockResolvedValue(streamReply([
      { event: 'answer_start', data: { intent: 'clarify', delivery_status: 'clarification', published: false } },
      { event: 'answer_delta', data: { text: '请补充' } }, { event: 'answer_delta', data: { text: '异常信息。' } },
      { event: 'done', data: { intent: 'clarify', delivery_status: 'clarification', published: false } },
    ])); chat(); await send('问题'); expect(screen.getByText('请补充异常信息。')).toBeInTheDocument();
  });
  it('renders handoff chunks from the server', async () => {
    fetchMock.mockResolvedValue(streamReply([
      { event: 'answer_start', data: { intent: 'human_handoff', delivery_status: 'human_handoff', published: false } },
      { event: 'answer_delta', data: { text: '已为你转入人工处理流程。' } },
      { event: 'done', data: { intent: 'human_handoff', delivery_status: 'human_handoff', published: false } },
    ])); chat(); await send('转人工'); expect(screen.getByText('已为你转入人工处理流程。')).toBeInTheDocument();
  });
  it('renders safe error events without exposing upstream details', async () => {
    fetchMock.mockResolvedValue(streamReply([{ event: 'error', data: { code: 'SERVICE_UNAVAILABLE', message: '问答服务暂不可用，请稍后重试。' } }]));
    chat(); await send('问题'); expect(screen.getByText('问答服务暂不可用，请稍后重试。')).toBeInTheDocument();
  });
  it('aborts browser consumption when starting a new session', async () => {
    let capturedSignal;
    fetchMock.mockImplementation((_, options) => {
      capturedSignal = options.signal;
      return new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))));
    });
    chat(); await userEvent.type(screen.getByLabelText('输入问题'), '正在处理');
    await userEvent.click(screen.getByRole('button', { name: /发送问题/ }));
    await screen.findByRole('status'); await userEvent.click(screen.getByRole('button', { name: /新建会话/ }));
    expect(capturedSignal.aborted).toBe(true); expect(screen.queryByText('正在处理')).not.toBeInTheDocument();
  });
  it('does not render heartbeat comments as chat text', async () => {
    fetchMock.mockResolvedValue(streamReply([
      ': ping\n\n',
      { event: 'answer_start', data: { intent: 'consult_qa', delivery_status: 'auto_released', published: false } },
      { event: 'answer_delta', data: { text: '答案' } }, { event: 'done', data: { intent: 'consult_qa', delivery_status: 'auto_released', published: false } },
    ])); chat(); await send('问题'); expect(screen.getByText('答案')).toBeInTheDocument(); expect(screen.queryByText('ping')).not.toBeInTheDocument();
  });
  it('does not retry the JSON endpoint when the stream transport fails', async () => {
    fetchMock.mockRejectedValue(new TypeError('network dropped'));
    chat(); await send('问题');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/agent/chat/stream');
    expect(screen.getByText('无法连接服务，请检查网络。')).toBeInTheDocument();
  });
});
describe('review', () => {
  it('does not persist token in either browser storage', async () => { const spy = vi.spyOn(Storage.prototype, 'setItem'); review(); await login(); expect(spy).not.toHaveBeenCalled(); expect(fetchMock.mock.calls[0][1].headers['X-Review-Token']).toBe('test-only-token'); expect(fetchMock.mock.calls[0][0]).not.toContain('test-only-token'); });
  it('unmount removes token', async () => { const view = review(); await login(); view.unmount(); review(); expect(screen.getByLabelText('Review Token')).toHaveValue(''); });
  it('shows authentication failure safely', async () => { review(); fetchMock.mockResolvedValue(reply({ detail: 'secret SQL' }, 401)); await userEvent.type(screen.getByLabelText('Review Token'), 'wrong'); await userEvent.click(screen.getByRole('button', { name: '验证并进入' })); expect(await screen.findByRole('alert')).toHaveTextContent('管理员认证失败'); expect(screen.queryByText('secret SQL')).not.toBeInTheDocument(); });
  it('list shows summaries only', async () => { review(); await login(); expect(screen.getByText(summary.question)).toBeInTheDocument(); expect(screen.queryByText(detail.draft_answer)).not.toBeInTheDocument(); });
  it('detail shows authorized draft and snapshot', async () => { review(); await open(); await userEvent.click(screen.getByText(/PDF 第 15 页/)); expect(screen.getByText('内部证据内容')).toBeInTheDocument(); });
  it('requires nonblank review note', async () => { review(); await open(); expect(screen.getByRole('button', { name: '通过审核' })).toBeDisabled(); });
  it('approve refreshes server state but does not publish', async () => { review(); await open(); fetchMock.mockImplementation(async (path, options) => reply(options?.method === 'POST' ? {} : path.includes('?') ? [{ ...summary, review_status: 'approved' }] : { ...detail, review_status: 'approved' })); await userEvent.type(screen.getByLabelText('审核备注'), '已核对'); await userEvent.click(screen.getByRole('button', { name: '通过审核' })); await screen.findByRole('button', { name: '正式发布' }); expect(fetchMock.mock.calls.some(([path]) => path.endsWith('/publish'))).toBe(false); const call = fetchMock.mock.calls.find(([path]) => path.endsWith('/decision')); expect(JSON.parse(call[1].body)).toEqual({ decision: 'approved', review_note: '已核对' }); });
  it('reject calls decision and removes publish action', async () => { review(); await open(); fetchMock.mockImplementation(async (path, options) => reply(options?.method === 'POST' ? {} : path.includes('?') ? [{ ...summary, review_status: 'rejected' }] : { ...detail, review_status: 'rejected' })); await userEvent.type(screen.getByLabelText('审核备注'), '依据不足'); await userEvent.click(screen.getByRole('button', { name: '驳回' })); await waitFor(() => expect(screen.queryByRole('button', { name: '驳回' })).not.toBeInTheDocument()); expect(screen.queryByRole('button', { name: '正式发布' })).not.toBeInTheDocument(); expect(JSON.parse(fetchMock.mock.calls.find(([path]) => path.endsWith('/decision'))[1].body).decision).toBe('rejected'); });
  it('publishes only through an independent request', async () => { review(); await open(); fetchMock.mockImplementation(async path => reply(path.includes('?') ? [{ ...summary, review_status: 'approved' }] : { ...detail, review_status: 'approved' })); await userEvent.click(screen.getByRole('button', { name: '刷新' })); await userEvent.click(await screen.findByRole('button', { name: '正式发布' })); await waitFor(() => expect(fetchMock.mock.calls.some(([path, opt]) => path.endsWith('/publish') && opt.method === 'POST')).toBe(true)); });
});
describe('public and errors', () => {
  function publicPage() { render(<MemoryRouter initialEntries={['/published/101']}><Routes><Route path="/published/:recordId" element={<PublishedPage />} /></Routes></MemoryRouter>); }
  it('public shows only published fields', async () => { fetchMock.mockResolvedValue(reply({ question: '公开问题', answer: '正式答案', published_at: '2026-10-06', evidence: '内部证据', review_note: '审核意见' })); publicPage(); expect(await screen.findByText('正式答案')).toBeInTheDocument(); expect(screen.queryByText('内部证据')).not.toBeInTheDocument(); expect(screen.queryByText('审核意见')).not.toBeInTheDocument(); });
  it('404 does not reveal internal state', async () => { fetchMock.mockResolvedValue(reply({ detail: 'approved unpublished' }, 404)); publicPage(); expect(await screen.findByRole('alert')).toHaveTextContent('该回答尚未发布或不存在。'); });
  it('503 is friendly and hides upstream text', async () => { fetchMock.mockResolvedValue(reply({ detail: 'Traceback database password' }, 503)); chat(); await send('主刷'); expect(screen.getByText('系统暂时不可用，请稍后重试。')).toBeInTheDocument(); });
  it('network failure is friendly', async () => { fetchMock.mockRejectedValue(new Error('raw connection secret')); await expect(api.checkHealth()).rejects.toThrow('无法连接服务'); });
  it('publish conflict is specific', async () => { fetchMock.mockResolvedValue(reply({}, 409)); await expect(api.publishReview(101, 'test')).rejects.toThrow('当前记录不能发布。'); });
  it('422 hides backend payload', async () => { fetchMock.mockResolvedValue(reply({ detail: 'internal' }, 422)); await expect(api.sendAgentMessage('', 'test')).rejects.toThrow('请求数据有误'); });
  it('readiness uses its real endpoint', async () => { await api.checkReady(); expect(fetchMock.mock.calls[0][0]).toBe('/ready'); });
});
