import { describe, it, expect } from 'vitest';
import { consumeSseResponse, createSseParser, parseSseFrame } from '../src/api/sse.js';

const frame = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
function responseWithByteChunks(chunks) {
  return {
    body: new ReadableStream({
      start(controller) { for (const chunk of chunks) controller.enqueue(chunk); controller.close(); },
    }),
  };
}

describe('incremental SSE parser', () => {
  it('parses a complete single frame', () => {
    expect(parseSseFrame(frame('status', { stage: 'received' }).trim())).toEqual({
      event: 'status', data: { stage: 'received' },
    });
  });

  it('buffers a frame split across two pushes', () => {
    const seen = [];
    const parser = createSseParser(event => seen.push(event));
    parser.push('event: status\ndata: {"stage":');
    expect(seen).toHaveLength(0);
    parser.push('"processing"}\n\n');
    expect(seen).toEqual([{ event: 'status', data: { stage: 'processing' } }]);
  });

  it('emits multiple frames received in one chunk', () => {
    const parser = createSseParser();
    expect(parser.push(frame('status', { stage: 'received' }) + frame('done', { published: false })))
      .toHaveLength(2);
  });

  it('decodes Chinese split inside a UTF-8 byte sequence', async () => {
    const bytes = new TextEncoder().encode(frame('answer_delta', { text: '中文答案' }));
    const split = bytes.findIndex(byte => byte >= 0xE4) + 1;
    const events = [];
    await consumeSseResponse(responseWithByteChunks([bytes.slice(0, split), bytes.slice(split)]), e => events.push(e));
    expect(events[0].data.text).toBe('中文答案');
  });

  it('ignores heartbeat comments', () => {
    const events = [];
    const parser = createSseParser(event => events.push(event));
    parser.push(': ping\n\n');
    expect(events).toEqual([]);
  });

  it('rejects invalid JSON without echoing the input', () => {
    expect(() => parseSseFrame('event: done\ndata: {secret\n\n'.trim()))
      .toThrow('SSE event data is invalid JSON.');
  });

  it('joins multiline data before JSON parsing', () => {
    expect(parseSseFrame('event: status\ndata: {\ndata: "stage":"received"}\n\n'))
      .toEqual({ event: 'status', data: { stage: 'received' } });
  });

  it('discards an incomplete frame when the stream closes', async () => {
    const events = [];
    const bytes = new TextEncoder().encode('event: answer_delta\ndata: {"text":"partial"}');
    await consumeSseResponse(responseWithByteChunks([bytes]), e => events.push(e));
    expect(events).toEqual([]);
  });
});
