/** Parse one complete SSE frame. Comments/heartbeats and empty frames are ignored. */
export function parseSseFrame(frame) {
  const normalized = frame.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  let event = null;
  const dataLines = [];
  for (const line of normalized.split('\n')) {
    if (!line || line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') dataLines.push(value);
  }
  if (!dataLines.length) return null;
  if (!event) throw new Error('SSE event is missing its event name.');
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) };
  } catch {
    throw new Error('SSE event data is invalid JSON.');
  }
}

/** Incrementally frames SSE text; an unterminated frame is discarded at EOF. */
export function createSseParser(onEvent) {
  let buffer = '';
  return {
    push(text) {
      buffer += text;
      const events = [];
      let delimiter;
      while ((delimiter = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, delimiter.index);
        buffer = buffer.slice(delimiter.index + delimiter[0].length);
        const parsed = parseSseFrame(frame);
        if (parsed) {
          events.push(parsed);
          onEvent?.(parsed);
        }
      }
      return events;
    },
    finish() {
      // A truncated network frame must never become a partial UI answer.
      buffer = '';
    },
  };
}

/** Read a POST response.body incrementally, decoding split UTF-8 safely. */
export async function consumeSseResponse(response, onEvent) {
  if (!response.body?.getReader) throw new Error('SSE response body is unavailable.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  const parser = createSseParser(onEvent);
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.finish();
  } finally {
    reader.releaseLock();
  }
}
