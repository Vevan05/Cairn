import type { SearchResponse, AnswerResponse } from './types';

// In dev: empty string -> Vite proxy forwards to :8000
// In prod: VITE_API_URL points at the Render backend, or rewrites handle it
const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j?.detail) detail = j.detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  const body = await res.text();
  try {
    return JSON.parse(body) as T;
  } catch {
    // Almost always means API_BASE is empty, so the request hit the static
    // frontend, which answers unknown paths with index.html instead of JSON.
    throw new Error(
      `Expected JSON from ${API_BASE || location.origin}${path} but got ` +
      `${res.headers.get('content-type') ?? 'a non-JSON body'}.` +
      (API_BASE ? '' : ' Set VITE_API_URL to the deployed API URL and redeploy.'),
    );
  }
}

function post<T>(path: string, data: Record<string, unknown>): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}
function patch<T>(path: string, data: Record<string, unknown>): Promise<T> {
  return request<T>(path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export interface HealthResponse {
  status: string;
  index_version: string | null;
  vector_count: number;
  chunk_count: number;
}

export const health = () => request<HealthResponse>('/health');

export const search = (q: string, userId: string) =>
  post<SearchResponse>('/search', { query: q, user_id: userId });

export const answer = (q: string, userId: string) =>
  post<AnswerResponse>('/answer', { query: q, user_id: userId });

export const feedback = (payload: {
  event_id: string;
  user_id: string;
  chunk_id?: string;
  metric: string;
  value: number;
  variant?: string;
  experiment_id?: string;
}) => post<{ stored: number; idempotent: boolean }>('/events', { events: [payload] });

// --- Experiments ---
export async function listExperiments(): Promise<{ experiments: any[]; active?: string }> {
  try { return await request<{ experiments: any[]; active?: string }>('/experiments'); }
  catch { return { experiments: [] }; }
}
export const getExperiment = (id: string) => request<any>(`/experiments/${id}`);
export const getResults = (id: string) => request<any>(`/experiments/${id}/results`);
export const createExperiment = (body: Record<string, unknown>) => post<any>('/experiments', body);
export const updateExperiment = (id: string, body: Record<string, unknown>) => patch<any>(`/experiments/${id}`, body);
export const activateExperiment = (id: string) => post<any>(`/experiments/${id}/activate`, {});
export const simulateExperiment = (id: string, body: Record<string, unknown>) => post<any>(`/experiments/${id}/simulate`, body);
export const deleteExperiment = (id: string) => request<any>(`/experiments/${id}`, { method: 'DELETE' });

// --- Eval ---
export const listEvalSets = () => request<{ sets: any[] }>('/eval/sets');
export const createEvalSet = (body: Record<string, unknown>) => post<any>('/eval/sets', body);
export const seedSampleEvalSet = () => post<any>('/eval/sets/sample', {});
export const runEval = (body: Record<string, unknown>) => post<any>('/eval/run', body);

// --- Streaming answer (SSE over fetch) ---
export interface StreamHandlers {
  onChunks?: (chunks: any[]) => void;
  onToken?: (token: string) => void;
  onDone?: (data: { answer: string; citations: string[] }) => void;
  onError?: (err: unknown) => void;
}

export async function streamAnswer(query: string, userId: string, handlers: StreamHandlers) {
  const url = `${API_BASE}/answer/stream?query=${encodeURIComponent(query)}&user_id=${encodeURIComponent(userId)}`;
  try {
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);
    const ctype = res.headers.get('content-type') ?? '';
    if (!ctype.includes('text/event-stream')) {
      throw new Error(
        `Expected an event stream from ${API_BASE || location.origin}/answer/stream but got ${ctype}.` +
        (API_BASE ? '' : ' Set VITE_API_URL to the deployed API URL and redeploy.'),
      );
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const flushEvent = (raw: string) => {
      const evMatch = raw.match(/^event:\s*(.+)$/m);
      const dataMatch = raw.match(/^data:\s*(.+)$/m);
      if (!dataMatch) return;
      const event = evMatch ? evMatch[1].trim() : 'message';
      const data = dataMatch[1];
      try {
        if (event === 'chunks') handlers.onChunks?.(JSON.parse(data));
        else if (event === 'token') handlers.onToken?.(JSON.parse(data));
        else if (event === 'done') handlers.onDone?.(JSON.parse(data));
      } catch { /* ignore malformed */ }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        flushEvent(raw);
      }
    }
  } catch (err) {
    handlers.onError?.(err);
  }
}
