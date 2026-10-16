import type { SearchResponse, AnswerResponse, ExperimentSummary } from './types';

// In dev: empty string -> Vite proxy forwards to :8000
// In prod: VITE_API_URL points at the Render backend, or rewrites handle it
const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} (${path})`);
  return (await res.json()) as T;
}

function post<T>(path: string, data: Record<string, unknown>): Promise<T> {
  return request<T>(path, {
    method: 'POST',
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

export const feedback = (payload: { event_id: string; user_id: string; chunk_id?: string; metric: string; value: number }) =>
  post<{ stored: number; idempotent: boolean }>('/events', { events: [payload] });

export async function listExperiments(): Promise<ExperimentSummary[]> {
  try {
    const r = await request<{ experiments: ExperimentSummary[] }>('/experiments');
    return r.experiments ?? [];
  } catch {
    return [];
  }
}
