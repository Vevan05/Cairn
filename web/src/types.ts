export interface SearchResult {
  rank: number;
  chunk_id: string | null;
  text: string;
}
export interface SearchResponse {
  query: string;
  variant: 'control' | 'treatment' | string;
  results: SearchResult[];
  experiment_id: string | null;
}
export interface AnswerResponse {
  query: string;
  answer: string;
  citations: string[];
  variant?: string;
  cached?: boolean;
  retrieved_chunks?: number;
  results?: Array<{ rank: number; chunk_id: string | null; text: string }>;
}
export interface ExperimentSummary {
  id: string;
  name: string;
  status?: string;
  primary_metric?: string | null;
}