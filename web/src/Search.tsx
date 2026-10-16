import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import * as api from './api';
import Experiments from './Experiments';

type View = 'search' | 'answer' | 'experiments';

const TAB_DESCRIPTIONS: Record<View, string> = {
  search: 'Find the closest passages in the corpus, ranked by cosine similarity. No LLM — just fast, from-scratch retrieval.',
  answer: 'Retrieve the best passages, then generate a plain-language answer with citations back to the source chunks.',
  experiments: 'A/B test retrieval configs. Deterministic bucketing, z-tests, and Sample Ratio Mismatch checks — all built in.',
};

export function Search() {
  const [query, setQuery] = useState('');
  const [userId, setUserId] = useState(() => 'u_' + Math.floor(Math.random() * 100000));
  const [results, setResults] = useState<any[]>([]);
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState<string[]>([]);
  const [variant, setVariant] = useState('');
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [view, setView] = useState<View>('search');

  useEffect(() => {
    api.health().then(() => setOnline(true)).catch(() => setOnline(false));
  }, []);

  async function run(fn: 'search' | 'answer') {
    if (!query.trim()) return;
    setView(fn);
    setLoading(true);
    try {
      if (fn === 'search') {
        const r = await api.search(query, userId);
        setResults(r.results || []);
        setVariant(r.variant || '');
        setAnswer('');
        setCitations([]);
      } else {
        const r = await api.answer(query, userId);
        setAnswer(r.answer || '(no answer returned)');
        setCitations(r.citations || []);
        setVariant(r.variant || variant);
      }
    } catch (err: any) {
      alert('Request failed: ' + err);
    } finally {
      setLoading(false);
    }
  }

  async function feedback(chunkId: string | null, v: number) {
    if (!chunkId) return;
    try {
      await api.feedback({ event_id: 'ev_' + Date.now(), user_id: userId, chunk_id: chunkId, metric: 'thumbs', value: v });
    } catch {}
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-slate-900/80 backdrop-blur border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm">C</div>
              <span className="font-semibold text-lg tracking-tight">Cairn</span>
            </Link>
          </div>
          <nav className="flex gap-1 ml-4">
            <button onClick={() => setView('search')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${view === 'search' ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300' : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}>
              Search
            </button>
            <button onClick={() => setView('answer')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${view === 'answer' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300' : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}>
              Answer
            </button>
            <button onClick={() => setView('experiments')}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${view === 'experiments' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300' : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}>
              Experiments
            </button>
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {variant && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 font-medium">
                {variant}
              </span>
            )}
            <span className={`w-2 h-2 rounded-full ${online ? 'bg-green-500' : online === false ? 'bg-red-500' : 'bg-slate-400 animate-pulse'}`} />
            <input
              className="w-32 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 px-3 py-1 text-xs"
              value={userId} onChange={e => setUserId(e.target.value)} placeholder="user_id"
            />
          </div>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────── */}
      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        <p className="text-sm text-slate-500 dark:text-slate-400 -mb-2">{TAB_DESCRIPTIONS[view]}</p>
        {view === 'experiments' ? (
          <Experiments />
        ) : (
        <>

        {/* Query input */}
        <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-6">
          <label className="block text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">
            {view === 'search' ? 'Search query' : 'Ask a question'}
          </label>
          <textarea
            className="w-full min-h-[80px] p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 resize-y focus:ring-2 focus:ring-indigo-500 focus:outline-none text-base"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(view); } }}
            placeholder={view === 'search' ? 'e.g. battleship decommissioning…' : 'e.g. What is the capital of France?'}
          />
          <div className="flex gap-2 mt-4">
            <button
              disabled={loading || !query.trim()}
              onClick={() => run(view)}
              className="px-6 py-2.5 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Loading…' : view === 'search' ? 'Search' : 'Get Answer'}
            </button>
            {query && (
              <button onClick={() => { setQuery(''); setResults([]); setAnswer(''); setCitations([]); }}
                className="px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors">
                Clear
              </button>
            )}
          </div>
        </section>

        {/* LLM Answer */}
        {answer && view === 'answer' && (
          <section className="rounded-2xl border border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950 p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">Answer</h3>
            <p className="whitespace-pre-wrap leading-relaxed text-emerald-900 dark:text-emerald-100">{answer}</p>
            {citations.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {citations.map(c => (
                  <span key={c} className="px-2 py-1 rounded-md bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 text-xs font-mono">{c}</span>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Results */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-700 dark:text-slate-300">Results</h3>
            {results.length > 0 && (
              <span className="text-xs text-slate-400">{results.length} chunks</span>
            )}
          </div>

          {loading && (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-24 rounded-2xl bg-slate-200 dark:bg-slate-800 animate-pulse" />
              ))}
            </div>
          )}

          {!loading && results.length === 0 && (
            <div className="text-center text-slate-400 dark:text-slate-500 py-12 border border-dashed border-slate-300 dark:border-slate-700 rounded-2xl">
              <p className="text-lg mb-1">No results yet</p>
              <p className="text-sm">Enter a query and press Search or Get Answer</p>
            </div>
          )}

          {results.map((r, i) => (
            <article key={i} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 hover:border-indigo-300 dark:hover:border-indigo-700 transition-colors">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xs font-bold text-slate-400 w-6">#{r.rank}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 font-mono">{r.chunk_id}</span>
                <div className="ml-auto flex items-center gap-2">
                  <div className="text-xs font-semibold text-indigo-600 dark:text-indigo-400">
                    {typeof r.score === 'number' ? r.score.toFixed(3) : '—'}
                  </div>
                  <div className="w-20 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${Math.min(100, (r.score || 0) * 100)}%` }} />
                  </div>
                  <button onClick={() => feedback(r.chunk_id, 1)} className="p-1 rounded hover:bg-green-100 dark:hover:bg-green-900 text-base leading-none" title="Helpful">👍</button>
                  <button onClick={() => feedback(r.chunk_id, -1)} className="p-1 rounded hover:bg-red-100 dark:hover:bg-red-900 text-base leading-none" title="Not helpful">👎</button>
                </div>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed line-clamp-4">{r.text}</p>
            </article>
          ))}
        </section>
        </>
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="max-w-5xl mx-auto px-6 py-6 text-center text-xs text-slate-400 dark:text-slate-600">
        Cairn · from-scratch RAG search engine · pure Python engine · free-tier deployable
      </footer>
    </div>
  );
}

export default Search;
