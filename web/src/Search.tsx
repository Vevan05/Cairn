import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import * as api from './api';
import Experiments from './Experiments';
import Eval from './Eval';

type View = 'search' | 'answer' | 'experiments' | 'eval';

const TAB_DESCRIPTIONS: Record<View, string> = {
  search: 'Find the closest passages in the corpus, ranked by cosine similarity. No LLM — just fast, from-scratch retrieval.',
  answer: 'Retrieve the best passages, then stream a citation-grounded answer. Demo mode: the prose is templated by a stub — retrieval, streaming and citations are real, but no LLM is called.',
  experiments: 'A/B test retrieval configs. Create experiments, set variant weights, and read z-tests + SRM checks.',
  eval: 'Score retrieval configs offline on a labeled set: recall@k, MRR, and nDCG@k for vector vs hybrid.',
};

const TABS: { id: View; label: string; color: string }[] = [
  { id: 'search', label: 'Search', color: 'indigo' },
  { id: 'answer', label: 'Answer', color: 'emerald' },
  { id: 'experiments', label: 'Experiments', color: 'purple' },
  { id: 'eval', label: 'Eval', color: 'sky' },
];

const ACTIVE: Record<string, string> = {
  indigo: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300',
  emerald: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300',
  purple: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
  sky: 'bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-300',
};

// Accent band styles for the per-tab description, matched to each tab's color.
const DESC_ACCENT: Record<View, string> = {
  search: 'border-l-indigo-400 bg-indigo-50/50 dark:bg-indigo-950/30',
  answer: 'border-l-emerald-400 bg-emerald-50/50 dark:bg-emerald-950/30',
  experiments: 'border-l-purple-400 bg-purple-50/50 dark:bg-purple-950/30',
  eval: 'border-l-sky-400 bg-sky-50/50 dark:bg-sky-950/30',
};

export function Search() {
  const [query, setQuery] = useState('');
  const [userId, setUserId] = useState(() => 'u_' + Math.floor(Math.random() * 100000));
  const [results, setResults] = useState<any[]>([]);
  const [chunks, setChunks] = useState<any[]>([]);
  const [answerText, setAnswerText] = useState('');
  const [citations, setCitations] = useState<string[]>([]);
  const [variant, setVariant] = useState('');
  const [experimentId, setExperimentId] = useState('');
  const [loading, setLoading] = useState(false);
  const [rated, setRated] = useState<Record<string, 1 | -1>>({});
  const [toast, setToast] = useState('');
  const [online, setOnline] = useState<boolean | null>(null);
  const [warming, setWarming] = useState(false);
  const [healthError, setHealthError] = useState('');
  const [view, setView] = useState<View>('search');
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const chunksRef = useRef<HTMLDivElement>(null);

  // Poll /health instead of checking once: on Render's free tier the first
  // request of the session cold-starts the API (~30-60s), so a single
  // mount-time check reports "offline" even though the backend is fine.
  const markReachable = () => { setOnline(true); setWarming(false); setHealthError(''); };

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const startedAt = Date.now();
    let delay = 2000;

    const check = async () => {
      try {
        await api.health();
        if (cancelled) return;
        setOnline(true); setWarming(false); setHealthError('');
        delay = 60000; // healthy -> re-check once a minute
      } catch (e) {
        if (cancelled) return;
        const cold = Date.now() - startedAt < 60000;
        setOnline(false); setWarming(cold);
        setHealthError(String((e as Error)?.message ?? e));
        delay = cold ? Math.min(delay * 1.5, 10000) : 30000;
      }
      if (!cancelled) timer = setTimeout(check, delay);
    };

    check();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  async function runSearch() {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const r = await api.search(query, userId);
      setResults(r.results || []);
      setVariant(r.variant || '');
      setExperimentId(r.experiment_id || '');
      markReachable();          // a successful call proves the API is up
    } catch (err: any) { alert('Search failed: ' + err); }
    finally { setLoading(false); }
  }

  async function runAnswer() {
    if (!query.trim()) return;
    setLoading(true);
    setAnswerText(''); setChunks([]); setCitations([]); setHighlighted(null);
    await api.streamAnswer(query, userId, {
      onChunks: (c) => { setChunks(c); markReachable(); },
      onToken: (t) => { setAnswerText(prev => prev + t); },
      onDone: (d) => { setCitations(d.citations || []); },
      onError: (e) => { alert('Answer failed: ' + e); },
    });
    setLoading(false);
  }

  function highlight(cid: string) {
    setHighlighted(prev => (prev === cid ? null : cid));
    setTimeout(() => {
      document.getElementById('chunk-' + cid)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 30);
  }

  async function feedback(chunkId: string | null, v: 1 | -1) {
    if (!chunkId) return;
    const prev = rated[chunkId];
    // optimistic UI update
    setRated(r => ({ ...r, [chunkId]: v }));
    try {
      await api.feedback({
        event_id: 'ev_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7),
        user_id: userId,
        chunk_id: chunkId,
        metric: 'thumbs',
        value: v,
        variant,
        experiment_id: experimentId,
      });
      setToast(v === 1 ? 'Thanks — marked as helpful' : 'Thanks — marked as not helpful');
      setTimeout(() => setToast(''), 2200);
    } catch (e: any) {
      // revert on failure so the UI doesn't lie
      setRated(r => {
        const next = { ...r };
        if (prev === undefined) delete next[chunkId];
        else next[chunkId] = prev;
        return next;
      });
      alert('Could not record feedback: ' + e);
    }
  }

  function runActive() { view === 'answer' ? runAnswer() : runSearch(); }
  const showingAnswer = view === 'answer';

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-slate-900/80 backdrop-blur border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center gap-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm">C</div>
            <span className="font-semibold text-lg tracking-tight">Cairn</span>
          </Link>
          <nav className="flex gap-1 ml-4">
            {TABS.map(t => (
              <button key={t.id} onClick={() => setView(t.id)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${view === t.id ? ACTIVE[t.color] : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}`}>
                {t.label}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {variant && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 font-medium">{variant}</span>
            )}
            <span
              title={
                online === true ? 'API online'
                : warming ? 'Waiting for the API to wake up (free-tier cold start)…'
                : online === false ? `API unreachable: ${healthError}`
                : 'Checking API…'
              }
              className={`w-2 h-2 rounded-full ${
                online === true ? 'bg-green-500'
                : warming ? 'bg-amber-400 animate-pulse'
                : online === false ? 'bg-red-500'
                : 'bg-slate-400 animate-pulse'
              }`} />
            <input className="w-32 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 px-3 py-1 text-xs"
              value={userId} onChange={e => setUserId(e.target.value)} placeholder="user_id" />
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        <div className={`rounded-xl border border-slate-200 dark:border-slate-800 border-l-4 px-4 py-3 ${DESC_ACCENT[view]}`}>
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
              {TABS.find(t => t.id === view)?.label}
            </span>
            <span className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              {TAB_DESCRIPTIONS[view]}
            </span>
          </div>
        </div>

        {view === 'experiments' && <Experiments />}
        {view === 'eval' && <Eval />}

        {(view === 'search' || view === 'answer') && (
          <>
            <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-6">
              <label className="block text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">
                {showingAnswer ? 'Ask a question' : 'Search query'}
              </label>
              <textarea
                className="w-full min-h-[80px] p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 resize-y focus:ring-2 focus:ring-indigo-500 focus:outline-none text-base"
                value={query} onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runActive(); } }}
                placeholder={showingAnswer ? 'e.g. What is the Zrínyi battleship?' : 'e.g. battleship decommissioning…'}
              />
              <div className="flex gap-2 mt-4">
                <button disabled={loading || !query.trim()} onClick={runActive}
                  className="px-6 py-2.5 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
                  {loading ? 'Loading…' : showingAnswer ? 'Get Answer' : 'Search'}
                </button>
                {query && (
                  <button onClick={() => { setQuery(''); setResults([]); setAnswerText(''); setChunks([]); setCitations([]); }}
                    className="px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
                    Clear
                  </button>
                )}
              </div>
            </section>

            {/* Streaming answer + citations */}
            {showingAnswer && (answerText || chunks.length > 0) && (
              <section className="rounded-2xl border border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950 p-6">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <h3 className="font-semibold text-emerald-800 dark:text-emerald-200">Answer</h3>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 font-medium">
                    Demo mode · templated answer from retrieved passages
                  </span>
                </div>
                <p className="whitespace-pre-wrap leading-relaxed text-emerald-900 dark:text-emerald-100">{answerText}<span className="animate-pulse">▍</span></p>
                {citations.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {citations.map(c => (
                      <button key={c} onClick={() => highlight(c)}
                        className={`px-2 py-1 rounded-md text-xs font-mono transition-colors ${highlighted === c ? 'bg-amber-400 text-amber-950' : 'bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 hover:bg-amber-200'}`}>
                        {c}
                      </button>
                    ))}
                  </div>
                )}
              </section>
            )}

            {/* Results / source chunks */}
            <section className="space-y-3" ref={chunksRef}>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-700 dark:text-slate-300">
                  {showingAnswer ? 'Sources' : 'Results'}
                </h3>
                {(showingAnswer ? chunks : results).length > 0 && (
                  <span className="text-xs text-slate-400">{(showingAnswer ? chunks : results).length} chunks</span>
                )}
              </div>

              {loading && !showingAnswer && (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => <div key={i} className="h-24 rounded-2xl bg-slate-200 dark:bg-slate-800 animate-pulse" />)}
                </div>
              )}

              {!loading && (showingAnswer ? chunks : results).length === 0 && (
                <div className="text-center text-slate-400 dark:text-slate-500 py-12 border border-dashed border-slate-300 dark:border-slate-700 rounded-2xl">
                  <p className="text-lg mb-1">No results yet</p>
                  <p className="text-sm">{showingAnswer ? 'Ask a question to stream a cited answer' : 'Enter a query and press Search'}</p>
                </div>
              )}

              {(showingAnswer ? chunks : results).map((r, i) => (
                <article key={i} id={'chunk-' + r.chunk_id}
                  className={`rounded-2xl border p-5 transition-all ${highlighted === r.chunk_id
                    ? 'border-amber-400 ring-2 ring-amber-300 bg-amber-50 dark:bg-amber-950'
                    : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 hover:border-indigo-300'}`}>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-xs font-bold text-slate-400 w-6">#{r.rank}</span>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 font-mono">{r.chunk_id}</span>
                    <div className="ml-auto flex items-center gap-2">
                      <div className="text-xs font-semibold text-indigo-600 dark:text-indigo-400">{typeof r.score === 'number' ? r.score.toFixed(3) : ''}</div>
                      <div className="w-20 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${Math.min(100, (r.score || 0) * 100)}%` }} />
                      </div>
                      <button onClick={() => feedback(r.chunk_id, 1)} title="Helpful"
                        className={`p-1 rounded transition-colors ${rated[r.chunk_id] === 1 ? 'bg-green-200 dark:bg-green-800 ring-1 ring-green-400' : 'hover:bg-green-100 dark:hover:bg-green-900 opacity-60 hover:opacity-100'}`}>👍</button>
                      <button onClick={() => feedback(r.chunk_id, -1)} title="Not helpful"
                        className={`p-1 rounded transition-colors ${rated[r.chunk_id] === -1 ? 'bg-red-200 dark:bg-red-800 ring-1 ring-red-400' : 'hover:bg-red-100 dark:hover:bg-red-900 opacity-60 hover:opacity-100'}`}>👎</button>
                    </div>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{r.text}</p>
                </article>
              ))}
            </section>
          </>
        )}
      </main>

      <footer className="max-w-5xl mx-auto px-6 py-6 text-center text-xs text-slate-400 dark:text-slate-600">
        Cairn · from-scratch RAG search engine · pure Python + NumPy engine · free-tier deployable ·{' '}
        <Link to="/docs" className="text-indigo-500 hover:underline">Simulation docs</Link>
      </footer>

      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20 px-4 py-2 rounded-full bg-slate-900 text-white text-sm shadow-lg dark:bg-white dark:text-slate-900">
          {toast}
        </div>
      )}
    </div>
  );
}

export default Search;
