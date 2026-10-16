import { Link } from 'react-router-dom';

const FEATURES = [
  {
    icon: '🔍',
    title: 'Search',
    desc: 'Find the closest passages in the corpus, ranked by cosine similarity. No LLM — just fast, from-scratch retrieval.',
    color: 'indigo',
  },
  {
    icon: '💬',
    title: 'Answer',
    desc: 'Retrieve the best passages, then generate a plain-language answer with citations back to the source chunks.',
    color: 'emerald',
  },
  {
    icon: '🧪',
    title: 'Experiments',
    desc: 'A/B test retrieval configs. Deterministic bucketing, z-tests, and Sample Ratio Mismatch checks — all built in.',
    color: 'purple',
  },
];

const STACK = [
  ['Engine', 'HNSW · BM25 · RRF · int8 quantization'],
  ['Experiments', 'SHA-256 bucketing · z-test · Welch t · χ² SRM'],
  ['Serving', 'FastAPI · React · Postgres'],
  ['Deploy', 'Render · Vercel · Neon (free tier)'],
];

export function Landing() {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      {/* Nav */}
      <header className="max-w-6xl mx-auto px-6 h-16 flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm">C</div>
        <span className="font-semibold text-lg tracking-tight">Cairn</span>
        <nav className="ml-auto flex items-center gap-4 text-sm">
          <a href="#features" className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Features</a>
          <a href="#stack" className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Stack</a>
          <Link to="/app" className="px-4 py-2 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 transition-colors">
            Launch app
          </Link>
        </nav>
      </header>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-16 pb-24 text-center">
        <div className="inline-flex items-center gap-2 text-xs font-medium px-3 py-1 rounded-full bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 mb-6">
          <span className="w-2 h-2 rounded-full bg-indigo-500" />
          From-scratch vector index + built-in A/B testing
        </div>
        <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-tight">
          RAG search that
          <br />
          <span className="bg-gradient-to-r from-indigo-600 to-emerald-500 bg-clip-text text-transparent">
            measures itself
          </span>
        </h1>
        <p className="mt-6 max-w-2xl mx-auto text-lg text-slate-500 dark:text-slate-400">
          A retrieval-augmented search engine with a hand-built HNSW + BM25 hybrid index
          and an experimentation layer that tells you, with statistics, which configuration actually wins.
        </p>
        <div className="mt-8 flex items-center justify-center gap-3">
          <Link to="/app" className="px-6 py-3 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 transition-colors">
            Open the app →
          </Link>
          <a href="#how" className="px-6 py-3 rounded-xl border border-slate-300 dark:border-slate-700 font-medium text-slate-600 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-900 transition-colors">
            How it works
          </a>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="max-w-6xl mx-auto px-6 pb-24">
        <h2 className="text-2xl font-bold mb-8 text-center">Three things you can do</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {FEATURES.map(f => (
            <div key={f.title} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-3xl mb-4">{f.icon}</div>
              <h3 className="font-semibold text-lg mb-2">{f.title}</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="bg-white dark:bg-slate-900 border-y border-slate-200 dark:border-slate-800">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <h2 className="text-2xl font-bold mb-10 text-center">How it works</h2>
          <div className="grid md:grid-cols-4 gap-6 text-center">
            {[
              ['1', 'Offline', 'Documents are chunked, embedded (384-dim), and compiled into a read-only index.'],
              ['2', 'Retrieve', 'A query runs through HNSW + BM25, fused with Reciprocal Rank Fusion.'],
              ['3', 'Assign', 'Each user is deterministically bucketed into a variant via a stable hash.'],
              ['4', 'Measure', 'Outcomes are logged; z-tests and SRM checks decide which variant wins.'],
            ].map(([n, title, desc]) => (
              <div key={n}>
                <div className="w-10 h-10 mx-auto rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold mb-4">{n}</div>
                <h3 className="font-semibold mb-2">{title}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stack */}
      <section id="stack" className="max-w-6xl mx-auto px-6 py-20">
        <h2 className="text-2xl font-bold mb-8 text-center">Built from scratch</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {STACK.map(([label, desc]) => (
            <div key={label} className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
              <div className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wide mb-2">{label}</div>
              <div className="text-sm text-slate-600 dark:text-slate-300">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800">
        <div className="max-w-6xl mx-auto px-6 py-10 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-slate-400 dark:text-slate-600">
          <div>Cairn · from-scratch RAG search engine · free-tier deployable</div>
          <Link to="/app" className="text-indigo-600 hover:underline">Open the app →</Link>
        </div>
      </footer>
    </div>
  );
}

export default Landing;
