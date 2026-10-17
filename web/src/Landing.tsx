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
    desc: 'Stream a grounded answer token-by-token, with citations you can click to highlight the source passage.',
    color: 'emerald',
  },
  {
    icon: '🧪',
    title: 'Experiments',
    desc: 'Create and manage A/B tests. Deterministic bucketing, z-tests, and Sample Ratio Mismatch checks — all built in.',
    color: 'purple',
  },
  {
    icon: '📊',
    title: 'Eval',
    desc: 'Score any retrieval config offline on a labeled set: recall@k, MRR and nDCG@k for vector vs hybrid.',
    color: 'sky',
  },
];

const STACK = [
  ['Engine', 'HNSW · BM25 · RRF · int8 quantization'],
  ['Experiments', 'SHA-256 bucketing · z-test · Welch t · χ² SRM'],
  ['Serving', 'FastAPI · React · Postgres'],
  ['Deploy', 'Render · Vercel · Neon (free tier)'],
];

const STEPS = [
  ['1', 'Offline', 'Documents are chunked, embedded (384-dim), and compiled into a read-only index.'],
  ['2', 'Retrieve', 'A query runs through HNSW + BM25, fused with Reciprocal Rank Fusion.'],
  ['3', 'Assign', 'Each user is deterministically bucketed into a variant via a stable hash.'],
  ['4', 'Measure', 'Outcomes are logged; z-tests and SRM checks decide which variant wins.'],
];

export function Landing() {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      {/* Nav */}
      <header className="max-w-7xl mx-auto px-8 h-24 flex items-center gap-4">
        <div className="w-12 h-12 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold text-xl">C</div>
        <span className="font-semibold text-2xl tracking-tight">Cairn</span>
        <nav className="ml-auto flex items-center gap-8 text-lg">
          <a href="#features" className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Features</a>
          <a href="#stack" className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Stack</a>
          <Link to="/docs" className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200">Docs</Link>
          <Link to="/app" className="px-7 py-3 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 transition-colors">
            Launch app
          </Link>
        </nav>
      </header>

      {/* Hero */}
      <section className="max-w-7xl mx-auto px-8 pt-28 pb-40 text-center">
        <div className="inline-flex items-center gap-2 text-base font-medium px-5 py-2 rounded-full bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 mb-10">
          <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" />
          From-scratch vector index + built-in A/B testing
        </div>
        <h1 className="text-6xl md:text-8xl font-bold tracking-tight leading-[1.05]">
          RAG search that
          <br />
          <span className="bg-gradient-to-r from-indigo-600 to-emerald-500 bg-clip-text text-transparent">
            measures itself
          </span>
        </h1>
        <p className="mt-10 max-w-3xl mx-auto text-xl md:text-2xl text-slate-500 dark:text-slate-400 leading-relaxed">
          A retrieval-augmented search engine with a hand-built HNSW + BM25 hybrid index
          and an experimentation layer that tells you, with statistics, which configuration actually wins.
        </p>
        <div className="mt-12 flex items-center justify-center gap-4">
          <Link to="/app" className="px-9 py-4 rounded-xl bg-indigo-600 text-white font-medium text-lg hover:bg-indigo-700 transition-colors">
            Open the app →
          </Link>
          <a href="#how" className="px-9 py-4 rounded-xl border border-slate-300 dark:border-slate-700 font-medium text-lg text-slate-600 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-900 transition-colors">
            How it works
          </a>
        </div>
        <p className="mt-8 text-sm text-slate-400 dark:text-slate-500">
          Runs on a free tier, so the first load after a period of inactivity can take up to a minute while the API wakes up.
        </p>
      </section>

      {/* Features */}
      <section id="features" className="max-w-7xl mx-auto px-8 pb-40">
        <h2 className="text-4xl md:text-5xl font-bold mb-16 text-center tracking-tight">Four things you can do</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {FEATURES.map(f => (
            <div key={f.title} className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-9 shadow-sm hover:shadow-lg transition-shadow">
              <div className="text-5xl mb-6">{f.icon}</div>
              <h3 className="font-semibold text-2xl mb-3">{f.title}</h3>
              <p className="text-base text-slate-500 dark:text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="bg-white dark:bg-slate-900 border-y border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-8 py-36">
          <h2 className="text-4xl md:text-5xl font-bold mb-20 text-center tracking-tight">How it works</h2>
          <div className="grid md:grid-cols-4 gap-10 text-center">
            {STEPS.map(([n, title, desc]) => (
              <div key={n}>
                <div className="w-20 h-20 mx-auto rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-3xl mb-8">{n}</div>
                <h3 className="font-semibold text-2xl mb-3">{title}</h3>
                <p className="text-base text-slate-500 dark:text-slate-400 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stack */}
      <section id="stack" className="max-w-7xl mx-auto px-8 py-36">
        <h2 className="text-4xl md:text-5xl font-bold mb-16 text-center tracking-tight">Built from scratch</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {STACK.map(([label, desc]) => (
            <div key={label} className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-9">
              <div className="text-sm font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider mb-4">{label}</div>
              <div className="text-lg text-slate-600 dark:text-slate-300 leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-8 py-16 flex flex-col sm:flex-row items-center justify-between gap-6 text-lg text-slate-400 dark:text-slate-600">
          <div>Cairn · from-scratch RAG search engine · free-tier deployable</div>
          <Link to="/app" className="text-indigo-600 hover:underline text-lg">Open the app →</Link>
        </div>
      </footer>
    </div>
  );
}

export default Landing;
