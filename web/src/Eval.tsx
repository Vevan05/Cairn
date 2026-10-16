import { useEffect, useState } from 'react';
import * as api from './api';

interface EvalRow {
  retrieval: string;
  n_queries: number;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
}

function Bar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-3 justify-end whitespace-nowrap">
      <div className="hidden sm:block w-20 h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${Math.min(100, value * 100)}%` }} />
      </div>
      <span className="tabular-nums text-sm w-14 text-right">{value.toFixed(3)}</span>
    </div>
  );
}

export function Eval() {
  const [sets, setSets] = useState<any[]>([]);
  const [selected, setSelected] = useState('');
  const [rows, setRows] = useState<EvalRow[]>([]);
  const [topK, setTopK] = useState(10);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [paste, setPaste] = useState('');
  const [pasteName, setPasteName] = useState('my_eval');

  async function loadSets(keepSelection = false) {
    try {
      const r = await api.listEvalSets();
      setSets(r.sets || []);
      if (!keepSelection && r.sets?.length && !selected) setSelected(r.sets[0].id);
    } catch { /* ignore */ }
  }

  useEffect(() => { loadSets(); }, []);

  async function seedSample() {
    setError('');
    try {
      const s = await api.seedSampleEvalSet();
      await loadSets();
      setSelected(s.id);
      setRows([]);
    } catch (e: any) { setError('Seeding failed: ' + e); }
  }

  async function addPasted() {
    setError('');
    try {
      const labeled = JSON.parse(paste);
      if (!Array.isArray(labeled)) throw new Error('expected a JSON array');
      const s = await api.createEvalSet({ id: pasteName.trim(), name: pasteName.trim(), labeled });
      setPaste('');
      await loadSets();
      setSelected(s.id);
      setRows([]);
    } catch (e: any) { setError('Add failed: ' + e.message); }
  }

  async function run() {
    if (!selected) return;
    setRunning(true); setError('');
    try {
      const r = await api.runEval({ set_id: selected, top_k: topK, compare: ['vector', 'hybrid'] });
      setRows(r.results || []);
    } catch (e: any) { setError('Eval failed: ' + e); }
    finally { setRunning(false); }
  }

  const bestBy = (key: keyof EvalRow) => {
    if (!rows.length) return '';
    return rows.reduce((a, b) => (Number(a[key]) >= Number(b[key]) ? a : b)).retrieval;
  };

  return (
    <div className="space-y-6">
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-6 space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="font-semibold text-lg">Offline Evaluation</h2>
          <span className="text-xs text-slate-400">score a labeled set with recall@k · MRR · nDCG@k</span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select className="rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 px-3 py-1.5 text-sm"
            value={selected} onChange={e => setSelected(e.target.value)}>
            {sets.length === 0 && <option value="">no sets yet</option>}
            {sets.map(s => <option key={s.id} value={s.id}>{s.name} ({s.n})</option>)}
          </select>
          <label className="text-sm text-slate-500">k =</label>
          <input type="number" min={1} max={50} value={topK} onChange={e => setTopK(Number(e.target.value))}
            className="w-16 rounded-lg border px-2 py-1.5 text-sm dark:bg-slate-800 dark:border-slate-600" />
          <button onClick={run} disabled={running || !selected}
            className="px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm hover:bg-indigo-700 disabled:opacity-40">
            {running ? 'Running…' : 'Run eval'}
          </button>
          <button onClick={seedSample}
            className="px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 text-sm hover:bg-slate-50 dark:hover:bg-slate-800">
            Seed sample set
          </button>
        </div>

        <details className="text-sm">
          <summary className="cursor-pointer text-slate-500">Upload / paste a labeled set</summary>
          <div className="mt-3 space-y-2">
            <input className="rounded-lg border px-3 py-2 text-sm w-48 dark:bg-slate-800 dark:border-slate-600"
              value={pasteName} onChange={e => setPasteName(e.target.value)} placeholder="set id" />
            <textarea className="w-full min-h-[100px] p-3 rounded-lg border font-mono text-xs dark:bg-slate-800 dark:border-slate-600"
              placeholder='[{"query":"...","relevant_chunk_ids":["doc3::chunk0"]}]'
              value={paste} onChange={e => setPaste(e.target.value)} />
            <button onClick={addPasted} disabled={!paste.trim() || !pasteName.trim()}
              className="px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm disabled:opacity-40">Add set</button>
          </div>
        </details>
      </section>

      {error && <div className="text-red-600 text-sm">{error}</div>}

      {rows.length > 0 && (
        <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500">
              <tr>
                <th className="text-left px-4 py-3 font-medium whitespace-nowrap">Retrieval</th>
                <th className="text-right px-4 py-3 font-medium whitespace-nowrap">Queries</th>
                <th className="text-right px-4 py-3 font-medium whitespace-nowrap">Recall@{topK}</th>
                <th className="text-right px-4 py-3 font-medium whitespace-nowrap">MRR</th>
                <th className="text-right px-4 py-3 font-medium whitespace-nowrap">nDCG@{topK}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.retrieval} className="border-t border-slate-100 dark:border-slate-800">
                  <td className="px-4 py-3 font-medium whitespace-nowrap">
                    <span className="inline-flex items-center gap-2">
                      {r.retrieval}
                      {r.retrieval === bestBy('ndcg_at_k') && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-200">best</span>
                      )}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">{r.n_queries}</td>
                  <td className="px-6 py-3"><Bar value={r.recall_at_k} /></td>
                  <td className="px-6 py-3"><Bar value={r.mrr} /></td>
                  <td className="px-6 py-3"><Bar value={r.ndcg_at_k} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <p className="text-xs text-slate-400">
        A labeled set maps each query to the chunk id(s) that should be retrieved. Higher is better.
        The seeded sample is corpus-derived (query = a real chunk opening), so it demonstrates the harness
        rather than measuring true relevance.
      </p>
    </div>
  );
}

export default Eval;
