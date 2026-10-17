import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import * as api from './api';

interface VariantStat {
  variant: string;
  exposures: number;
  n: number;
  successes: number;
  thumbs_up_rate: number | null;
  p95_latency_ms: number | null;
}
type Variant = { name: string; weight: number; config: { retrieval?: string; topK?: number } };

const DEFAULT_VARIANTS: Variant[] = [
  { name: 'control', weight: 50, config: { retrieval: 'vector', topK: 5 } },
  { name: 'treatment', weight: 50, config: { retrieval: 'hybrid', topK: 5 } },
];

const STATUS_COLORS: Record<string, string> = {
  'treatment better': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200',
  'control better': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  'no detectable difference': 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  'not enough data': 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
  'not enough data (provisional)': 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
  'SRM warning — experiment untrustworthy': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
};
const statusClass = (v: string) => STATUS_COLORS[v] ?? 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300';

export function Experiments() {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [activeId, setActiveId] = useState('');
  const [selected, setSelected] = useState('exp_hybrid_vs_vector');
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [showAdmin, setShowAdmin] = useState(false);
  const [simUsers, setSimUsers] = useState(800);
  const [simLift, setSimLift] = useState(0.05);

  const [newId, setNewId] = useState('');
  const [newName, setNewName] = useState('');
  const [variants, setVariants] = useState<Variant[]>(DEFAULT_VARIANTS);

  async function loadExperiments() {
    const r = await api.listExperiments();
    setExperiments(r.experiments);
    setActiveId(r.active ?? '');
  }
  async function loadResults(id: string) {
    setError('');
    try { setResults(await api.getResults(id)); }
    catch (e: any) { setError('Could not load results: ' + e); }
  }

  useEffect(() => { loadExperiments(); }, []);
  useEffect(() => { loadResults(selected); }, [selected]);

  async function createExperiment() {
    setError(''); setNotice('');
    const id = newId.trim();
    try {
      await api.createExperiment({ id, name: newName.trim() || id, status: 'draft', config_json: { variants } });
      setNewId(''); setNewName(''); setVariants(DEFAULT_VARIANTS);
      await loadExperiments();
      setSelected(id);
      setNotice(`Created "${id}" as a draft. Press Activate to start serving traffic to it.`);
    } catch (e: any) { setError('Create failed: ' + e); }
  }

  async function activate(id: string) {
    setError(''); setNotice('');
    try {
      await api.activateExperiment(id);
      await loadExperiments();
      await loadResults(id);
      setNotice(`"${id}" is now serving all /search traffic.`);
    } catch (e: any) { setError('Activate failed: ' + e); }
  }

  /** Split 100 evenly across the given variants (used when adding/removing). */
  function rebalanceEven(vs: Variant[]): Variant[] {
    const n = vs.length || 1;
    const base = Math.floor(100 / n);
    const remainder = 100 - base * n;
    return vs.map((v, j) => ({ ...v, weight: Math.max(1, base + (j < remainder ? 1 : 0)) }));
  }

  /** Set one variant's weight and auto-balance the rest so all sum to 100. */  function setWeight(index: number, raw: number) {
    const value = Math.max(1, Math.min(99, Math.round(Number.isFinite(raw) ? raw : 0)));
    setVariants(vs => {
      const next = vs.map(v => ({ ...v }));
      next[index].weight = value;
      const others = vs.map((_, j) => j).filter(j => j !== index);
      if (others.length === 0) return next;
      const remaining = 100 - value;
      if (others.length === 1) {
        next[others[0]].weight = remaining;
      } else {
        // distribute the remainder across the others, preserving their ratio
        const otherSum = others.reduce((s, j) => s + (Number(vs[j].weight) || 0), 0);
        let assigned = 0;
        others.forEach((j, k) => {
          if (k === others.length - 1) {
            next[j].weight = Math.max(1, remaining - assigned);
          } else {
            const w = otherSum > 0
              ? Math.round((Number(vs[j].weight) / otherSum) * remaining)
              : Math.round(remaining / others.length);
            next[j].weight = w;
            assigned += w;
          }
        });
      }
      return next;
    });
  }

  async function pause() {
    setError(''); setNotice('');
    try {
      await api.updateExperiment(selected, { status: 'paused' });
      await loadExperiments();
      await loadResults(selected);
    } catch (e: any) { setError('Update failed: ' + e); }
  }

  async function simulate() {
    setError(''); setNotice('');
    try {
      const r = await api.simulateExperiment(selected, { n_users: simUsers, treatment_lift: simLift });
      await loadExperiments();
      await loadResults(selected);
      setNotice(`Simulated ${r.simulated_users} users of traffic into "${selected}".`);
    } catch (e: any) { setError('Simulate failed: ' + e); }
  }

  async function deleteExperiment() {
    if (!confirm(`Delete experiment "${selected}"?`)) return;
    setError(''); setNotice('');
    try {
      await api.deleteExperiment(selected);
      const r = await api.listExperiments();
      setExperiments(r.experiments); setActiveId(r.active ?? '');
      const next = r.experiments[0]?.id ?? 'exp_hybrid_vs_vector';
      setSelected(next);
      await loadResults(next);
    } catch (e: any) { setError('Delete failed: ' + e); }
  }

  const totalWeight = variants.reduce((s, v) => s + (Number(v.weight) || 0), 0);
  const isActive = selected === activeId;
  const hasTraffic = results?.variants?.some((v: VariantStat) => v.exposures > 0);

  return (
    <div className="space-y-6">
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-6">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="font-semibold text-lg">Experiment Admin</h2>
          {isActive
            ? <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-medium">● serving traffic</span>
            : <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">inactive</span>}
          <button onClick={() => setShowAdmin(s => !s)}
            className="ml-auto text-sm px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-800">
            {showAdmin ? 'Close' : '＋ New experiment'}
          </button>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <select className="rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 px-3 py-1.5 text-sm"
            value={selected} onChange={e => setSelected(e.target.value)}>
            {experiments.map(e => (
              <option key={e.id} value={e.id}>
                {e.id === activeId ? '● ' : ''}{e.name || e.id} ({e.status})
              </option>
            ))}
          </select>
          <button onClick={() => loadResults(selected)}
            className="text-sm px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700">Refresh</button>
          {!isActive && (
            <button onClick={() => activate(selected)}
              className="text-sm px-3 py-1.5 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700">Activate</button>
          )}
          {isActive && (
            <button onClick={pause} className="text-sm px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600">Pause</button>
          )}
          <button onClick={deleteExperiment}
            className="text-sm px-3 py-1.5 rounded-lg border border-red-300 text-red-600 hover:bg-red-50 dark:hover:bg-red-950">Delete</button>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <span className="text-slate-500">Simulate traffic:</span>
          <input type="number" min={10} max={20000} value={simUsers} onChange={e => setSimUsers(Number(e.target.value))}
            className="w-24 rounded-lg border px-2 py-1.5 text-sm dark:bg-slate-800 dark:border-slate-600" />
          <span className="text-slate-500">users, lift</span>
          <input type="number" min={0} max={1} step={0.01} value={simLift} onChange={e => setSimLift(Number(e.target.value))}
            className="w-20 rounded-lg border px-2 py-1.5 text-sm dark:bg-slate-800 dark:border-slate-600" />
          <button onClick={simulate}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-800">
            Run simulation
          </button>
          <span className="text-xs text-slate-400">fills this experiment with synthetic exposures + thumbs</span>
          <Link to="/docs" className="text-xs text-indigo-500 hover:underline">how it works →</Link>
        </div>

        {notice && <div className="mt-3 text-sm text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950 rounded-lg px-3 py-2">{notice}</div>}

        {showAdmin && (
          <div className="mt-5 border-t border-slate-100 dark:border-slate-800 pt-5 space-y-4">
            <div className="grid sm:grid-cols-2 gap-3">
              <input className="rounded-lg border px-3 py-2 text-sm dark:bg-slate-800 dark:border-slate-600"
                placeholder="experiment id (e.g. exp_rerank)" value={newId} onChange={e => setNewId(e.target.value)} />
              <input className="rounded-lg border px-3 py-2 text-sm dark:bg-slate-800 dark:border-slate-600"
                placeholder="display name" value={newName} onChange={e => setNewName(e.target.value)} />
            </div>
            <div className="space-y-2">
              {variants.map((v, i) => (
                <div key={i} className="flex flex-wrap items-center gap-2">
                  <input className="rounded-lg border px-2 py-1.5 text-sm w-32 dark:bg-slate-800 dark:border-slate-600"
                    value={v.name} onChange={e => setVariants(vs => vs.map((x, j) => j === i ? { ...x, name: e.target.value } : x))} />
                  <input className="rounded-lg border px-2 py-1.5 text-sm w-20 dark:bg-slate-800 dark:border-slate-600"
                    type="number" min={1} max={99} value={v.weight}
                    onChange={e => setWeight(i, Number(e.target.value))} />
                  <select className="rounded-lg border px-2 py-1.5 text-sm dark:bg-slate-800 dark:border-slate-600"
                    value={v.config.retrieval} onChange={e => setVariants(vs => vs.map((x, j) => j === i ? { ...x, config: { ...x.config, retrieval: e.target.value } } : x))}>
                    <option value="vector">vector</option>
                    <option value="hybrid">hybrid</option>
                  </select>
                  {variants.length > 2 && (
                    <button onClick={() => setVariants(vs => rebalanceEven(vs.filter((_, j) => j !== i)))} className="text-xs text-red-500 px-2">remove</button>
                  )}
                </div>
              ))}
              <button onClick={() => setVariants(vs => rebalanceEven([...vs, { name: `variant_${vs.length + 1}`, weight: 1, config: { retrieval: 'hybrid', topK: 5 } }]))}
                className="text-sm text-indigo-600 hover:underline">＋ add variant</button>
            </div>
            <div className="flex items-center gap-3">
              <span className={`text-xs ${totalWeight === 100 ? 'text-emerald-600' : 'text-amber-600'}`}>
                weights sum: {totalWeight} (split = {variants.map(v => `${Math.round(v.weight / (totalWeight || 1) * 100)}%`).join(' / ')})
              </span>
              <button onClick={createExperiment} disabled={!newId.trim()}
                className="ml-auto px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm hover:bg-indigo-700 disabled:opacity-40">
                Create experiment
              </button>
            </div>
          </div>
        )}
      </section>

      {error && <div className="text-red-600 text-sm">{error}</div>}

      {results && results.demo_seeded && (
        <section className="rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 p-4 text-sm text-amber-800 dark:text-amber-200">
          <span className="font-semibold">Simulated demo data.</span> This instance has no persistence, so the
          experiment is seeded with synthetic traffic at startup to keep the dashboard meaningful.
          Use <span className="font-medium">Simulate traffic</span> below, or run real searches, to add your own.
        </section>
      )}

      {results && (
        <>
          <section className={`rounded-2xl p-5 flex items-center gap-3 ${statusClass(results.verdict)}`}>
            <span className="text-2xl">🧪</span>
            <div>
              <div className="font-semibold">{results.verdict}</div>
              <div className="text-xs opacity-80">
                primary metric: {results.primary_metric} · status: {results.status}
                {isActive ? ' · serving traffic' : ' · not serving (activate to collect data)'}
              </div>
            </div>
          </section>

          {!hasTraffic && (
            <section className="rounded-2xl border border-dashed border-slate-300 dark:border-slate-700 p-6 text-center text-sm text-slate-500">
              This experiment has no traffic yet.
              {!isActive && (
                <button onClick={() => activate(selected)}
                  className="ml-2 px-3 py-1.5 rounded-lg bg-emerald-600 text-white text-xs hover:bg-emerald-700">
                  Activate it
                </button>
              )}
              {isActive && <span className="ml-1">Run some searches on the Search tab to generate exposures.</span>}
            </section>
          )}

          <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500">
                <tr>
                  <th className="text-left px-4 py-3 font-medium whitespace-nowrap">Variant</th>
                  <th className="text-right px-4 py-3 font-medium whitespace-nowrap">Exposures</th>
                  <th className="text-right px-4 py-3 font-medium whitespace-nowrap">Outcomes</th>
                  <th className="text-right px-4 py-3 font-medium whitespace-nowrap">Thumbs-up rate</th>
                  <th className="text-right px-4 py-3 font-medium whitespace-nowrap">p95 latency</th>
                </tr>
              </thead>
              <tbody>
                {results.variants.map((v: VariantStat) => (
                  <tr key={v.variant} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="px-4 py-3 font-medium">{v.variant}</td>
                    <td className="px-4 py-3 text-right tabular-nums">{v.exposures}</td>
                    <td className="px-4 py-3 text-right tabular-nums">{v.n}</td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      {v.thumbs_up_rate === null ? '—' : (v.thumbs_up_rate * 100).toFixed(1) + '%'}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      {v.p95_latency_ms === null ? '—' : v.p95_latency_ms + ' ms'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <div className="grid md:grid-cols-2 gap-4">
            <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5">
              <h3 className="font-semibold mb-3">Statistical test</h3>
              {results.test ? (
                <dl className="space-y-2 text-sm">
                  <div className="flex justify-between"><dt className="text-slate-500">z-statistic</dt><dd className="tabular-nums">{results.test.z}</dd></div>
                  <div className="flex justify-between"><dt className="text-slate-500">p-value</dt><dd className="tabular-nums">{results.test.p_value}</dd></div>
                  <div className="flex justify-between"><dt className="text-slate-500">absolute lift</dt><dd className="tabular-nums">{(results.test.absolute_lift * 100).toFixed(2)} pp</dd></div>
                  {results.test.relative_lift_pct !== undefined && (
                    <div className="flex justify-between"><dt className="text-slate-500">relative lift</dt><dd className="tabular-nums">{results.test.relative_lift_pct}%</dd></div>
                  )}
                  <div className="flex justify-between"><dt className="text-slate-500">significant (α=0.05)</dt>
                    <dd className={results.test.significant ? 'text-emerald-600 font-semibold' : 'text-slate-400'}>{String(results.test.significant)}</dd>
                  </div>
                </dl>
              ) : <p className="text-sm text-slate-400">No outcomes logged yet.</p>}
            </section>

            <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5">
              <h3 className="font-semibold mb-3">Sample Ratio Mismatch</h3>
              {results.srm ? (
                <dl className="space-y-2 text-sm">
                  <div className="flex justify-between"><dt className="text-slate-500">chi-square</dt><dd className="tabular-nums">{results.srm.chi2}</dd></div>
                  <div className="flex justify-between"><dt className="text-slate-500">p-value</dt><dd className="tabular-nums">{results.srm.p_value}</dd></div>
                  <div className="flex justify-between"><dt className="text-slate-500">flag</dt>
                    <dd className={results.srm.srm_detected ? 'text-red-600 font-semibold' : 'text-emerald-600'}>{results.srm.flag}</dd>
                  </div>
                  <p className="text-xs text-slate-400 pt-2">SRM catches bucketing/logging bugs. A flagged experiment is untrustworthy.</p>
                </dl>
              ) : <p className="text-sm text-slate-400">Not enough traffic to check.</p>}
            </section>
          </div>
        </>
      )}
    </div>
  );
}

export default Experiments;
