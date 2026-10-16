import { useEffect, useState } from 'react';

interface VariantStat {
  variant: string;
  exposures: number;
  n: number;
  successes: number;
  thumbs_up_rate: number | null;
  p95_latency_ms: number | null;
}
interface ExperimentResults {
  experiment_id: string;
  status: string;
  primary_metric: string | null;
  variants: VariantStat[];
  test: null | {
    z: number; p_value: number; significant: boolean;
    a: string; b: string; absolute_lift: number; relative_lift_pct?: number;
  };
  srm: null | { chi2: number; p_value: number; srm_detected: boolean; flag: string };
  verdict: string;
}

const STATUS_COLORS: Record<string, string> = {
  'treatment better': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200',
  'control better': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  'no detectable difference': 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  'not enough data': 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
  'not enough data (provisional)': 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
  'SRM warning — experiment untrustworthy': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
};
function statusClass(v: string) {
  return STATUS_COLORS[v] ?? 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300';
}

export function Experiments() {
  const [results, setResults] = useState<ExperimentResults | null>(null);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selected, setSelected] = useState('exp_hybrid_vs_vector');
  const [error, setError] = useState('');

  async function loadExperiments() {
    try {
      const r = await fetch('/experiments');
      const j = await r.json();
      setExperiments(j.experiments || []);
    } catch { /* ignore */ }
  }
  async function loadResults(id: string) {
    setError('');
    try {
      const r = await fetch(`/experiments/${id}/results`);
      if (!r.ok) throw new Error(`${r.status}`);
      setResults(await r.json());
    } catch (e: any) { setError('Could not load results: ' + e); }
  }

  useEffect(() => { loadExperiments(); }, []);
  useEffect(() => { loadResults(selected); }, [selected]);

  return (
    <div className="space-y-6">
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-6">
        <div className="flex items-center gap-4">
          <h2 className="font-semibold text-lg">Experiments</h2>
          <select
            className="ml-auto rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 px-3 py-1.5 text-sm"
            value={selected} onChange={e => setSelected(e.target.value)}
          >
            {experiments.length === 0 && <option>{selected}</option>}
            {experiments.map(e => <option key={e.id} value={e.id}>{e.name || e.id}</option>)}
          </select>
          <button onClick={() => loadResults(selected)}
            className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-sm hover:bg-indigo-700">Refresh</button>
        </div>
      </section>

      {error && <div className="text-red-600 text-sm">{error}</div>}

      {results && (
        <>
          {/* Verdict banner */}
          <section className={`rounded-2xl p-5 flex items-center gap-3 ${statusClass(results.verdict)}`}>
            <span className="text-2xl">🧪</span>
            <div>
              <div className="font-semibold">{results.verdict}</div>
              <div className="text-xs opacity-80">
                primary metric: {results.primary_metric} · status: {results.status}
              </div>
            </div>
          </section>

          {/* Variant table */}
          <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-800 text-slate-500">
                <tr>
                  <th className="text-left px-4 py-3 font-medium">Variant</th>
                  <th className="text-right px-4 py-3 font-medium">Exposures</th>
                  <th className="text-right px-4 py-3 font-medium">Outcomes</th>
                  <th className="text-right px-4 py-3 font-medium">Thumbs-up rate</th>
                  <th className="text-right px-4 py-3 font-medium">p95 latency</th>
                </tr>
              </thead>
              <tbody>
                {results.variants.map(v => (
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

          {/* Stats cards */}
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
                    <dd className={results.test.significant ? 'text-emerald-600 font-semibold' : 'text-slate-400'}>
                      {String(results.test.significant)}
                    </dd>
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
                  <p className="text-xs text-slate-400 pt-2">
                    SRM catches bucketing/logging bugs. A flagged experiment is untrustworthy.
                  </p>
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
