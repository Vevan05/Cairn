import { Link } from 'react-router-dom';

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-6 space-y-3">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="text-sm text-slate-600 dark:text-slate-300 space-y-3 leading-relaxed">{children}</div>
    </section>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <pre className="bg-slate-900 dark:bg-slate-950 text-slate-100 rounded-xl p-4 overflow-x-auto text-xs leading-relaxed">
      <code>{children}</code>
    </pre>
  );
}

export function Docs() {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-slate-900/80 backdrop-blur border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-3xl mx-auto px-6 h-14 flex items-center gap-3">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm">C</div>
            <span className="font-semibold text-lg tracking-tight">Cairn</span>
          </Link>
          <span className="text-slate-400">/</span>
          <span className="font-medium">Docs</span>
          <Link to="/app" className="ml-auto text-sm px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700">
            Open app
          </Link>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10 space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Simulating an experiment</h1>
          <p className="mt-2 text-slate-500 dark:text-slate-400">
            How to fill an A/B test with synthetic traffic so the statistics and Sample Ratio
            Mismatch checks actually run — no real users required.
          </p>
        </div>

        <Section title="Why simulate?">
          <p>
            A brand-new experiment starts empty, and live traffic is a trickle: it can take thousands of
            users to reach statistical significance. An A/B test also needs both arms populated — with only
            one visitor, all traffic lands in a single variant and the z-test can't run.
          </p>
          <p>
            Simulation generates many distinct users, assigns each to a variant with the real bucketing
            function, and logs exposures and thumbs with a configurable lift. Everything flows through the
            <span className="font-medium"> same code paths</span> as live traffic, so the results, p-values,
            and SRM check are computed identically.
          </p>
        </Section>

        <Section title="Step by step (UI)">
          <ol className="list-decimal list-inside space-y-2">
            <li>Open the app and go to the <span className="font-medium">Experiments</span> tab.</li>
            <li>Create an experiment (e.g. <code className="text-xs bg-slate-100 dark:bg-slate-800 px-1 rounded">control 50 / treatment 50</code>).</li>
            <li>Click <span className="font-medium">Activate</span> so it serves <code className="text-xs bg-slate-100 dark:bg-slate-800 px-1 rounded">/search</code> traffic.</li>
            <li>In the <span className="font-medium">Simulate traffic</span> row, set <span className="font-medium">users</span> and <span className="font-medium">lift</span>.</li>
            <li>Click <span className="font-medium">Run simulation</span>, then watch the variant table, statistical test, and SRM cards populate.</li>
          </ol>
        </Section>

        <Section title="The inputs">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-slate-500">
                <tr>
                  <th className="text-left py-2 font-medium">Field</th>
                  <th className="text-left py-2 font-medium">Meaning</th>
                  <th className="text-left py-2 font-medium">Typical</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                <tr><td className="py-2 pr-3 font-mono text-xs">users</td><td className="py-2 pr-3">Number of synthetic users to generate</td><td className="py-2">500–2000</td></tr>
                <tr><td className="py-2 pr-3 font-mono text-xs">lift</td><td className="py-2 pr-3">Absolute boost to the thumbs-up rate for non-control variants</td><td className="py-2">0.05–0.15</td></tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-400">
            The first variant is the control; every other variant gets <code>control_rate + lift</code>.
            Base control rate is 10%.
          </p>
        </Section>

        <Section title="Scenario cheat-sheet">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-slate-500">
                <tr>
                  <th className="text-left py-2 font-medium">Goal</th>
                  <th className="text-left py-2 font-medium">users</th>
                  <th className="text-left py-2 font-medium">lift</th>
                  <th className="text-left py-2 font-medium">Expected verdict</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                <tr><td className="py-2 pr-3">Clear winner</td><td className="py-2 pr-3">2000</td><td className="py-2 pr-3">0.15</td><td className="py-2">treatment better</td></tr>
                <tr><td className="py-2 pr-3">Modest effect</td><td className="py-2 pr-3">1000</td><td className="py-2 pr-3">0.08</td><td className="py-2">treatment better</td></tr>
                <tr><td className="py-2 pr-3">A/A (no real difference)</td><td className="py-2 pr-3">1500</td><td className="py-2 pr-3">0.00</td><td className="py-2">no detectable difference</td></tr>
                <tr><td className="py-2 pr-3">Underpowered</td><td className="py-2 pr-3">60</td><td className="py-2 pr-3">0.10</td><td className="py-2">not enough data (provisional)</td></tr>
                <tr><td className="py-2 pr-3">Effect too small to detect</td><td className="py-2 pr-3">800</td><td className="py-2 pr-3">0.01</td><td className="py-2">no detectable difference</td></tr>
              </tbody>
            </table>
          </div>
        </Section>

        <Section title="Reading the results">
          <ul className="list-disc list-inside space-y-2">
            <li><span className="font-medium">Thumbs-up rate</span> — successes ÷ outcomes, per variant. The primary metric.</li>
            <li><span className="font-medium">absolute lift</span> — treatment rate minus control rate, in percentage points.</li>
            <li><span className="font-medium">z-statistic / p-value</span> — a two-proportion z-test. <code>p &lt; 0.05</code> means the difference is unlikely to be chance.</li>
            <li><span className="font-medium">significant</span> — whether p &lt; 0.05 (α = 0.05).</li>
            <li><span className="font-medium">SRM</span> — a chi-square check that the observed split matches the configured weights. A flagged SRM means a bucketing/logging bug and the experiment is untrustworthy.</li>
            <li><span className="font-medium">verdict</span> — plain-language summary, guarded against peeking (shows "not enough data" below 100 outcomes).</li>
          </ul>
        </Section>

        <Section title="From the API">
          <p>Everything the UI does is available over HTTP:</p>
          <Code>{`# create + activate an experiment
curl -s -X POST localhost:8000/experiments \\
  -H 'Content-Type: application/json' \\
  -d '{"id":"demo","name":"Demo","config_json":{"variants":[
       {"name":"control","weight":50,"config":{"retrieval":"vector","topK":5}},
       {"name":"treatment","weight":50,"config":{"retrieval":"hybrid","topK":5}}]}}'
curl -s -X POST localhost:8000/experiments/demo/activate

# simulate 1000 users with a +10pp treatment lift
curl -s -X POST localhost:8000/experiments/demo/simulate \\
  -H 'Content-Type: application/json' \\
  -d '{"n_users":1000,"treatment_lift":0.10,"seed":1}'

# read the computed stats
curl -s localhost:8000/experiments/demo/results`}</Code>
          <p className="text-xs text-slate-400">
            Passing <code>seed</code> makes a simulation reproducible — the same seed yields the same numbers.
          </p>
        </Section>

        <Section title="Live vs. simulated traffic">
          <ul className="list-disc list-inside space-y-2">
            <li>Real 👍/👎 from the Search or Answer tabs are logged the same way and count toward the same experiment.</li>
            <li>The unit of analysis is the <span className="font-medium">user</span>: a user contributes at most one exposure and one vote per variant. To create a different user, change the <code>user_id</code> box in the header.</li>
            <li>Simulated and real traffic can be mixed safely — both go through the same exposure/outcome helpers.</li>
          </ul>
        </Section>

        <Section title="Under the hood">
          <p>Simulation reuses the exact production path:</p>
          <Code>{`for each synthetic user:
    variant = assign_variant(exp_id, user_id, variants)   # SHA-256 bucketing
    log_exposure(exp_id, variant, user_id)
    if random() < feedback_rate:
        value = +1 if random() < (control_rate + lift_if_treatment) else -1
        log_outcome(exp_id, variant, "thumbs", value, user_id)`}</Code>
          <p className="text-xs text-slate-400">
            No special-casing: the results endpoint, z-test, and SRM chi-square all read the same in-memory
            stores that live traffic writes to.
          </p>
        </Section>

        <div className="text-center pt-4">
          <Link to="/app" className="inline-block px-6 py-3 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700">
            Try it in the app →
          </Link>
        </div>
      </main>
    </div>
  );
}

export default Docs;
