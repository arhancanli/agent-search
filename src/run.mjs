// Runs the registered cells and appends one JSON line per search to runs/<cell>.jsonl. Resumable: a
// (searcher, model, budget, seed) already present is skipped. Stops before a token cap for the day.
//   node src/run.mjs --kind null|alt|baseline-null|baseline-alt --model gpt-5-mini --cap 2000000 [--limit N]
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";

import { makeMarket } from "./market.mjs";
import { agentNonAdaptive, agentSearch, gridSearch, randomSearch } from "./searchers.mjs";
import { evaluateSearch } from "./evaluate.mjs";

const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i > 0 ? process.argv[i + 1] : d; };
const kind = arg("kind"), model = arg("model", "gpt-5-mini"), cap = Number(arg("cap", "2000000")), limit = Number(arg("limit", "1000000"));
const RANGES = { null: [1000001, 1000150], alt: [2000001, 2000100], "baseline-null": [3000001, 3002000], "baseline-alt": [4000001, 4001000] };
if (!RANGES[kind]) throw new Error("--kind must be null, alt, baseline-null or baseline-alt");
const BUDGETS = [20, 100];
const planted = (seed) => (kind.endsWith("alt") ? { characteristic: seed % 10, sharpe: 2.0 } : null);
mkdirSync("runs", { recursive: true });
const file = `runs/${kind}.jsonl`;
const done = new Set(existsSync(file) ? readFileSync(file, "utf8").trim().split("\n").filter(Boolean).map((l) => { const r = JSON.parse(l); return `${r.searcher}|${r.model}|${r.budget}|${r.seed}`; }) : []);
let tokens = 0, ran = 0;
const record = (row) => { appendFileSync(file, `${JSON.stringify(row)}\n`); done.add(`${row.searcher}|${row.model}|${row.budget}|${row.seed}`); };
const [lo, hi] = RANGES[kind];
for (let seed = lo; seed <= hi && ran < limit; seed += 1) {
  const market = makeMarket({ seed, planted: planted(seed) });
  for (const B of BUDGETS) {
    const jobs = [["random", null, () => randomSearch(market, B, seed)], ["grid", null, () => gridSearch(market, B)]];
    if (!kind.startsWith("baseline")) jobs.push(["agent", model, () => agentSearch(market, B, { model })], ["agent_na", model, () => agentNonAdaptive(market, B, { model })]);
    for (const [searcher, m, go] of jobs) {
      if (done.has(`${searcher}|${m}|${B}|${seed}`)) continue;
      if (m && tokens > cap) { console.error(`STOP token cap ${tokens} > ${cap}`); process.exit(0); }
      let s = await go();
      if (s.error && m) s = await go(); // one retry on the same market, as registered
      tokens += s.tokens ?? 0;
      const e = evaluateSearch(market, s);
      record({ ...e, searcher, model: m, budget: B, seed, kind });
    }
  }
  ran += 1;
  if (ran % 10 === 0) console.error(`${kind} seed ${seed}: ${ran} markets, ${tokens} tokens`);
}
console.error(`DONE ${kind} ${model}: ${ran} markets, ${tokens} tokens`);
