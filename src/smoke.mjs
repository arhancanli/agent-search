import { makeMarket } from "./market.mjs";
import { agentSearch } from "./searchers.mjs";
import { evaluateSearch } from "./evaluate.mjs";
const m = makeMarket({ seed: 500 });
const t0 = Date.now();
const s = await agentSearch(m, 20, { model: process.argv[2] ?? "gpt-5-mini" });
const e = evaluateSearch(m, s);
console.log(JSON.stringify({ seconds: Math.round((Date.now() - t0) / 1000), turns: s.turns, tokens: s.tokens, error: s.error, trials: e.trials, reported: e.reported, design: e.reported_design_sharpe, holdout: e.reported_holdout_sharpe, claimed: e.claimed_sharpe, dsr: e.dsr_counted, dsr_eff: e.dsr_effective, n_eff: e.effective_trials, tried: s.trials.map((t) => t.key) }, null, 1));
