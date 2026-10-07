import assert from "node:assert/strict";
import test from "node:test";

import { makeMarket } from "../src/market.mjs";
import { enumerateGrid, runTrial, validateSpec } from "../src/grammar.mjs";
import { agentNonAdaptive, agentSearch, gridSearch, randomSearch } from "../src/searchers.mjs";
import { effectiveTrials, evaluateSearch } from "../src/evaluate.mjs";

const market = makeMarket({ seed: 11, assets: 40, designDays: 300, holdoutDays: 200 });
const spec = (c, extra = {}) => ({ characteristic: c, transform: "level", sign: 1, quantile: 0.2, rebalance: "monthly", ...extra });
const reply = (calls) => ({ usage: { prompt_tokens: 10, completion_tokens: 5 }, choices: [{ message: { role: "assistant", tool_calls: calls.map(([name, args], i) => ({ id: `c${i}`, type: "function", function: { name, arguments: JSON.stringify(args) } })) } }] });

test("the grammar refuses anything outside it, so lookahead and free-form code cannot be expressed", () => {
  assert.throws(() => validateSpec(spec(10), 10), /characteristic/);
  assert.throws(() => validateSpec(spec(1, { transform: "future" }), 10), /transform/);
  assert.throws(() => validateSpec(spec(1, { transform: "mean", window: 7 }), 10), /window/);
  assert.equal(enumerateGrid(10).length, 840);
  assert.equal(validateSpec(spec(1, { window: 21 }), 10).window, undefined, "level ignores the window");
});

test("every trial shares the same dates, as the overfitting statistics require", () => {
  const a = runTrial(market, spec(0)), b = runTrial(market, spec(1, { transform: "mean", window: 63, rebalance: "weekly" }));
  assert.deepEqual(a.dates, b.dates);
});

test("random and grid searches spend exactly the budget on distinct strategies", () => {
  for (const s of [randomSearch(market, 25, 1), gridSearch(market, 25)]) {
    assert.equal(s.trials.length, 25);
    assert.equal(new Set(s.trials.map((t) => t.key)).size, 25);
  }
});

test("the adaptive agent is charged once per distinct strategy, stopped at the budget, and ends by submitting", async () => {
  const script = [
    reply([["backtest", { strategies: [spec(0), spec(1), spec(0)] }]]),
    reply([["backtest", { strategies: [spec(2), spec(3), spec(4)] }]]),
    reply([["submit", { strategy: spec(2), expected_sharpe: 1.1 }]]),
  ];
  const seenToolText = [];
  const request = async (model, messages) => { seenToolText.push(messages.filter((m) => m.role === "tool").map((m) => m.content)); return script.shift(); };
  const out = await agentSearch(market, 4, { model: "mock", request });
  assert.equal(out.trials.length, 4, "the repeat is free and the fifth strategy is refused");
  assert.match(seenToolText[2].at(-1), /budget exhausted/);
  assert.match(seenToolText[1].at(-1), /already tested/);
  assert.equal(out.submitted.characteristic, 2);
  assert.equal(out.claimed_sharpe, 1.1);
  assert.equal(out.tokens, 45);
});

test("the non-adaptive agent's proposals are deduplicated and capped at the budget", async () => {
  const request = async () => reply([["propose", { strategies: [spec(0), spec(0), spec(1), spec(2), spec(3)] }]]);
  const out = await agentNonAdaptive(market, 3, { model: "mock", request });
  assert.deepEqual(out.trials.map((t) => t.spec.characteristic), [0, 1, 2]);
});

test("evaluation reports the submitted strategy's held-out Sharpe, and the ledger's deflated Sharpe", async () => {
  const s = randomSearch(market, 12, 3);
  s.submitted = s.trials[5].spec;
  const e = evaluateSearch(market, s);
  assert.equal(e.reported, s.trials[5].key);
  assert.ok(Number.isFinite(e.reported_holdout_sharpe));
  assert.ok(e.dsr_counted >= 0 && e.dsr_counted <= 1);
  assert.match(e.chain_head, /^[0-9a-f]{64}$/);
});

test("the effective number of trials is k for independent series and 1 for identical ones", () => {
  let s = 1; const u = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
  const indep = Array.from({ length: 5 }, () => Array.from({ length: 4000 }, u));
  assert.ok(Math.abs(effectiveTrials(indep) - 5) < 0.1);
  const same = Array.from({ length: 5 }, () => indep[0]);
  assert.ok(Math.abs(effectiveTrials(same) - 1) < 1e-9);
});

test("under the alternative the planted strategy is recognised as planted", () => {
  const m = makeMarket({ seed: 12, assets: 40, designDays: 300, holdoutDays: 200, planted: { characteristic: 3, sharpe: 2 } });
  const s = { searcher: "x", trials: [runTrial(m, spec(3)), runTrial(m, spec(4))], submitted: spec(3) };
  assert.equal(evaluateSearch(m, s).reported_is_planted, true);
});
