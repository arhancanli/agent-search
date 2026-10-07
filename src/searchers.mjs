// The four searchers, each spending the same trial budget B on the design window of one market:
//   random    B strategies drawn uniformly from the grammar, without replacement
//   grid      a fixed order over the grammar (shuffled once per market seed), first B
//   agent     an LLM that backtests up to 10 strategies per call, sees each result, adapts, then submits one
//   agent_na  the same LLM proposing all B strategies at once, before seeing any result
// Every backtest goes through runTrial and is returned in the order it was run.
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";

import { mulberry32 } from "./market.mjs";
import { GRAMMAR, enumerateGrid, runTrial, specKey, validateSpec } from "./grammar.mjs";

const shuffle = (arr, rng) => { const a = [...arr]; for (let i = a.length - 1; i > 0; i -= 1) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

export function randomSearch(market, B, seed) {
  const specs = shuffle(enumerateGrid(market.o.characteristics), mulberry32(seed)).slice(0, B);
  return { searcher: "random", trials: specs.map((s) => runTrial(market, s)), submitted: null };
}
export function gridSearch(market, B) {
  // A practitioner's grid: characteristics in order, every variant of each, until the budget is spent.
  // Coarse to fine: every characteristic at each setting before the next setting, so a budget smaller
  // than the grammar still covers all characteristics.
  const all = enumerateGrid(market.o.characteristics);
  const coarse = (s) => [s.rebalance === "monthly" ? 0 : 1, s.quantile === 0.2 ? 0 : s.quantile === 0.1 ? 1 : 2, ["level", "mean", "change"].indexOf(s.transform), s.window ?? 0, s.sign > 0 ? 0 : 1, s.characteristic];
  const specs = [...all].sort((a, b) => { const x = coarse(a), y = coarse(b); for (let i = 0; i < x.length; i += 1) if (x[i] !== y[i]) return x[i] - y[i]; return 0; }).slice(0, B);
  return { searcher: "grid", trials: specs.map((s) => runTrial(market, s)), submitted: null };
}

const SPEC_SCHEMA = {
  type: "object", additionalProperties: false,
  properties: {
    characteristic: { type: "integer", description: "Characteristic index, 0 to 9." },
    transform: { type: "string", enum: GRAMMAR.transform, description: "level: today's value; change: value minus value `window` days ago; mean: average over the last `window` days." },
    window: { type: "integer", enum: GRAMMAR.window, description: "Lookback in trading days; ignored for level." },
    sign: { type: "integer", enum: GRAMMAR.sign, description: "1 buys high values, -1 buys low values." },
    quantile: { type: "number", enum: GRAMMAR.quantile, description: "Fraction of assets held long and short." },
    rebalance: { type: "string", enum: GRAMMAR.rebalance },
  },
  required: ["characteristic", "transform", "sign", "quantile", "rebalance"],
};
const SYSTEM = (B) => `You are a quantitative researcher. A market of 100 assets has 10 firm characteristics (indices 0-9). Find the long-short strategy with the highest annualized Sharpe ratio. You have a budget of ${B} backtests in total; each strategy you test counts once. Report the strategy you believe is best.`;

const key = () => readFileSync(resolve(homedir(), ".config/canli/openai_benchmark_key"), "utf8").trim();
async function chat(model, messages, tools) {
  for (let attempt = 0; attempt < 6; attempt += 1) {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST", headers: { authorization: `Bearer ${key()}`, "content-type": "application/json" },
      body: JSON.stringify({ model, messages, tools, tool_choice: "required" }),
    });
    if (res.ok) return res.json();
    if (res.status === 429 || res.status >= 500) { await new Promise((r) => setTimeout(r, 2000 * 2 ** attempt)); continue; }
    throw new Error(`OpenAI ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  throw new Error("OpenAI: retries exhausted");
}

const summary = (t) => ({ strategy: t.key, sharpe: Number(t.sharpe?.toFixed(3)), annual_return: Number(t.annualReturn?.toFixed(4)), max_drawdown: Number(t.maxDrawdown?.toFixed(4)) });

export async function agentSearch(market, B, { model, request = chat, maxTurns = 40 }) {
  const K = market.o.characteristics;
  const tools = [
    { type: "function", function: { name: "backtest", description: `Backtest up to 10 strategies on the design window. Each counts against the budget of ${B}.`, parameters: { type: "object", additionalProperties: false, properties: { strategies: { type: "array", items: SPEC_SCHEMA, minItems: 1, maxItems: 10 } }, required: ["strategies"] } } },
    { type: "function", function: { name: "submit", description: "Report the strategy you believe is best. Ends the search.", parameters: { type: "object", additionalProperties: false, properties: { strategy: SPEC_SCHEMA, expected_sharpe: { type: "number", description: "The annualized Sharpe ratio you expect it to earn on new data." } }, required: ["strategy", "expected_sharpe"] } } },
  ];
  const messages = [{ role: "system", content: SYSTEM(B) }, { role: "user", content: "Begin." }];
  const trials = [], seen = new Map(); let tokens = 0, submitted = null, claimed = null, error = null, turns = 0;
  try {
    while (turns < maxTurns && submitted === null) {
      turns += 1;
      const out = await request(model, messages, tools);
      tokens += (out.usage?.prompt_tokens ?? 0) + (out.usage?.completion_tokens ?? 0);
      const msg = out.choices[0].message; messages.push(msg);
      for (const call of msg.tool_calls ?? []) {
        let args = {}; try { args = JSON.parse(call.function.arguments || "{}"); } catch { /* reported below */ }
        let reply;
        if (call.function.name === "submit") {
          try { submitted = validateSpec(args.strategy, K); claimed = args.expected_sharpe; reply = { ok: true }; }
          catch (e) { reply = { error: e.message }; }
        } else {
          const results = [];
          for (const raw of args.strategies ?? []) {
            let spec; try { spec = validateSpec(raw, K); } catch (e) { results.push({ error: e.message }); continue; }
            const k = specKey(spec);
            if (seen.has(k)) { results.push({ ...summary(seen.get(k)), note: "already tested; not counted again" }); continue; }
            if (trials.length >= B) { results.push({ error: "budget exhausted; submit your best strategy" }); continue; }
            const t = runTrial(market, spec); trials.push(t); seen.set(k, t); results.push(summary(t));
          }
          reply = { results, budget_left: B - trials.length };
        }
        messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(reply) });
      }
    }
    if (submitted === null) error = "no submission within the turn limit";
  } catch (e) { error = e.message; }
  return { searcher: "agent", model, trials, submitted, claimed_sharpe: claimed, tokens, turns, error };
}

export async function agentNonAdaptive(market, B, { model, request = chat }) {
  const K = market.o.characteristics;
  const tools = [{ type: "function", function: { name: "propose", description: `Propose exactly ${B} distinct strategies to backtest. You will not see any result before proposing.`, parameters: { type: "object", additionalProperties: false, properties: { strategies: { type: "array", items: SPEC_SCHEMA, minItems: 1 } }, required: ["strategies"] } } }];
  const messages = [{ role: "system", content: SYSTEM(B) }, { role: "user", content: `Propose your ${B} strategies now; the best by design-window Sharpe will be reported.` }];
  let tokens = 0, error = null; const trials = [], seen = new Set();
  try {
    const out = await request(model, messages, tools);
    tokens += (out.usage?.prompt_tokens ?? 0) + (out.usage?.completion_tokens ?? 0);
    const call = out.choices[0].message.tool_calls?.[0];
    const args = JSON.parse(call?.function?.arguments || "{}");
    for (const raw of args.strategies ?? []) {
      if (trials.length >= B) break;
      let spec; try { spec = validateSpec(raw, K); } catch { continue; }
      const k = specKey(spec); if (seen.has(k)) continue; seen.add(k);
      trials.push(runTrial(market, spec));
    }
    if (!trials.length) error = "no valid strategy proposed";
  } catch (e) { error = e.message; }
  return { searcher: "agent_na", model, trials, submitted: null, tokens, error };
}
