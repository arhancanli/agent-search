// The strategy grammar every searcher uses, and the trial runner. A strategy ranks assets on one
// transformed characteristic and holds a top-minus-bottom portfolio. Signals at day t use characteristic
// values up to day t only, and the engine trades on them no earlier than the next day, so lookahead cannot
// be expressed.
import { describe, runBacktest } from "../vendor/engine.mjs";
import { pricePaths } from "./market.mjs";

export const GRAMMAR = Object.freeze({
  transform: ["level", "change", "mean"],
  window: [5, 21, 63],
  sign: [1, -1],
  quantile: [0.1, 0.2, 0.3],
  rebalance: ["weekly", "monthly"],
});
export const WARMUP = 63; // every strategy starts after the longest window, so all trials share dates
const COST_BPS = 5;

export function validateSpec(spec, K) {
  const errs = [];
  if (!Number.isInteger(spec?.characteristic) || spec.characteristic < 0 || spec.characteristic >= K) errs.push(`characteristic must be an integer 0-${K - 1}`);
  for (const [key, allowed] of Object.entries(GRAMMAR)) {
    if (key === "window" && spec?.transform === "level") continue;
    if (!allowed.includes(spec?.[key])) errs.push(`${key} must be one of ${allowed.join(", ")}`);
  }
  if (errs.length) throw new RangeError(errs.join("; "));
  const out = { characteristic: spec.characteristic, transform: spec.transform, sign: spec.sign, quantile: spec.quantile, rebalance: spec.rebalance };
  if (spec.transform !== "level") out.window = spec.window;
  return out;
}
export const specKey = (s) => `c${s.characteristic}:${s.transform}${s.window ? `/${s.window}` : ""}:${s.sign > 0 ? "+" : "-"}:q${s.quantile}:${s.rebalance}`;

export function enumerateGrid(K) {
  const out = [];
  for (let c = 0; c < K; c += 1) for (const transform of GRAMMAR.transform) for (const window of transform === "level" ? [null] : GRAMMAR.window)
    for (const sign of GRAMMAR.sign) for (const quantile of GRAMMAR.quantile) for (const rebalance of GRAMMAR.rebalance)
      out.push(validateSpec({ characteristic: c, transform, window, sign, quantile, rebalance }, K));
  return out;
}

function signalValue(series, t, spec) {
  if (spec.transform === "level") return spec.sign * series[t];
  const w = spec.window;
  if (t - w < 0) return NaN;
  if (spec.transform === "change") return spec.sign * (series[t] - series[t - w]);
  let s = 0; for (let j = t - w + 1; j <= t; j += 1) s += series[j];
  return spec.sign * (s / w);
}

// window: "design" (days WARMUP..designDays) or "holdout" (designDays..end, lookbacks may reach back).
export function runTrial(market, rawSpec, window = "design") {
  const spec = validateSpec(rawSpec, market.o.characteristics);
  const from = window === "design" ? WARMUP : market.designDays;
  const to = window === "design" ? market.designDays : market.designDays + market.holdoutDays;
  const dates = market.dates.slice(from, to);
  const signals = {};
  market.tickers.forEach((tk, i) => {
    const series = market.chars[spec.characteristic][i];
    const a = new Array(to - from);
    for (let t = from; t < to; t += 1) a[t - from] = signalValue(series, t, spec);
    signals[tk] = a;
  });
  const result = runBacktest({ dates, tickers: market.tickers, prices: pricePaths(market, from, to), signalDates: dates, signals,
    rebalance: spec.rebalance, lagDays: 1, side: "long_short", quantile: spec.quantile, costBps: COST_BPS, minNames: 10, periodsPerYear: 252 });
  const d = describe(result);
  return { spec, key: specKey(spec), window, dates: result.dates, returns: result.returns, sharpe: d.sharpe_annualized, annualReturn: d.annualized_return, maxDrawdown: d.max_drawdown, turnoverPerYear: d.turnover_per_year };
}
