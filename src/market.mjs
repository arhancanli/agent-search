// Synthetic markets with known ground truth. N assets, K characteristics per asset, T trading days split
// into a design window (the searcher sees it) and a held-out window (it never does).
//
// Characteristics follow a persistent AR(1) per asset and are standardized across assets each day, like
// firm characteristics. Returns are a market factor plus idiosyncratic noise. Under the null no
// characteristic predicts returns. Under the alternative exactly one characteristic, at a lag of one day,
// shifts expected returns so that a top-minus-bottom quintile portfolio rebalanced monthly has about the
// requested annualized Sharpe ratio.
import { createHash } from "node:crypto";

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function normals(rng) {
  let spare = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0; while (u === 0) u = rng(); v = rng();
    const r = Math.sqrt(-2 * Math.log(u)); spare = r * Math.sin(2 * Math.PI * v); return r * Math.cos(2 * Math.PI * v);
  };
}

export const DEFAULTS = Object.freeze({ assets: 100, characteristics: 10, designDays: 756, holdoutDays: 504, phi: 0.99, marketVol: 0.01, idioVol: 0.02 });

// Business-day labels from 2000-01-03, so the backtest engine can rebalance by week or month.
export function tradingDates(n) {
  const out = []; const d = new Date(Date.UTC(2000, 0, 3));
  while (out.length < n) { const w = d.getUTCDay(); if (w !== 0 && w !== 6) out.push(d.toISOString().slice(0, 10)); d.setUTCDate(d.getUTCDate() + 1); }
  return out;
}

// planted: null, or { characteristic: index, sharpe: annualized target for the quintile spread }.
export function makeMarket({ seed, planted = null, ...opts }) {
  const o = { ...DEFAULTS, ...opts };
  const { assets: N, characteristics: K, designDays, holdoutDays, phi, marketVol, idioVol } = o;
  const T = designDays + holdoutDays;
  const z = normals(mulberry32(seed));
  // chars[k][i] is a Float64Array over days, standardized across assets each day.
  const chars = Array.from({ length: K }, () => Array.from({ length: N }, () => new Float64Array(T)));
  const innov = Math.sqrt(1 - phi * phi);
  for (let k = 0; k < K; k += 1) for (let i = 0; i < N; i += 1) {
    let x = z(); chars[k][i][0] = x;
    for (let t = 1; t < T; t += 1) { x = phi * x + innov * z(); chars[k][i][t] = x; }
  }
  for (let k = 0; k < K; k += 1) for (let t = 0; t < T; t += 1) {
    let s = 0, s2 = 0; for (let i = 0; i < N; i += 1) { const v = chars[k][i][t]; s += v; s2 += v * v; }
    const m = s / N, sd = Math.sqrt(Math.max(s2 / N - m * m, 1e-12));
    for (let i = 0; i < N; i += 1) chars[k][i][t] = (chars[k][i][t] - m) / sd;
  }
  // A top-minus-bottom quintile spread of an N(0,1) characteristic has a mean gap of about 2.8 standard
  // units, and its daily volatility is about idioVol * sqrt(2 / (N / 5)). Solve for the slope b so the
  // annualized Sharpe of the spread is near the target.
  const spreadVol = idioVol * Math.sqrt(2 / (N / 5));
  const b = planted ? (planted.sharpe / Math.sqrt(252)) * spreadVol / 2.8 : 0;
  const returns = Array.from({ length: N }, () => new Float64Array(T));
  for (let t = 0; t < T; t += 1) {
    const mkt = marketVol * z();
    for (let i = 0; i < N; i += 1) {
      const signal = planted && t > 0 ? b * chars[planted.characteristic][i][t - 1] : 0;
      returns[i][t] = mkt + signal + idioVol * z();
    }
  }
  const dates = tradingDates(T);
  const tickers = Array.from({ length: N }, (_, i) => `A${String(i).padStart(3, "0")}`);
  const fingerprint = createHash("sha256").update(JSON.stringify({ seed, planted, o })).digest("hex").slice(0, 16);
  return { o, seed, planted, dates, tickers, chars, returns, designDays, holdoutDays, fingerprint };
}

// Price paths from returns over a day range [from, to), for the backtest engine.
export function pricePaths(market, from, to) {
  const prices = {};
  market.tickers.forEach((tk, i) => {
    const p = new Array(to - from); let level = 100;
    for (let t = from; t < to; t += 1) { level *= 1 + market.returns[i][t]; p[t - from] = level; }
    prices[tk] = p;
  });
  return prices;
}
