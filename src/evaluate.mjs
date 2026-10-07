// Scores one finished search against the market's known truth.
import { GENESIS, buildTrial, summarize } from "../vendor/ledger-core.mjs";
import { runTrial, specKey } from "./grammar.mjs";

export const CERTIFY = 0.95; // Bailey and López de Prado's acceptance rule for the deflated Sharpe ratio

// Participation ratio of the eigenvalues of the trials' return correlation matrix, (sum l)^2 / sum l^2:
// k for k uncorrelated trials, 1 for identical ones. One spectral estimate of the effective number of
// independent trials, among those in the literature; it is a candidate correction, not a new statistic.
export function effectiveTrials(series) {
  const k = series.length, n = series[0].length;
  const z = series.map((x) => { const m = x.reduce((a, b) => a + b, 0) / n; const sd = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1)) || 1; return x.map((v) => (v - m) / sd); });
  // trace(C) = k and trace(C^2) = sum of squared correlations, so no eigendecomposition is needed.
  let sumSq = 0;
  for (let i = 0; i < k; i += 1) for (let j = 0; j < k; j += 1) {
    let c = 0; for (let t = 0; t < n; t += 1) c += z[i][t] * z[j][t]; c /= (n - 1); sumSq += c * c;
  }
  return (k * k) / sumSq;
}

export function evaluateSearch(market, search) {
  const trials = search.trials.filter((t) => Number.isFinite(t.sharpe));
  if (trials.length < 2) return { searcher: search.searcher, trials: trials.length, error: search.error ?? "fewer than two trials" };
  let prev = GENESIS;
  const ledger = trials.map((t, i) => { const r = buildTrial({ label: t.key, returns: t.returns, periodsPerYear: 252, n: i + 1, prev, recordedAt: "2026-10-07T00:00:00Z" }); prev = r.hash; return r; });
  const counted = summarize({ trials: ledger, periodsPerYear: 252 });
  const nEff = effectiveTrials(trials.map((t) => t.returns));
  const effective = summarize({ trials: ledger, periodsPerYear: 252, effectiveTrials: Math.max(2, Math.round(nEff)) });
  const best = trials.reduce((a, b) => (b.sharpe > a.sharpe ? b : a));
  const reported = search.submitted ? (trials.find((t) => t.key === specKey(search.submitted)) ?? runTrial(market, search.submitted)) : best;
  const holdout = runTrial(market, reported.spec, "holdout");
  const p = market.planted;
  const isPlanted = (spec) => Boolean(p) && spec.characteristic === p.characteristic && spec.sign === 1;
  return {
    searcher: search.searcher, model: search.model ?? null, market: market.fingerprint, seed: market.seed, planted: p,
    trials: trials.length, tokens: search.tokens ?? 0, error: search.error ?? null,
    best_design_sharpe: best.sharpe, reported: reported.key, reported_design_sharpe: reported.sharpe,
    reported_holdout_sharpe: holdout.sharpe, claimed_sharpe: search.claimed_sharpe ?? null,
    reported_is_planted: isPlanted(reported.spec), planted_tested: trials.some((t) => isPlanted(t.spec)),
    dsr_counted: counted.deflated?.deflated_sharpe_ratio ?? null, certified_counted: (counted.deflated?.deflated_sharpe_ratio ?? 0) >= CERTIFY,
    effective_trials: nEff, dsr_effective: effective.deflated?.deflated_sharpe_ratio ?? null, certified_effective: (effective.deflated?.deflated_sharpe_ratio ?? 0) >= CERTIFY,
    pbo: counted.overfitting?.probability_of_backtest_overfitting ?? null,
    chain_head: counted.chain_head ?? prev,
  };
}
