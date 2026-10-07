// mcp/src/ledger-core.mjs
//
// The trial ledger: every strategy variant an agent tries in one research search, recorded as it is
// tried, so the result it finally reports is judged against the whole search instead of alone. An
// agent can run thousands of variants in an hour; the best of them looks skilled by luck alone, and
// the size of that luck depends on how many were tried and how spread out their results were.
//
// Each trial is one line in a hash chain (sha256 over the previous hash and the trial's canonical
// JSON), so an exported ledger shows any trial removed, reordered or edited after the fact. The
// statistics reuse the site's checked cores: perPeriodMoments and calculateDsr (pinned to
// standards/validation-api/vectors.json) and pboCscv. Pure functions only; storage is in ledger.mjs.
import { createHash } from "node:crypto";

import { calculateDsr, expectedMaxStandardNormal } from "./local/js/dsr-core.js";
import { finiteNumbers, perPeriodMoments } from "./local/js/moments-core.js";
import { pboCscv } from "./local/js/pbo-core.js";

export const LEDGER_SCHEMA = "canli.trial-ledger.v1";
export const GENESIS = "0".repeat(64);
export const MAX_TRIALS = 10000;
export const MAX_RETURNS = 20000;

export const LEDGER_LIMITS_TEXT = Object.freeze([
  "The ledger counts the trials recorded in it; a trial run but never recorded is invisible to it, so the correction is only as complete as the record.",
  "Treating every recorded trial as independent overstates the search when trials are correlated variants of one idea, which makes the deflated Sharpe conservative; pass effective_trials when you have a better estimate.",
  "A deflated Sharpe ratio is a probability that the best trial's true Sharpe is above zero after the search; it is not a forecast of returns and not a recommendation.",
]);

const sha256 = (text) => createHash("sha256").update(text).digest("hex");

// Keys sorted at every level, so the same trial always hashes the same way.
export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(",")}}`;
  return JSON.stringify(value);
}

// A trial as the agent describes it -> the record that is chained. Exactly one of returns, or a
// Sharpe with its observation count. periodsPerYear annualizes; it is fixed for the whole ledger.
export function buildTrial({ label, returns, sharpe_annualized, observations, params, periodsPerYear, n, prev, recordedAt }) {
  if (typeof label !== "string" || !label.trim()) throw new RangeError("label must be a non-empty string naming the variant");
  if (!(periodsPerYear > 0)) throw new RangeError("periods_per_year must be greater than zero");
  const hasReturns = returns !== undefined && returns !== null;
  const hasSharpe = sharpe_annualized !== undefined && sharpe_annualized !== null;
  if (hasReturns === hasSharpe) throw new RangeError("give either returns or sharpe_annualized with observations, not both and not neither");
  let trial;
  if (hasReturns) {
    const x = finiteNumbers(returns, "returns");
    if (x.length > MAX_RETURNS) throw new RangeError(`returns holds ${x.length} observations; the limit is ${MAX_RETURNS}`);
    const m = perPeriodMoments(x);
    trial = { sharpe_annualized: m.sharpe_per_period * Math.sqrt(periodsPerYear), observations: m.observations, skew: m.skew, non_excess_kurtosis: m.non_excess_kurtosis, returns: x };
  } else {
    if (!Number.isFinite(sharpe_annualized)) throw new RangeError("sharpe_annualized must be a finite number");
    if (!Number.isInteger(observations) || observations < 2) throw new RangeError("observations must be an integer of at least 2 when a Sharpe is given");
    trial = { sharpe_annualized, observations, skew: null, non_excess_kurtosis: null, returns: null };
  }
  const body = { n, label: label.trim(), params: params ?? null, recorded_at: recordedAt, ...trial };
  const hash = sha256(prev + canonicalJson(body));
  return { ...body, prev, hash };
}

// Recomputes the chain. Returns { ok, head, trials } or { ok: false, at, reason }.
export function verifyChain(trials) {
  let prev = GENESIS;
  for (let i = 0; i < trials.length; i += 1) {
    const { prev: recordedPrev, hash, ...body } = trials[i];
    if (body.n !== i + 1) return { ok: false, at: i + 1, reason: `trial ${i + 1} is numbered ${body.n}` };
    if (recordedPrev !== prev) return { ok: false, at: i + 1, reason: `trial ${i + 1} does not follow trial ${i}` };
    if (sha256(prev + canonicalJson(body)) !== hash) return { ok: false, at: i + 1, reason: `trial ${i + 1} was changed after it was recorded` };
    prev = hash;
  }
  return { ok: true, head: prev, trials: trials.length };
}

const sampleSd = (xs) => {
  if (xs.length < 2) return null;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1));
};

// The largest even number of CSCV blocks (at most 16) that leaves at least two periods per block.
const cscvSplits = (length) => {
  for (let s = 16; s >= 4; s -= 2) if (length / s >= 2) return s;
  return null;
};

const fmt = (x, d = 2) => (x === null || x === undefined ? "n/a" : Number(x).toFixed(d));

// The verdict on the whole search so far.
export function summarize({ trials, periodsPerYear, effectiveTrials }) {
  const n = trials.length;
  if (n === 0) return { trials: 0, plain_reading: "No trials are recorded in this ledger yet.", limits: LEDGER_LIMITS_TEXT };
  const sharpes = trials.map((t) => t.sharpe_annualized);
  const bestIndex = sharpes.reduce((bi, s, i) => (s > sharpes[bi] ? i : bi), 0);
  const best = trials[bestIndex];
  const spread = sampleSd(sharpes);
  const counted = effectiveTrials ?? n;
  const assumedShape = best.skew === null;
  let deflated = null;
  if (counted >= 2 && spread !== null) {
    const d = calculateDsr({
      observed_sharpe_annualized: best.sharpe_annualized, observations: best.observations, periods_per_year: periodsPerYear,
      skew: assumedShape ? 0 : best.skew, non_excess_kurtosis: assumedShape ? 3 : best.non_excess_kurtosis,
      effective_independent_trials: counted, cross_trial_sharpe_sd_annualized: spread,
    });
    deflated = { deflated_sharpe_ratio: d.deflated_sharpe_ratio, psr_against_zero: d.psr_against_zero, expected_max_sharpe_annualized: d.expected_max_sharpe_annualized, search_haircut_annualized: d.search_haircut_annualized };
  }
  let overfitting = null;
  const withReturns = trials.filter((t) => Array.isArray(t.returns));
  const lengths = new Set(withReturns.map((t) => t.returns.length));
  if (withReturns.length === n && n >= 4 && lengths.size === 1) {
    const length = [...lengths][0];
    const splits = cscvSplits(length);
    if (splits) {
      const matrix = Array.from({ length }, (_, r) => withReturns.map((t) => t.returns[r]));
      const p = pboCscv(matrix, { nSplits: splits });
      overfitting = { probability_of_backtest_overfitting: p.pbo, n_splits: splits, combinations: p.n_combinations };
    }
  }
  const luck = spread !== null && counted >= 2 ? spread * expectedMaxStandardNormal(counted) : null;
  const sentences = [
    `Across ${n} recorded trial${n === 1 ? "" : "s"}, the best is "${best.label}" with an annualized Sharpe of ${fmt(best.sharpe_annualized)}.`,
  ];
  if (luck !== null) sentences.push(`With this spread of results, the best of ${counted} skill-less trials would be expected to reach about ${fmt(luck)} by luck alone.`);
  if (deflated) sentences.push(`Its deflated Sharpe ratio is ${fmt(deflated.deflated_sharpe_ratio)}: the probability that its true Sharpe is above zero once the search is counted.`);
  else sentences.push("A deflated Sharpe needs at least two trials with different results.");
  if (overfitting) sentences.push(`The probability of backtest overfitting (CSCV, ${overfitting.n_splits} blocks) is ${fmt(overfitting.probability_of_backtest_overfitting)}.`);
  if (assumedShape) sentences.push("The best trial was recorded as a Sharpe without returns, so normal returns were assumed for it.");
  return {
    trials: n, effective_trials: counted,
    best: { n: best.n, label: best.label, params: best.params, sharpe_annualized: best.sharpe_annualized, observations: best.observations },
    cross_trial_sharpe_sd_annualized: spread,
    expected_best_by_luck_annualized: luck,
    deflated, overfitting,
    plain_reading: sentences.join(" "),
    limits: LEDGER_LIMITS_TEXT,
  };
}
