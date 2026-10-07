// js/pbo-core.js
// Probability of Backtest Overfitting by Combinatorially Symmetric Cross-Validation.
// A line-for-line port of alphaforge.validation.pbo.pbo_cscv. Where the Python samples
// combinations with numpy's generator this port uses the repo's mulberry32 and SAYS SO
// in the result (`exhaustive: false`): the exhaustive case is bit-comparable, the sampled
// case is comparable within sampling noise. Pinned to standards/validation-api/vectors.json.
import { makeRandom } from "./selection-risk-core.js";

function blockSharpe(rows, nConfigs) {
  // Population std (ddof=0); zero variance -> -Infinity so it is never IS-best and ranks worst OOS.
  const out = new Array(nConfigs);
  for (let j = 0; j < nConfigs; j++) {
    let sum = 0;
    for (const row of rows) sum += row[j];
    const mean = sum / rows.length;
    let ss = 0;
    for (const row of rows) { const d = row[j] - mean; ss += d * d; }
    const std = Math.sqrt(ss / rows.length);
    out[j] = std > 0 ? mean / std : Number.NEGATIVE_INFINITY;
  }
  return out;
}

function averageRanks(values) {
  // scipy.stats.rankdata(method="average"): 1-based, ties share the mean rank.
  const order = values.map((v, i) => [v, i]).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const ranks = new Array(values.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
    const rank = (i + 1 + j + 1) / 2;
    for (let k = i; k <= j; k++) ranks[order[k][1]] = rank;
    i = j + 1;
  }
  return ranks;
}

function binomial(n, k) {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
}

function* combinations(n, k) {
  const idx = Array.from({ length: k }, (_, i) => i);
  while (true) {
    yield idx.slice();
    let i = k - 1;
    while (i >= 0 && idx[i] === n - k + i) i--;
    if (i < 0) return;
    idx[i]++;
    for (let j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
  }
}

function sampleCombinations(n, k, count, seed) {
  const next = makeRandom(seed);
  const seen = new Set();
  const out = [];
  while (out.length < count) {
    const pool = Array.from({ length: n }, (_, i) => i);
    const pick = [];
    for (let d = 0; d < k; d++) {
      const at = Math.floor(next() * pool.length);
      pick.push(pool[at]);
      pool.splice(at, 1);
    }
    pick.sort((a, b) => a - b);
    const key = pick.join(",");
    if (!seen.has(key)) { seen.add(key); out.push(pick); }
  }
  return out.sort((a, b) => a.join(",").localeCompare(b.join(","), undefined, { numeric: true }));
}

export function pboCscv(matrix, { nSplits = 16, maxCombinations = 2000, seed = 42 } = {}) {
  if (!Number.isInteger(nSplits) || nSplits < 2 || nSplits % 2 !== 0) throw new RangeError(`n_splits must be an even integer >= 2; got ${nSplits}`);
  if (!Number.isInteger(maxCombinations) || maxCombinations < 1) throw new RangeError(`max_combinations must be >= 1; got ${maxCombinations}`);
  if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])) throw new RangeError("matrix must be a non-empty array of rows");
  const nConfigs = matrix[0].length;
  if (nConfigs < 2) throw new RangeError(`matrix needs >= 2 config columns to rank cross-sectionally; got ${nConfigs}`);
  for (const row of matrix) {
    if (row.length !== nConfigs) throw new RangeError("every row must have the same number of columns");
    for (const v of row) if (!Number.isFinite(Number(v))) throw new RangeError("matrix values must be finite numbers");
  }
  const nRows = matrix.length;
  if (nRows < nSplits) throw new RangeError(`matrix has ${nRows} rows < n_splits=${nSplits}; cannot form that many non-empty contiguous blocks`);
  const blockLen = Math.floor(nRows / nSplits);
  const blocks = Array.from({ length: nSplits }, (_, b) => matrix.slice(b * blockLen, (b + 1) * blockLen).map((r) => r.map(Number)));
  const half = nSplits / 2;
  const total = binomial(nSplits, half);
  const exhaustive = total <= maxCombinations;
  const combos = exhaustive ? [...combinations(nSplits, half)] : sampleCombinations(nSplits, half, maxCombinations, seed);
  const lambdas = [];
  const isOos = [];
  for (const isBlocks of combos) {
    const isSet = new Set(isBlocks);
    const isRows = isBlocks.flatMap((b) => blocks[b]);
    const oosRows = [];
    for (let b = 0; b < nSplits; b++) if (!isSet.has(b)) oosRows.push(...blocks[b]);
    const srIs = blockSharpe(isRows, nConfigs);
    let best = 0;
    for (let j = 1; j < nConfigs; j++) if (srIs[j] > srIs[best]) best = j;
    const srOos = blockSharpe(oosRows, nConfigs);
    const ranks = averageRanks(srOos);
    const omega = ranks[best] / (nConfigs + 1);
    lambdas.push(Math.log(omega / (1 - omega)));
    isOos.push([srIs[best], srOos[best]]);
  }
  const pbo = lambdas.filter((l) => l <= 0).length / lambdas.length;
  return { pbo, n_combinations: combos.length, lambdas, is_oos_pairs: isOos, exhaustive, block_length: blockLen };
}
