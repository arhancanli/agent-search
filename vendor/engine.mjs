// mcp-backtest/src/engine.mjs
//
// A cross-sectional backtest of any signal against prices the caller supplies, built so the usual
// ways to fool yourself are refused in code:
//
// - Each signal row is dated when it became KNOWN. A trade on day t uses only signal rows dated at
//   least lag_days trading days before t; every rebalance records the signal date it used, and the
//   run stops if one ever reaches past that bound.
// - Positions drift with prices between rebalances. Turnover and costs are charged on the actual
//   trades from drifted positions to targets, not on an idealized daily rebalance.
// - A held name with no price on a day earns nothing that day and is counted; nothing is filled in.
//
// Pure: no files, no network. Inputs are aligned arrays; series-io.mjs reads CSV files into them.

export const ENGINE_LIMITS_TEXT = Object.freeze([
  "Prices are the caller's: survivorship, splits, dividends and data errors in them pass straight into the result.",
  "A backtest measures one path of history under these costs; it does not establish that the signal will keep working.",
  "Costs are a flat rate per unit traded; market impact, borrow fees for shorts and financing are not charged unless included in cost_bps.",
]);

const REBALANCE = Object.freeze({ weekly: "week", monthly: "month", quarterly: "quarter" });
const periodKey = (date, every) => {
  const [y, m] = date.split("-").map(Number);
  if (every === "month") return `${y}-${m}`;
  if (every === "quarter") return `${y}-Q${Math.ceil(m / 3)}`;
  const d = new Date(`${date}T00:00:00Z`); // ISO week: Thursday's year and week number
  d.setUTCDate(d.getUTCDate() + 3 - ((d.getUTCDay() + 6) % 7));
  const jan4 = new Date(Date.UTC(d.getUTCFullYear(), 0, 4));
  return `${d.getUTCFullYear()}-W${1 + Math.round(((d - jan4) / 86400000 - 3 + ((jan4.getUTCDay() + 6) % 7)) / 7)}`;
};

// Indices of the first trading day of each period (or every N days from the first usable day).
export function rebalanceIndices(dates, rebalance, start = 0) {
  const out = [];
  if (Number.isInteger(rebalance)) {
    if (rebalance < 1) throw new RangeError("rebalance as a number of days must be at least 1");
    for (let i = start; i < dates.length; i += rebalance) out.push(i);
    return out;
  }
  const every = REBALANCE[rebalance];
  if (!every) throw new RangeError(`rebalance must be weekly, monthly, quarterly or a whole number of trading days; got ${JSON.stringify(rebalance)}`);
  let last = null;
  for (let i = start; i < dates.length; i += 1) {
    const key = periodKey(dates[i], every);
    if (key !== last) { out.push(i); last = key; }
  }
  return out;
}

const finite = (x) => typeof x === "number" && Number.isFinite(x);

function checkInputs({ dates, tickers, prices, signalDates, signals }) {
  if (!Array.isArray(dates) || dates.length < 3) throw new RangeError("prices need at least 3 dates");
  for (let i = 1; i < dates.length; i += 1) if (!(dates[i] > dates[i - 1])) throw new RangeError(`price dates must be strictly increasing; ${dates[i - 1]} then ${dates[i]}`);
  for (let i = 1; i < signalDates.length; i += 1) if (!(signalDates[i] > signalDates[i - 1])) throw new RangeError(`signal dates must be strictly increasing; ${signalDates[i - 1]} then ${signalDates[i]}`);
  if (!tickers.length) throw new RangeError("no ticker appears in both the prices and the signal");
  for (const t of tickers) {
    if (prices[t]?.length !== dates.length) throw new RangeError(`prices for ${t} are not aligned with the price dates`);
    if (signals[t]?.length !== signalDates.length) throw new RangeError(`signal for ${t} is not aligned with the signal dates`);
  }
}

// The latest signal row dated on or before `bound` (binary search), or -1.
function lastRowOnOrBefore(signalDates, bound) {
  let lo = 0, hi = signalDates.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (signalDates[mid] <= bound) { found = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return found;
}

export function runBacktest({ dates, tickers, prices, signalDates, signals, rebalance = "monthly", lagDays = 1, side = "long_short", quantile = 0.2, costBps = 10, minNames = 5, periodsPerYear = 252 }) {
  checkInputs({ dates, tickers, prices, signalDates, signals });
  if (!Number.isInteger(lagDays) || lagDays < 1) throw new RangeError("lag_days must be a whole number of at least 1: a signal cannot trade on the day it becomes known");
  if (!(quantile > 0 && quantile <= 0.5)) throw new RangeError("quantile must be in (0, 0.5]");
  if (!["long_short", "long_only"].includes(side)) throw new RangeError("side must be long_short or long_only");
  if (!(costBps >= 0)) throw new RangeError("cost_bps cannot be negative");

  const n = dates.length;
  const rebal = new Set(rebalanceIndices(dates, rebalance, lagDays));
  let nav = 1;
  let pos = new Map(); // ticker -> signed value (fraction of starting capital, drifts with price)
  const out = { dates: [], returns: [] };
  const used = []; // { trade_date, signal_date }
  let pendingCost = 1; // the first rebalance's cost, charged on the first recorded day
  let started = false, turnoverSum = 0, costSum = 0, rebalances = 0, missingDays = 0, longNames = 0, shortNames = 0;

  for (let t = 1; t < n; t += 1) {
    // 1. Positions earn day t's return.
    if (started) {
      let pnl = 0;
      for (const [tk, v] of pos) {
        const a = prices[tk][t - 1], b = prices[tk][t];
        if (finite(a) && finite(b) && a > 0) { const r = b / a - 1; pnl += v * r; pos.set(tk, v * (1 + r)); }
        else missingDays += 1;
      }
      const before = nav;
      nav += pnl;
      out.dates.push(dates[t]);
      out.returns.push((1 + (before > 0 ? pnl / before : 0)) * pendingCost - 1);
      pendingCost = 1;
    }
    // 2. Rebalance at the close of day t, on signals known lagDays trading days earlier.
    if (!rebal.has(t) || t - lagDays < 0) continue;
    const bound = dates[t - lagDays];
    const row = lastRowOnOrBefore(signalDates, bound);
    if (row < 0) continue;
    if (!(signalDates[row] < dates[t])) throw new Error(`causality violated: a trade on ${dates[t]} would use a signal dated ${signalDates[row]}`);
    const ranked = tickers
      .filter((tk) => finite(signals[tk][row]) && finite(prices[tk][t]) && prices[tk][t] > 0)
      .sort((x, y) => signals[y][row] - signals[x][row] || (x < y ? -1 : 1));
    const k = Math.floor(ranked.length * quantile);
    if (k < 1 || ranked.length < minNames) continue;
    const target = new Map();
    const long = ranked.slice(0, k), short = side === "long_short" ? ranked.slice(-k) : [];
    for (const tk of long) target.set(tk, nav / long.length);
    for (const tk of short) target.set(tk, (target.get(tk) ?? 0) - nav / short.length);
    let traded = 0;
    for (const tk of new Set([...pos.keys(), ...target.keys()])) traded += Math.abs((target.get(tk) ?? 0) - (pos.get(tk) ?? 0));
    const cost = traded * costBps / 1e4;
    nav -= cost;
    const factor = 1 - cost / (nav + cost);
    if (started && out.returns.length) {
      const last = out.returns.length - 1; // cost is charged on the rebalance day's return
      out.returns[last] = (1 + out.returns[last]) * factor - 1;
    } else pendingCost = factor;
    pos = new Map([...target].filter(([, v]) => v !== 0));
    turnoverSum += traded / (nav + cost);
    costSum += cost;
    rebalances += 1; longNames += long.length; shortNames += short.length;
    used.push({ trade_date: dates[t], signal_date: signalDates[row] });
    started = true;
  }
  if (!rebalances) throw new RangeError("no rebalance had enough names with both a known signal and a price; check the dates overlap and min_names");
  return { ...out, used, nav, rebalances, turnoverSum, costSum, missingDays, longNames, shortNames, periodsPerYear };
}

// Plain statistics of a return series, in the same Sharpe convention as the validators (sample sd).
export function describe(result) {
  const r = result.returns;
  const m = r.length;
  const mean = r.reduce((a, b) => a + b, 0) / m;
  const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, m - 1));
  let peak = 1, wealth = 1, maxDd = 0;
  for (const x of r) { wealth *= 1 + x; peak = Math.max(peak, wealth); maxDd = Math.max(maxDd, 1 - wealth / peak); }
  const years = m / result.periodsPerYear;
  const gaps = result.used.map((u) => (Date.parse(u.trade_date) - Date.parse(u.signal_date)) / 86400000);
  return {
    first_date: result.dates[0], last_date: result.dates.at(-1), days: m,
    total_return: wealth - 1,
    annualized_return: years > 0 ? wealth ** (1 / years) - 1 : null,
    annualized_volatility: sd * Math.sqrt(result.periodsPerYear),
    sharpe_annualized: sd > 0 ? (mean / sd) * Math.sqrt(result.periodsPerYear) : null,
    max_drawdown: maxDd,
    rebalances: result.rebalances,
    turnover_per_year: years > 0 ? result.turnoverSum / years : null,
    cost_paid: result.costSum,
    average_long_names: result.longNames / result.rebalances,
    average_short_names: result.shortNames / result.rebalances,
    missing_price_days: result.missingDays,
    causality: { checked_rebalances: result.used.length, min_days_signal_to_trade: Math.min(...gaps), first: result.used.slice(0, 3) },
  };
}
