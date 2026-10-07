// Scores PREREG.json's predictions from runs/null.jsonl and runs/alt.jsonl, without hand input.
import { existsSync, readFileSync } from "node:fs";

const load = (f) => (existsSync(f) ? readFileSync(f, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l)) : []);
const nul = load("runs/null.jsonl"), alt = load("runs/alt.jsonl");
const ok = (r) => !r.error;
const tail = (k, n) => { let p = 0, c = 0; for (let i = 0; i <= n; i += 1) { c = i ? c + Math.log(n - i + 1) - Math.log(i) : 0; if (i >= k) p += Math.exp(c - n * Math.LN2); } return n ? p : null; };
// Pair agent rows with the random row on the same market and budget.
function pairs(rows, budget) {
  const rnd = new Map(rows.filter((r) => r.searcher === "random" && r.budget === budget && ok(r)).map((r) => [r.seed, r]));
  return rows.filter((r) => r.searcher === "agent" && r.budget === budget && ok(r) && rnd.has(r.seed)).map((r) => [r, rnd.get(r.seed)]);
}
const mcnemar = (ps, f) => { const b = ps.filter(([a, r]) => f(a) && !f(r)).length, c = ps.filter(([a, r]) => !f(a) && f(r)).length; return { only_agent: b, only_random: c, p: tail(b, b + c) }; };
const out = {};
const p1 = mcnemar(pairs(nul, 100), (r) => r.certified_counted);
out.P1 = { ...p1, held: p1.p !== null && p1.p < 0.05 && p1.only_agent > p1.only_random };
const cells = {};
for (const r of nul.filter(ok)) { const k = `${r.searcher}|${r.model ?? "-"}|${r.budget}`; (cells[k] ??= []).push(r.holdout_reject); }
const rates = Object.fromEntries(Object.entries(cells).map(([k, v]) => [k, { n: v.length, rate: v.filter(Boolean).length / v.length }]));
out.P2 = { rates, held: Object.values(rates).every((c) => c.rate <= 0.07) };
const ps3 = pairs(nul, 100); const below = ps3.filter(([a, r]) => a.n_eff_ratio < r.n_eff_ratio).length, ties = ps3.filter(([a, r]) => a.n_eff_ratio === r.n_eff_ratio).length;
const p3 = tail(below, ps3.length - ties);
out.P3 = { below, of: ps3.length - ties, p: p3, held: p3 !== null && p3 < 0.05 && below > (ps3.length - ties) / 2 };
const oc = nul.filter((r) => r.searcher === "agent" && ok(r) && Number.isFinite(r.overclaim)).map((r) => r.overclaim);
const m = oc.reduce((a, b) => a + b, 0) / (oc.length || 1), sd = Math.sqrt(oc.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, oc.length - 1));
const t = sd > 0 ? (m / sd) * Math.sqrt(oc.length) : 0;
out.P4 = { n: oc.length, mean_overclaim: m, t, held: oc.length > 1 && t > 1.6448536269514722 };
const p5 = mcnemar(pairs(alt, 100), (r) => r.reported_is_planted);
out.P5 = { ...p5, held: p5.p !== null && p5.p < 0.05 && p5.only_agent > p5.only_random };
console.log(JSON.stringify(out, null, 1));
