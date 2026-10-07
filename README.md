# agent-search

Is the deflated Sharpe ratio calibrated when an AI agent chooses the trials? Synthetic markets with known
truth; four searchers at equal trial budgets (an adaptive LLM agent, a non-adaptive LLM agent, random
search, a coarse-to-fine grid); every trial recorded in a hash-chained ledger; outcomes scored against the
planted truth and a held-out window the searchers never see.

- `src/market.mjs`: market generator (null, or one planted characteristic).
- `src/grammar.mjs`: the strategy grammar (840 strategies; lookahead not expressible) and the trial runner.
- `src/searchers.mjs`: the four searchers.
- `src/evaluate.mjs`: ledger, deflated Sharpe ratio (counted and effective trials), PBO, held-out Sharpe.
- `vendor/`: pinned copies of canli-backtest-mcp's engine and canli-validation-mcp's ledger (source commit in `vendor/SOURCE_COMMIT`).

`npm test` runs the tests without any API call. Design: see the paper's design notes; the pre-registration will be committed before any full run.
