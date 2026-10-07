#!/bin/zsh
# Daily driver for the registered runs, inside OpenAI's free small-model pool (2.5M tokens/day; we use 2.2M).
# Waits until the pre-registration commit is public on GitHub, then from 2026-10-09 runs after each
# 00:30 UTC reset: null markets first, then alternative, for each model, about 1.1M tokens per model per day.
cd ~/agent-search
PREREG=$(git log --format=%H -1 -- PREREG.json)
until git ls-remote origin 2>/dev/null | grep -q . && git fetch -q origin 2>/dev/null && git merge-base --is-ancestor $PREREG origin/main 2>/dev/null; do
  echo "waiting for the pre-registration commit to be public $(date -u +%FT%TZ)"; sleep 1800; done
echo "PREREG_PUBLIC $(date -u +%FT%TZ)"
node src/run.mjs --kind baseline-null 2>&1 | tail -1
node src/run.mjs --kind baseline-alt 2>&1 | tail -1
last=""
while true; do
  day=$(date -u +%F)
  if [[ "$day" < "2026-10-09" || "$day" == "$last" || $(date -u +%H%M) -lt 0030 ]]; then sleep 600; continue; fi
  echo "DAY $day $(date -u +%T)"
  for m in gpt-5-mini gpt-5.4-mini; do
    out=$(node src/run.mjs --kind null --model $m --cap 1100000 2>&1 | tail -1); echo "$m null: $out"
    if [[ "$out" == DONE* ]]; then
      used=$(echo "$out" | grep -o '[0-9]* tokens' | grep -o '[0-9]*'); left=$((1100000 - ${used:-0}))
      (( left > 100000 )) && echo "$m alt: $(node src/run.mjs --kind alt --model $m --cap $left 2>&1 | tail -1)"
    fi
  done
  node src/score.mjs > runs/score-$day.json 2>&1
  last=$day
  if [[ $(grep -c '"searcher":"agent"' runs/alt.jsonl 2>/dev/null) -ge 400 ]]; then echo ALL_DONE; break; fi
done
