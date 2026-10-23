#!/usr/bin/env bash
# Runs every experiment of Section 5 in sequence; logs go to outputs/logs/.
# H and gamma are tuned on the dev split first and then fixed for the test runs (Appendix B).
# Usage: bash scripts/run_all.sh [workers] [replays]
#        FROM=crosswoz_main bash scripts/run_all.sh ...   # resume: skip the steps before FROM
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}      # interpreter; override with PY=/path/to/python
W=${1:-32}
R=${2:-40}
RS=${RS:-5}          # replays for ablations and sweeps
mkdir -p outputs/logs
FROM=${FROM:-}
run() { local name=$1; shift
  if [ -n "$FROM" ]; then [ "$name" = "$FROM" ] && FROM= || { echo "[skip] $name"; return; }; fi
  echo "[$(date '+%F %T')] start $name"; "$PY" scripts/run.py "$@" --workers "$W" > "outputs/logs/$name.log" 2>&1; echo "[$(date '+%F %T')] done  $name"; }

for ds in crosswoz multiwoz; do
  run "${ds}_runs"    runs    --dataset $ds
  run "${ds}_tune"    tune    --dataset $ds --replays "$RS"
  run "${ds}_main"    main    --dataset $ds --replays "$R"
  run "${ds}_reprice" reprice --dataset $ds
done
run crosswoz_tune_budget  tune --dataset crosswoz --param budget  --replays "$RS"
run crosswoz_tune_horizon tune --dataset crosswoz --param horizon --replays "$RS"
run crosswoz_tune_churn   tune --dataset crosswoz --param churn   --replays "$RS"
run crosswoz_ablation      ablation --dataset crosswoz --replays "$RS"
run crosswoz_sweep_horizon sweep    --dataset crosswoz --param horizon    --replays "$RS"
run crosswoz_sweep_budget  sweep    --dataset crosswoz --param budget     --replays "$RS"
run crosswoz_sweep_oracle  sweep    --dataset crosswoz --param oracle_mix --replays "$RS"
run crosswoz_churn         churn    --dataset crosswoz --replays "$RS"
run crosswoz_latency       latency  --dataset crosswoz
echo "[$(date '+%F %T')] all done"
