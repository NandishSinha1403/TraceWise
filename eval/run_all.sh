#!/usr/bin/env bash
# Run all three strategies one after another, resuming automatically after
# free-tier quota stops. Safe to re-run: finished strategies are skipped.
#
#   nohup eval/run_all.sh > results/run_all.log 2>&1 &
#
# On a quota stop it waits 15 minutes and resumes. If a resume makes no
# progress, it waits until 00:10 Pacific time (when daily quotas reset).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

PACE="${PACE:-5}"
K="${K:-4}"

latest_run() {  # newest results folder for a strategy label
  ls -1d "results/$1"_2* 2>/dev/null | sort | tail -1
}

n_ok() {
  [[ -f "$1/requests.csv" ]] || { echo 0; return; }
  python - "$1/requests.csv" <<'EOF'
import csv, sys
print(len({r["question_id"] for r in csv.DictReader(open(sys.argv[1])) if r["ok"] == "True"}))
EOF
}

seconds_until_pacific_reset() {
  python - <<'EOF'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo("America/Los_Angeles"))
nxt = (now + timedelta(days=1)).replace(hour=0, minute=10, second=0, microsecond=0)
print(int((nxt - now).total_seconds()))
EOF
}

total=$(grep -c . eval/questions.jsonl)

for spec in "full_history:" "sliding_window:--k $K" "rolling_summary:"; do
  strategy="${spec%%:*}"; extra="${spec#*:}"
  label="$strategy"; [[ "$strategy" == sliding_window ]] && label="sliding_window_k$K"
  run="$(latest_run "$label")"

  if [[ -n "$run" && "$(n_ok "$run")" -ge "$total" ]]; then
    echo "$(date '+%F %T') $label already complete: $run"; continue
  fi

  if [[ -z "$run" ]]; then
    echo "$(date '+%F %T') starting $label"
    python -m eval.run_eval --strategy "$strategy" $extra --pace "$PACE" && continue
    run="$(latest_run "$label")"
  fi

  while [[ "$(n_ok "$run")" -lt "$total" ]]; do
    before="$(n_ok "$run")"
    sleep 900
    echo "$(date '+%F %T') resuming $run ($before/$total done)"
    python -m eval.run_eval --resume "$run" --pace "$PACE" && break
    if [[ "$(n_ok "$run")" -eq "$before" ]]; then
      wait_s="$(seconds_until_pacific_reset)"
      echo "$(date '+%F %T') no progress; sleeping ${wait_s}s until the Pacific-time quota reset"
      sleep "$wait_s"
    fi
  done
  echo "$(date '+%F %T') $label complete: $run"
done

echo "$(date '+%F %T') ALL DONE"
python -m eval.summarize results/
