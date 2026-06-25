#!/bin/bash
set -euo pipefail

python3 --version
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

data_root=data
results_root=results
pixel_workers=30    # CPU workers for the per-pixel Lomb-Scargle step
filament_workers=2  # keep small: each filament worker holds its own CNN/GPU context

# By default, days that already have a results <day>_events.json are skipped.
# Re-analyze them by setting FORCE=1, e.g. FORCE=1 bash analysis.sh

# Analyze every preprocessed day folder found under data_root, one day at a time.
# Layout is nested: data_root/<year>/<month>/<day> = YYYY/YYYYMM/YYYYMMDD.
shopt -s nullglob
for day_dir in "$data_root"/[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/; do
  day=$(basename "$day_dir")
  year="${day:0:4}"
  month="${day:0:6}"
  events_json="${results_root}/${year}/${month}/${day}/${day}_events.json"

  if [[ -z "${FORCE:-}" && -s "$events_json" ]]; then
    echo "=== Skipping day $day: already analyzed ($events_json) ==="
    continue
  fi

  echo "=== Analyzing day: $day ==="

  if ! python3 -m solfilosc.analysis.cli --day "$day" \
        --data-root "$data_root" --results-root "$results_root" \
        --filament-workers "$filament_workers" --pixel-workers "$pixel_workers"; then
    echo "Skipping day $day: analysis failed (e.g. missing ${day}.h5 / ${day}_masks.h5)."
    continue
  fi

  sleep 5
done
