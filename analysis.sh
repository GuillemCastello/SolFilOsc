#!/bin/bash
set -euo pipefail

python3 --version
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

data_root=data
pixel_workers=30    # CPU workers for the per-pixel Lomb-Scargle step
filament_workers=2  # keep small: each filament worker holds its own CNN/GPU context

# Analyze every preprocessed day folder found under data_root, one day at a time.
shopt -s nullglob
for day_dir in "$data_root"/*/; do
  day=$(basename "$day_dir")
  echo "=== Analyzing day: $day ==="

  if ! python3 -m solfilosc.analysis.cli --day "$day" --data-root "$data_root" \
        --filament-workers "$filament_workers" --pixel-workers "$pixel_workers"; then
    echo "Skipping day $day: analysis failed (e.g. missing ${day}.h5 / ${day}_masks.h5)."
    continue
  fi

  sleep 5
done
