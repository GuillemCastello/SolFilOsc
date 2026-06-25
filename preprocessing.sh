#!/usr/bin/env bash
set -euo pipefail

python3 --version

ncores="${NCORES:-30}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Usage:
#   ./run_pipeline.sh 2014 02
#
# Defaults:
#   year=2014
#   month_number=02

year="${1:-2014}"
month_number="${2:-02}"

# Normalize month number so both "2" and "02" work
month_number="$(printf "%02d" "$((10#$month_number))")"

# Naming convention:
# year  = YYYY
# month = YYYYMM
# day   = YYYYMMDD
month="${year}${month_number}"

# Number of days in the selected month
last_day="$(date -d "${year}-${month_number}-01 +1 month -1 day" +%d)"

base_dir="/mnt/HPC.servers/codony7/GONG_DATA"

project_data_dir="${DATA_DIR:-$PWD/data}"
# True (exit 0) if this day is fully processed in data/<year>/<month>/<day>/
day_already_done() {
  local d="$1"
  local dir="${project_data_dir}/${year}/${month}/${d}"
  [[ -s "${dir}/${d}.h5" && -s "${dir}/${d}_masks.h5" ]]
}

for day_number in $(seq -w 1 "$last_day"); do
  day="${month}${day_number}"
  raw_dir="${base_dir}/${year}/${month}/${day}"

  if day_already_done "$day"; then
    echo "=== Skipping day $day: already present in data/${year}/${month}/${day}/ ==="
    continue
  fi

  echo "=== Processing day: $day ==="

  if ! python3 -m solfilosc.data_processing.preprocess_data \
      "$year" "$month" "$day" "$ncores" "$raw_dir"; then
    echo "Skipping day $day: raw-data folder missing or empty."
    continue
  fi

  sleep 5

  python3 -m solfilosc.data_processing.create_data_cube_file \
      "$year" "$month" "$day" "$ncores"

  sleep 5

  python3 segment_filaments.py "$day"
done