#!/bin/bash
set -euo pipefail

python3 --version
ncores=64
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
days=(
    20140101
    20140102
    20140103
    20140104
    20140105
    20140106
    20140107
    20140108
    20140109
    20140110
    20140111
    20140112
    20140113
    20140114
    20140115
    20140116
    20140117
    20140118
    20140119
    20140120
    20140121
    20140122
    20140123
    20140124
    20140125
    20140126
    20140127
    20140128
    20140129
    20140130
    20140131
)

for day in "${days[@]}"; do
  echo "=== Processing day: $day ==="

  python3 -m solfilosc.data_processing.preprocess_data 2014 201401 "$day" "$ncores" /mnt/HPC.servers/codony7/GONG_DATA/2014/201401/"$day"
  sleep 5

  python3 -m solfilosc.data_processing.create_data_cube_file 2014 201401 "$day" "$ncores"
  sleep 5

  python3 segment_filaments.py "$day"
done
