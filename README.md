# SolFilOsc

SolFilOsc prepares solar H-alpha image sequences and analyzes filament oscillations with a CNN noise model plus conformal-prediction thresholds.

## Repository layout

- `src/solfilosc/data_processing/`: preprocessing pipeline for raw `.fits.fz` files.
- `src/solfilosc/mapping/`: solar-map utilities used during image derotation.
- `src/solfilosc/analysis/`: CNN oscillation-analysis pipeline split from the original notebook.
- `notebooks/analysis_with_CNN.ipynb`: thin notebook launcher for interactive analysis runs.
- `CNN/`: CNN weights and saved models used by the analysis stage.
- `preprocessing.sh`: example batch preprocessing script.
- `docs/project_reorganization.tex`: technical report of the restructuring changes.

Generated data and results are intentionally not tracked:

- `data/`
- `results/`
- `FilamentSeg/`

## Installation

Create an environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you do not install the package, run commands from the repository root with:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Data pipeline

The preprocessing pipeline expects raw daily `.fits.fz` files under the repository-local data folder:

```text
data/raw/<month>/<day>/
```

For example, for `month=201401` and `day=20140102`, place the raw files in:

```text
data/raw/201401/20140102/*.fits.fz
```

If this folder does not exist, or if it contains no `.fits.fz` files, the first preprocessing stage stops with an explicit message telling the user to create the folder or pass a custom raw-data path. A custom raw-data path can be supplied as the optional fifth argument to `preprocess_data`.

For one day:

```bash
python3 -m solfilosc.data_processing.preprocess_data 2014 201401 20140102 64
python3 -m solfilosc.data_processing.create_data_cube_file 2014 201401 20140102 64
```

For a non-default raw-data location:

```bash
python3 -m solfilosc.data_processing.preprocess_data 2014 201401 20140102 64 /path/to/raw/fits/files
```

### Preprocessing command arguments

The scripts keep the original positional command-line interface.

`preprocess_data`:

```text
sys.argv[1] = year
sys.argv[2] = month
sys.argv[3] = day
sys.argv[4] = n_proc
sys.argv[5] = raw_data_dir, optional
```

- `year`: observation year, kept for compatibility with the old interface.
- `month`: month label used by the default raw-data path, usually `YYYYMM`.
- `day`: day label, usually `YYYYMMDD`; used for input lookup and output folder names.
- `n_proc`: number of multiprocessing workers for limb-darkening/background correction.
- `raw_data_dir`: optional directory containing the raw `.fits.fz` files. If omitted, the default is `data/raw/<month>/<day>/`.

`create_data_cube_file`:

```text
sys.argv[1] = year
sys.argv[2] = month
sys.argv[3] = day
sys.argv[4] = n_threads
```

- Reads `data/<day>/*.fits`.
- Derotates the corrected FITS files into an in-memory data cube.
- Post-processes that cube directly (removes bad frames, adjusts telescope-change intensity offsets, zeros pixels outside the disk) — no intermediate `.h5` is written.
- Writes the final cube `data/<day>/<day>.h5` once, then deletes the intermediate `*_updated.fits` so the day folder keeps only `<day>.h5`.
- `year` and `month` are currently kept only for interface consistency; `day` and `n_threads` control the run.

### Raw FITS input assumptions

The first stage expects compressed FITS files matching `*.fits.fz`. The current GONG-oriented parser assumes:

- the science image and relevant header are in HDU 1;
- the filename encodes the timestamp at `file[-16:-10]` as `HHMMSS`;
- the filename encodes the observatory/site letter at `file[-10]`;
- the FITS header contains `SHARPNSS`, used by the sharpness filter.

These stages:

1. Filter raw files by size, observing window, cadence density, and sharpness.
2. Correct limb darkening and smooth background structure.
3. Derotate images to a common reference time, post-process the resulting cube in memory (remove bad frames, adjust telescope-change intensity offsets, zero pixels outside the disk), save the final cube `data/<day>/<day>.h5`, and delete the intermediate files.

The example batch script runs these stages for the listed days, then calls the local segmentation wrapper `segment_filaments.py` if you provide one:

```bash
bash preprocessing.sh
```

## Filament masks

The CNN analysis requires a mask file:

```text
data/<day>/<day>_masks.h5
```

This file holds two datasets, both shaped like the data cube (`T, H, W`), `uint8`:

- `masks`: binary full-disk mask (1 = any filament, 0 = background).
- `masks_by_type`: per-type full-disk mask (QRF = 1, IRF = 2, ARF = 3, background = 0).

The segmentation code/models are not part of this repository. `FilamentSeg/` is ignored by git and should be installed or copied from the original FilamentSeg source repository. After producing masks, place the `.h5` file in the corresponding `data/<day>/` folder before running the oscillation analysis.

FilamentSeg related references:

- Public project/dataset repository: [antonio-reche/SWEFil on Hugging Face](https://huggingface.co/datasets/antonio-reche/SWEFil).
- Author project page: [Solar filament detection, classification, and tracking with deep learning](https://antonioreche.me/).
- Note: the local FilamentSeg copy does not include a public upstream code URL. If the authors provide a separate code repository, use that as the source for `FilamentSeg/`.
- Citation:

```bibtex
@inproceedings{reche2024solar,
  title     = {Solar filament detection, classification, and tracking with deep learning},
  author    = {Reche, Antonio and Cid, Consuelo},
  booktitle = {Proceedings of SPAICE2024: The First Joint European Space Agency/IAA Conference on AI in and for Space},
  pages     = {69--74},
  year      = {2024}
}
```

## CNN analysis

Analyze all filaments for one day:

```bash
solfilosc-analysis --day 20140102
```

Analyze one filament index:

```bash
solfilosc-analysis --day 20140102 --filament-index 0
```

Analyze every available `data/<day>/` folder:

```bash
solfilosc-analysis
```

Useful options:

```bash
solfilosc-analysis \
  --day 20140102 \
  --data-root data \
  --results-root results \
  --cnn-weights CNN/BestFit/BestFitWeights.h5 \
  --filament-workers 64 \
  --pixel-workers 64
```

Outputs are written under `results/<day>/`, including CP calibration caches, diagnostic plots, per-scale component tables, period-family summaries, and event tables.

## Analysis modules

- `constants.py`: frequency grid, detection band, CP delta, and default worker counts.
- `cnn.py`: CNN architecture, weight loading, scaler reconstruction, PSD computation, and noise-parameter prediction.
- `cp_calibration.py`: daily conformal-prediction calibration and cache creation.
- `degradation.py`: image block averaging, mask coverage, scale selection, and null transforms.
- `roi.py`: mask/ROI selection, bounding boxes, mask expansion, and weighted period helpers.
- `detection.py`: Lomb-Scargle PSD analysis, CP peak detection, period clustering, and connected-component extraction.
- `events.py`: spatial clustering of detections into events within period families.
- `plotting.py`: full-disk, scale, period-family, and CP diagnostic plots.
- `writers.py`: CSV/JSON serialization helpers.
- `pipeline.py`: end-to-end day/filament drivers.
- `cli.py`: lightweight `solfilosc-analysis` command-line entry point.

## Notes before running

- The default raw-data path is `data/raw/<month>/<day>/`. Pass the optional fifth argument to `preprocess_data` if your `.fits.fz` files are somewhere else.
- The pipeline can be memory- and CPU-heavy. Reduce `ncores`, `--filament-workers`, or `--pixel-workers` if the machine becomes unstable.
- TensorFlow is forced to CPU in the CNN module, matching the original notebook behavior.
- CP calibration is expensive but cached per day under `results/<day>/`.
- `data/`, `results/`, and `FilamentSeg/` are ignored because they are large or externally sourced.
-If a day is reprocessed, or if masks/model weights/calibration settings change, delete results/<day>/ before rerunning the analysis. The CP calibration cache is stored under results/<day>/ and is reused when present.
