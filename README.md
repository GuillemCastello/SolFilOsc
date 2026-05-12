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

The preprocessing pipeline expects raw daily `.fits.fz` files under:

```text
/mnt/scratch/guillem/<month>/<day>/
```

For one day:

```bash
python3 -m solfilosc.data_processing.preprocess_data 2014 201401 20140102 64
python3 -m solfilosc.data_processing.create_data_cube_file 2014 201401 20140102 64
python3 -m solfilosc.data_processing.post_process_data_cube 2014 201401 20140102
```

These stages:

1. Filter raw files by size, observing window, cadence density, and sharpness.
2. Correct limb darkening and smooth background structure.
3. Derotate images to a common reference time and save `data/<day>/updated/<day>_data.h5`.
4. Remove bad frames, adjust telescope-change intensity offsets, zero pixels outside the disk, and save `data/<day>/updated/<day>_data_modified.h5`.

The example batch script runs these stages for the listed days, then calls the local segmentation wrapper `segment_filaments.py` if you provide one:

```bash
bash preprocessing.sh
```

## Filament masks

The CNN analysis requires a mask file:

```text
data/<day>/updated/<day>_masks.h5
```

The segmentation code/models are not part of this repository. `FilamentSeg/` is ignored by git and should be installed or copied from the original FilamentSeg source repository. After producing masks, place the `.h5` file in the corresponding `data/<day>/updated/` folder before running the oscillation analysis.

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

- The default preprocessing paths are currently hard-coded for `/mnt/scratch/guillem/<month>/<day>/`.
- The pipeline can be memory- and CPU-heavy. Reduce `ncores`, `--filament-workers`, or `--pixel-workers` if the machine becomes unstable.
- TensorFlow is forced to CPU in the CNN module, matching the original notebook behavior.
- CP calibration is expensive but cached per day under `results/<day>/`.
- `data/`, `results/`, and `FilamentSeg/` are ignored because they are large or externally sourced.
