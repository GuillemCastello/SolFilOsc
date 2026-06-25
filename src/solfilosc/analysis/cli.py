"""Command-line entry point for the analysis pipeline."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the solar-filament oscillation analysis pipeline.")
    parser.add_argument("--day", help="Day to analyze, e.g. 20140102. If omitted, all data/<year>/<month>/<day>/ folders are processed.")
    parser.add_argument("--filament-index", type=int, help="Analyze one filament index instead of every filament for the day.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--cnn-weights", default="CNN/BestFit/BestFitWeights.h5")
    parser.add_argument("--cp-cache", default=None, help="Optional existing CP cache for --filament-index runs.")
    parser.add_argument("--filament-workers", type=int, default=64)
    parser.add_argument("--pixel-workers", type=int, default=64)
    parser.add_argument(
        "--plot-period-families", action="store_true", default=False,
        help="Also render per-family spatial-map plots (off by default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from .pipeline import run_all_days, run_day, run_filament, write_day_outputs

    if args.day and args.filament_index is not None:
        result = run_filament(
            args.day,
            args.filament_index,
            data_root=args.data_root,
            results_root=args.results_root,
            cnn_weights_path=args.cnn_weights,
            cp_cache_path=args.cp_cache,
            n_pixel_workers=args.pixel_workers,
            plot_period_families=args.plot_period_families,
        )
        write_day_outputs(args.day, args.results_root, [result])
    elif args.day:
        run_day(
            args.day,
            data_root=args.data_root,
            results_root=args.results_root,
            cnn_weights_path=args.cnn_weights,
            n_filament_workers=args.filament_workers,
            n_pixel_workers=args.pixel_workers,
            plot_period_families=args.plot_period_families,
        )
    else:
        run_all_days(
            data_root=args.data_root,
            results_root=args.results_root,
            cnn_weights_path=args.cnn_weights,
            n_filament_workers=args.filament_workers,
            n_pixel_workers=args.pixel_workers,
            plot_period_families=args.plot_period_families,
        )


if __name__ == "__main__":
    main()
