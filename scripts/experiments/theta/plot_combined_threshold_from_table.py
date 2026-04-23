import argparse
from pathlib import Path

from plot_combined_threshold_results import _load_table_rows, _plot_combined


DEFAULT_TABLE = Path("scripts/experiments/theta/results/threshold_sweep_combined.csv")
DEFAULT_OUT = Path("scripts/experiments/theta/results/threshold_sweep_combined.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the combined theta sweep plot from an editable combined CSV table."
    )
    parser.add_argument(
        "table",
        nargs="?",
        default=str(DEFAULT_TABLE),
        help=f"Editable combined CSV table. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output PNG path. Default: {DEFAULT_OUT}",
    )
    args = parser.parse_args()

    rows = _load_table_rows(Path(args.table))
    _plot_combined(rows, Path(args.out))


if __name__ == "__main__":
    main()
