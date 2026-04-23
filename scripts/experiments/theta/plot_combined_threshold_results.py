import argparse
import json
import math
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("scripts/experiments/theta/results")
PIL_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PIL_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_pil_font(size: int, bold: bool = False):
    from PIL import ImageFont

    return ImageFont.truetype(PIL_FONT_BOLD if bold else PIL_FONT_REGULAR, size=size)


def _text_size(draw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for json_path in sorted(results_dir.glob("threshold_sweep_*.json")):
        payload = json.loads(json_path.read_text())
        for row in payload.get("results", []):
            row = dict(row)
            row["_source"] = json_path.name
            rows.append(row)
    if not rows:
        raise SystemExit(f"No theta sweep results found in {results_dir}")
    return rows


def _draw_panel(
    draw,
    box,
    title: str,
    thetas: list[float],
    series: list[dict],
    y_label: str,
    fonts: dict[str, object],
    y_min: float = 0.0,
    y_max: float = 1.05,
    y_ticks: list[float] | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="black", width=2)

    tick_values = y_ticks if y_ticks is not None else [y_min + (y_max - y_min) * frac for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for tick in tick_values:
        y = bottom - ((tick - y_min) / (y_max - y_min)) * (bottom - top)
        draw.line((left, y, right, y), fill="#e6e6e6", width=1)
        tick_label = f"{tick:.2f}" if y_max <= 1.05 else f"{int(tick)}"
        tick_w, tick_h = _text_size(draw, tick_label, fonts["tick"])
        draw.text((left - tick_w - 16, y - tick_h / 2), tick_label, fill="black", font=fonts["tick"])

    if len(thetas) == 1:
        x_lookup = {thetas[0]: (left + right) / 2}
    else:
        theta_min = min(thetas)
        theta_max = max(thetas)
        x_lookup = {
            theta: left + ((theta - theta_min) / (theta_max - theta_min)) * (right - left)
            for theta in thetas
        }

    for theta in thetas:
        x = x_lookup[theta]
        draw.line((x, top, x, bottom), fill="#eeeeee", width=1)
        draw.line((x, bottom, x, bottom + 7), fill="black", width=1)
        theta_label = f"{theta:g}"
        theta_w, _ = _text_size(draw, theta_label, fonts["tick"])
        draw.text((x - theta_w / 2, bottom + 12), theta_label, fill="black", font=fonts["tick"])

    for spec in series:
        points = []
        for theta, value in spec["points"]:
            x = x_lookup[theta]
            y = bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=spec["color"], width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=spec["color"])

    title_w, title_h = _text_size(draw, title, fonts["subplot_title"])
    title_x = left + (right - left - title_w) / 2
    draw.text((title_x, top - title_h - 18), title, fill="black", font=fonts["subplot_title"])
    x_label = "Theta"
    x_label_w, _ = _text_size(draw, x_label, fonts["axis_label"])
    x_label_x = left + (right - left - x_label_w) / 2
    draw.text((x_label_x, bottom + 42), x_label, fill="black", font=fonts["axis_label"])


def _draw_vertical_text(img, text: str, x: int, y: int, font) -> None:
    from PIL import Image, ImageDraw

    tmp = Image.new("RGBA", (320, 120), (255, 255, 255, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    tmp_draw.text((0, 0), text, fill="black", font=font)
    rotated = tmp.rotate(90, expand=True)
    img.alpha_composite(rotated, (x, y))


def _plot_with_pillow(rows: list[dict], out_path: Path, categories: list[str], thetas: list[float]) -> None:
    from PIL import Image, ImageDraw

    by_category: dict[str, list[dict]] = {category: [] for category in categories}
    for row in rows:
        by_category[row["category"]].append(row)
    for category_rows in by_category.values():
        category_rows.sort(key=lambda r: float(r["theta"]))

    img = Image.new("RGBA", (1380, 1480), "white")
    draw = ImageDraw.Draw(img)
    fonts = {
        "title": _load_pil_font(40, bold=False),
        "subplot_title": _load_pil_font(30, bold=False),
        "axis_label": _load_pil_font(26, bold=False),
        "tick": _load_pil_font(22, bold=False),
        "legend": _load_pil_font(22, bold=False),
    }

    margin_left = 105
    margin_right = 40
    margin_top = 110
    margin_bottom = 155
    row_gap = 95
    panel_width = 1380 - margin_left - margin_right
    panel_height = (1480 - margin_top - margin_bottom - 2 * row_gap) // 3

    boxes = []
    for idx in range(3):
        top = margin_top + idx * (panel_height + row_gap)
        boxes.append((margin_left, top, margin_left + panel_width, top + panel_height))

    palette = ["#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#c62828", "#00838f"]
    color_by_category = {category: palette[idx % len(palette)] for idx, category in enumerate(categories)}

    success_series = [
        {
            "color": color_by_category[category],
            "points": [(row["theta"], row["success_rate"]) for row in by_category[category]],
        }
        for category in categories
    ]
    kept_series = [
        {
            "color": color_by_category[category],
            "points": [
                (
                    row["theta"],
                    (row["relations_kept"] / row["relations_total"]) if row["relations_total"] else 0.0,
                )
                for row in by_category[category]
            ],
        }
        for category in categories
    ]
    avg_kept_series = [
        {
            "color": color_by_category[category],
            "points": [
                (
                    row["theta"],
                    row["avg_relations_per_object"] + row["avg_relations_per_keyword"],
                )
                for row in by_category[category]
            ],
        }
        for category in categories
    ]

    avg_kept_max = max([point[1] for spec in avg_kept_series for point in spec["points"]] + [1.0])
    avg_kept_y_max = max(150.0, math.ceil(avg_kept_max / 25.0) * 25.0)
    avg_kept_ticks = list(range(0, int(avg_kept_y_max) + 1, 25))

    _draw_panel(
        draw,
        boxes[0],
        "Task Success Rate",
        thetas,
        success_series,
        "Success rate",
        fonts,
        y_min=0.0,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    _draw_panel(
        draw,
        boxes[1],
        "Average Kept Relations (Objects + Keywords)",
        thetas,
        avg_kept_series,
        "# relations",
        fonts,
        y_min=0.0,
        y_max=avg_kept_y_max,
        y_ticks=avg_kept_ticks,
    )
    _draw_panel(
        draw,
        boxes[2],
        "Overall Kept Ratio",
        thetas,
        kept_series,
        "Overall kept/total",
        fonts,
        y_min=0.0,
        y_max=1.0,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )

    for box, y_label in zip(
        boxes,
        ["Success rate", "# relations", "Overall kept/total"],
    ):
        left, top, _, bottom = box
        _draw_vertical_text(img, y_label, left - 95, int(top + (bottom - top) / 2 - 95), fonts["axis_label"])

    draw.text((margin_left, 24), "Theta Sensitivity Across Evaluation Categories", fill="black", font=fonts["title"])
    legend_x = margin_left + 40
    legend_y = 1420
    for category in categories:
        color = color_by_category[category]
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 12), fill=color)
        draw.text((legend_x + 26, legend_y - 10), category.replace("_", " "), fill="black", font=fonts["legend"])
        legend_x += 280

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing threshold_sweep_*.json files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to <results-dir>/threshold_sweep_combined.png",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = _load_rows(results_dir)

    categories = sorted({row["category"] for row in rows})
    thetas = sorted({float(row["theta"]) for row in rows})

    by_category: dict[str, list[dict]] = {category: [] for category in categories}
    for row in rows:
        by_category[row["category"]].append(row)
    for category_rows in by_category.values():
        category_rows.sort(key=lambda r: float(r["theta"]))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        out_path = Path(args.out) if args.out else results_dir / "threshold_sweep_combined.png"
        _plot_with_pillow(rows, out_path, categories, thetas)
        return

    avg_kept_max = max(
        [
            row["avg_relations_per_object"] + row["avg_relations_per_keyword"]
            for category in categories
            for row in by_category[category]
        ]
        + [1.0]
    )
    avg_kept_y_max = max(150.0, math.ceil(avg_kept_max / 25.0) * 25.0)
    avg_kept_ticks = list(range(0, int(avg_kept_y_max) + 1, 25))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.5, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.45, 1.0]},
    )

    ax = axes[0]
    for category in categories:
        category_rows = by_category[category]
        ax.plot(
            [row["theta"] for row in category_rows],
            [row["success_rate"] for row in category_rows],
            marker="o",
            linewidth=2,
            label=category.replace("_", " "),
        )
    ax.set_ylabel("Success rate", fontsize=24, fontweight="bold", rotation=90, labelpad=18)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_xlabel("Theta", fontsize=24, fontweight="bold")
    ax.set_title("Task Success Rate", fontsize=28, fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[1]
    for category in categories:
        category_rows = by_category[category]
        ax.plot(
            [row["theta"] for row in category_rows],
            [
                row["avg_relations_per_object"] + row["avg_relations_per_keyword"]
                for row in category_rows
            ],
            marker="o",
            linewidth=2.5,
            label=category.replace("_", " "),
        )
    ax.set_title("Average Kept Relations (Objects + Keywords)", fontsize=28, fontweight="bold", pad=12)
    ax.set_ylabel("# relations", fontsize=24, fontweight="bold", rotation=90, labelpad=18)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_xlabel("Theta", fontsize=24, fontweight="bold")
    ax.set_ylim(0.0, avg_kept_y_max)
    ax.set_yticks(avg_kept_ticks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[2]
    for category in categories:
        category_rows = by_category[category]
        ax.plot(
            [row["theta"] for row in category_rows],
            [
                row["relations_kept"] / row["relations_total"] if row["relations_total"] else 0.0
                for row in category_rows
            ],
            marker="o",
            linewidth=2,
            label=category.replace("_", " "),
        )
    ax.set_title("Overall Kept Ratio", fontsize=28, fontweight="bold", pad=12)
    ax.set_xlabel("Theta", fontsize=24, fontweight="bold")
    ax.set_ylabel("Overall kept/total", fontsize=24, fontweight="bold", rotation=90, labelpad=18)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    for ax in axes:
        ax.set_xticks(thetas)
        ax.tick_params(axis="both", labelsize=21)

    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    legend = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=21)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.suptitle("Theta Sensitivity Across Evaluation Categories", fontsize=30, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.17, hspace=0.42)

    out_path = Path(args.out) if args.out else results_dir / "threshold_sweep_combined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
