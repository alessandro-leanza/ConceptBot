import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("scripts/experiments/theta/results")
DEFAULT_TABLE_NAME = "threshold_sweep_combined.csv"
DEFAULT_PLOT_NAME = "threshold_sweep_combined.png"
DEFAULT_SUPPLEMENTAL_PLOT_NAME = "threshold_sweep_combined_zero_relations.png"
PIL_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PIL_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_FIELDS = [
    "category",
    "theta",
    "success_rate",
    "avg_kept_relations",
    "avg_kept_relations_min_for_theta",
    "avg_kept_relations_max_for_theta",
    "overall_kept_ratio",
    "overall_kept_ratio_min_for_theta",
    "overall_kept_ratio_max_for_theta",
    "avg_relations_per_object",
    "avg_relations_per_object_min_for_theta",
    "avg_relations_per_object_max_for_theta",
    "avg_relations_per_keyword",
    "avg_relations_per_keyword_min_for_theta",
    "avg_relations_per_keyword_max_for_theta",
    "ope_objects",
    "ope_zero_relation_objects",
    "ope_zero_relation_object_ratio",
    "ope_zero_relation_object_names",
    "urp_keywords",
    "urp_zero_relation_keywords",
    "urp_zero_relation_keyword_ratio",
    "urp_zero_relation_keyword_names",
    "zero_relation_queries",
    "total_relation_queries",
    "zero_relation_query_ratio",
    "zero_relation_terms",
    "relations_kept",
    "relations_total",
    "ope_relations_kept",
    "ope_relations_total",
    "urp_relations_kept",
    "urp_relations_total",
    "_source",
]


def _load_pil_font(size: int, bold: bool = False):
    from PIL import ImageFont

    return ImageFont.truetype(PIL_FONT_BOLD if bold else PIL_FONT_REGULAR, size=size)


def _text_size(draw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for json_path in sorted(results_dir.glob("threshold_sweep_*.json")):
        if ".pre_zero_terms." in json_path.name:
            continue
        payload = json.loads(json_path.read_text())
        for row in payload.get("results", []):
            row = dict(row)
            row["_source"] = json_path.name
            rows.append(row)
    if not rows:
        raise SystemExit(f"No theta sweep results found in {results_dir}")
    return rows


def _to_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _to_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _normalize_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        new_row = dict(row)
        new_row["theta"] = _to_float(new_row, "theta")
        new_row["success_rate"] = _to_float(new_row, "success_rate")
        new_row["avg_relations_per_object"] = _to_float(new_row, "avg_relations_per_object")
        new_row["avg_relations_per_keyword"] = _to_float(new_row, "avg_relations_per_keyword")
        new_row["relations_kept"] = _to_int(new_row, "relations_kept")
        new_row["relations_total"] = _to_int(new_row, "relations_total")
        new_row["ope_objects"] = _to_int(new_row, "ope_objects")
        new_row["ope_zero_relation_objects"] = _to_int(new_row, "ope_zero_relation_objects")
        new_row["ope_zero_relation_object_ratio"] = _to_float(new_row, "ope_zero_relation_object_ratio")
        new_row["ope_zero_relation_object_names"] = new_row.get("ope_zero_relation_object_names", "")
        new_row["ope_relations_kept"] = _to_int(new_row, "ope_relations_kept")
        new_row["ope_relations_total"] = _to_int(new_row, "ope_relations_total")
        new_row["urp_keywords"] = _to_int(new_row, "urp_keywords")
        new_row["urp_zero_relation_keywords"] = _to_int(new_row, "urp_zero_relation_keywords")
        new_row["urp_zero_relation_keyword_ratio"] = _to_float(new_row, "urp_zero_relation_keyword_ratio")
        new_row["urp_zero_relation_keyword_names"] = new_row.get("urp_zero_relation_keyword_names", "")
        new_row["urp_relations_kept"] = _to_int(new_row, "urp_relations_kept")
        new_row["urp_relations_total"] = _to_int(new_row, "urp_relations_total")
        new_row["zero_relation_queries"] = _to_int(new_row, "zero_relation_queries")
        new_row["total_relation_queries"] = _to_int(new_row, "total_relation_queries")
        new_row["zero_relation_query_ratio"] = _to_float(new_row, "zero_relation_query_ratio")
        new_row["zero_relation_terms"] = new_row.get("zero_relation_terms", "")
        if "avg_kept_relations" in new_row:
            new_row["avg_kept_relations"] = _to_float(new_row, "avg_kept_relations")
        if "overall_kept_ratio" in new_row:
            new_row["overall_kept_ratio"] = _to_float(new_row, "overall_kept_ratio")
        normalized.append(new_row)
    return normalized


def _avg_kept_relations(row: dict) -> float:
    if "avg_kept_relations" in row:
        return _to_float(row, "avg_kept_relations")
    return _to_float(row, "avg_relations_per_object") + _to_float(row, "avg_relations_per_keyword")


def _overall_kept_ratio(row: dict) -> float:
    if "overall_kept_ratio" in row:
        return _to_float(row, "overall_kept_ratio")
    relations_total = _to_float(row, "relations_total")
    if not relations_total:
        return 0.0
    return _to_float(row, "relations_kept") / relations_total


def _load_table_rows(table_path: Path) -> list[dict]:
    with open(table_path, newline="") as f:
        rows = _normalize_rows(list(csv.DictReader(f)))
    if not rows:
        raise SystemExit(f"No rows found in {table_path}")
    return rows


def _range_columns_by_theta(rows: list[dict]) -> dict[float, dict[str, float]]:
    values_by_theta: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        theta = float(row["theta"])
        values = values_by_theta.setdefault(
            theta,
            {
                "avg_relations_per_object": [],
                "avg_relations_per_keyword": [],
                "avg_kept_relations": [],
                "overall_kept_ratio": [],
            },
        )
        values["avg_relations_per_object"].append(_to_float(row, "avg_relations_per_object"))
        values["avg_relations_per_keyword"].append(_to_float(row, "avg_relations_per_keyword"))
        values["avg_kept_relations"].append(_avg_kept_relations(row))
        values["overall_kept_ratio"].append(_overall_kept_ratio(row))

    ranges_by_theta = {}
    for theta, values in values_by_theta.items():
        ranges = {}
        for metric, metric_values in values.items():
            ranges[f"{metric}_min_for_theta"] = min(metric_values)
            ranges[f"{metric}_max_for_theta"] = max(metric_values)
        ranges_by_theta[theta] = ranges
    return ranges_by_theta


def _write_table(rows: list[dict], table_path: Path) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    ranges_by_theta = _range_columns_by_theta(rows)
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["category"], float(r["theta"]))):
            theta_ranges = ranges_by_theta[float(row["theta"])]
            writer.writerow(
                {
                    **row,
                    "avg_kept_relations": _avg_kept_relations(row),
                    "overall_kept_ratio": _overall_kept_ratio(row),
                    **theta_ranges,
                }
            )
    print(f"Saved table to {table_path}")


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


def _panel_series(
    rows: list[dict],
    categories: list[str],
    metric: str,
    color_by_category: dict[str, str],
) -> list[dict]:
    by_category: dict[str, list[dict]] = {category: [] for category in categories}
    for row in rows:
        by_category[row["category"]].append(row)
    for category_rows in by_category.values():
        category_rows.sort(key=lambda r: float(r["theta"]))

    return [
        {
            "color": color_by_category[category],
            "points": [(row["theta"], _metric_value(row, metric)) for row in by_category[category]],
        }
        for category in categories
    ]


def _metric_value(row: dict, metric: str) -> float:
    if metric == "success_rate":
        return _to_float(row, "success_rate")
    if metric == "avg_kept_relations":
        return _avg_kept_relations(row)
    if metric == "overall_kept_ratio":
        return _overall_kept_ratio(row)
    if metric == "zero_relation_queries":
        return _to_float(row, "zero_relation_queries")
    raise ValueError(f"Unsupported metric: {metric}")


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
        "title": _load_pil_font(34, bold=False),
        "subplot_title": _load_pil_font(24, bold=False),
        "axis_label": _load_pil_font(20, bold=False),
        "tick": _load_pil_font(18, bold=False),
        "legend": _load_pil_font(18, bold=False),
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
                    _overall_kept_ratio(row),
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
                    _avg_kept_relations(row),
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


def _plot_with_pillow_zero_relations(rows: list[dict], out_path: Path, categories: list[str], thetas: list[float]) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (1380, 1480), "white")
    draw = ImageDraw.Draw(img)
    fonts = {
        "title": _load_pil_font(34, bold=False),
        "subplot_title": _load_pil_font(24, bold=False),
        "axis_label": _load_pil_font(20, bold=False),
        "tick": _load_pil_font(18, bold=False),
        "legend": _load_pil_font(18, bold=False),
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
    avg_kept_series = _panel_series(rows, categories, "avg_kept_relations", color_by_category)
    kept_series = _panel_series(rows, categories, "overall_kept_ratio", color_by_category)
    zero_rel_series = _panel_series(rows, categories, "zero_relation_queries", color_by_category)

    avg_kept_max = max([point[1] for spec in avg_kept_series for point in spec["points"]] + [1.0])
    avg_kept_y_max = max(150.0, math.ceil(avg_kept_max / 25.0) * 25.0)
    avg_kept_ticks = list(range(0, int(avg_kept_y_max) + 1, 25))
    zero_rel_max = max([point[1] for spec in zero_rel_series for point in spec["points"]] + [1.0])
    zero_rel_y_max = max(25.0, math.ceil(zero_rel_max / 25.0) * 25.0)
    zero_rel_ticks = list(range(0, int(zero_rel_y_max) + 1, 25))

    _draw_panel(draw, boxes[0], "Average Kept Relations (Objects + Keywords)", thetas, avg_kept_series, "# relations", fonts, y_min=0.0, y_max=avg_kept_y_max, y_ticks=avg_kept_ticks)
    _draw_panel(draw, boxes[1], "Overall Kept Ratio", thetas, kept_series, "Overall kept/total", fonts, y_min=0.0, y_max=1.0, y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    _draw_panel(draw, boxes[2], "Object + Keyword Queries With Zero Relations", thetas, zero_rel_series, "# zero relations", fonts, y_min=0.0, y_max=zero_rel_y_max, y_ticks=zero_rel_ticks)

    for box, y_label in zip(boxes, ["# relations", "Overall kept/total", "# zero relations"]):
        left, top, _, bottom = box
        _draw_vertical_text(img, y_label, left - 95, int(top + (bottom - top) / 2 - 95), fonts["axis_label"])

    draw.text((margin_left, 24), "Theta Sensitivity Without Success Rate", fill="black", font=fonts["title"])
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


def _plot_combined(rows: list[dict], out_path: Path) -> None:
    rows = _normalize_rows(rows)
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
        _plot_with_pillow(rows, out_path, categories, thetas)
        return

    avg_kept_max = max(
        [
            _avg_kept_relations(row)
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
    ax.set_ylabel("Success rate", fontsize=17, rotation=90, labelpad=16)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_xlabel("Theta", fontsize=17)
    ax.set_title("Task Success Rate", fontsize=20, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[1]
    for category in categories:
        category_rows = by_category[category]
        ax.plot(
            [row["theta"] for row in category_rows],
            [_avg_kept_relations(row) for row in category_rows],
            marker="o",
            linewidth=2.5,
            label=category.replace("_", " "),
        )
    ax.set_title("Average Kept Relations (Objects + Keywords)", fontsize=20, pad=10)
    ax.set_ylabel("# relations", fontsize=17, rotation=90, labelpad=16)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_xlabel("Theta", fontsize=17)
    ax.set_ylim(0.0, avg_kept_y_max)
    ax.set_yticks(avg_kept_ticks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[2]
    for category in categories:
        category_rows = by_category[category]
        ax.plot(
            [row["theta"] for row in category_rows],
            [_overall_kept_ratio(row) for row in category_rows],
            marker="o",
            linewidth=2,
            label=category.replace("_", " "),
        )
    ax.set_title("Overall Kept Ratio", fontsize=20, pad=10)
    ax.set_xlabel("Theta", fontsize=17)
    ax.set_ylabel("Overall kept/total", fontsize=17, rotation=90, labelpad=16)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    for ax in axes:
        ax.set_xticks(thetas)
        ax.tick_params(axis="both", labelsize=14)

    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=14)

    fig.suptitle("Theta Sensitivity Across Evaluation Categories", fontsize=22, y=0.985)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.17, hspace=0.42)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"Saved plot to {out_path}")


def _plot_combined_zero_relations(rows: list[dict], out_path: Path) -> None:
    rows = _normalize_rows(rows)
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
        _plot_with_pillow_zero_relations(rows, out_path, categories, thetas)
        return

    avg_kept_max = max([_avg_kept_relations(row) for category in categories for row in by_category[category]] + [1.0])
    avg_kept_y_max = max(150.0, math.ceil(avg_kept_max / 25.0) * 25.0)
    avg_kept_ticks = list(range(0, int(avg_kept_y_max) + 1, 25))
    zero_rel_max = max([_to_float(row, "zero_relation_queries") for category in categories for row in by_category[category]] + [1.0])
    zero_rel_y_max = max(25.0, math.ceil(zero_rel_max / 25.0) * 25.0)
    zero_rel_ticks = list(range(0, int(zero_rel_y_max) + 1, 25))

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12.8), sharex=True, gridspec_kw={"height_ratios": [1.45, 1.0, 1.0]})

    ax = axes[0]
    for category in categories:
        category_rows = by_category[category]
        ax.plot([row["theta"] for row in category_rows], [_avg_kept_relations(row) for row in category_rows], marker="o", linewidth=2.5, label=category.replace("_", " "))
    ax.set_title("Average Kept Relations (Objects + Keywords)", fontsize=20, pad=10)
    ax.set_ylabel("# relations", fontsize=17, rotation=90, labelpad=16)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_xlabel("Theta", fontsize=17)
    ax.set_ylim(0.0, avg_kept_y_max)
    ax.set_yticks(avg_kept_ticks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[1]
    for category in categories:
        category_rows = by_category[category]
        ax.plot([row["theta"] for row in category_rows], [_overall_kept_ratio(row) for row in category_rows], marker="o", linewidth=2, label=category.replace("_", " "))
    ax.set_title("Overall Kept Ratio", fontsize=20, pad=10)
    ax.set_xlabel("Theta", fontsize=17)
    ax.set_ylabel("Overall kept/total", fontsize=17, rotation=90, labelpad=16)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    ax = axes[2]
    for category in categories:
        category_rows = by_category[category]
        ax.plot([row["theta"] for row in category_rows], [_to_float(row, "zero_relation_queries") for row in category_rows], marker="o", linewidth=2, label=category.replace("_", " "))
    ax.set_title("Object + Keyword Queries With Zero Relations", fontsize=20, pad=10)
    ax.set_xlabel("Theta", fontsize=17)
    ax.set_ylabel("# zero relations", fontsize=17, rotation=90, labelpad=16)
    ax.yaxis.set_label_coords(-0.045, 0.5)
    ax.set_ylim(0.0, zero_rel_y_max)
    ax.set_yticks(zero_rel_ticks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.18)

    for ax in axes:
        ax.set_xticks(thetas)
        ax.tick_params(axis="both", labelsize=14)

    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=14)

    fig.suptitle("Theta Sensitivity Without Success Rate", fontsize=22, y=0.985)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.17, hspace=0.42)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing threshold_sweep_*.json files.",
    )
    parser.add_argument(
        "--table-in",
        default=None,
        help="Optional combined CSV to plot instead of reading threshold_sweep_*.json files.",
    )
    parser.add_argument(
        "--table-out",
        default=None,
        help=f"Output CSV path. Defaults to <results-dir>/{DEFAULT_TABLE_NAME}.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Do not write the combined CSV table.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"Output PNG path. Defaults to <results-dir>/{DEFAULT_PLOT_NAME}",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.table_in:
        rows = _load_table_rows(Path(args.table_in))
    else:
        rows = _normalize_rows(_load_rows(results_dir))

    table_out = Path(args.table_out) if args.table_out else results_dir / DEFAULT_TABLE_NAME
    if not args.no_table:
        _write_table(rows, table_out)

    out_path = Path(args.out) if args.out else results_dir / DEFAULT_PLOT_NAME
    _plot_combined(rows, out_path)
    _plot_combined_zero_relations(rows, results_dir / DEFAULT_SUPPLEMENTAL_PLOT_NAME)


if __name__ == "__main__":
    main()
