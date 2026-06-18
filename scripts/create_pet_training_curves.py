# -*- coding: utf-8 -*-
"""Draw training curves for the pet_fast_run_60 experiment."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "pet_fast_run_60"
METRICS_DIR = RUN_DIR / "outputs"
OUT_PATH = RUN_DIR / "pet_fast_run_60_training_curves.png"
SUMMARY_PATH = RUN_DIR / "pet_fast_run_60_final_metrics.csv"


MODELS = [
    ("SETR-tiny", "setr_tiny_pet_fast", "#3b7ea1"),
    ("SegFormer-tiny", "segformer_tiny_pet_fast", "#ef8a3a"),
    ("SegFormer-Edge", "segformer_edge_pet_fast", "#5a9c55"),
    ("Mask2Former-tiny", "mask2former_tiny_pet_fast", "#c85250"),
    ("OneFormer-tiny", "oneformer_tiny_pet_fast", "#8067b7"),
]


PANELS = [
    ("loss", "Training Loss", "Loss", False),
    ("pixel_acc", "Pixel Accuracy", "Accuracy (%)", True),
    ("mean_iou", "Mean IoU", "mIoU (%)", True),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_metrics() -> list[dict[str, object]]:
    series = []
    for display_name, folder, color in MODELS:
        path = METRICS_DIR / folder / "metrics.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        parsed = []
        for row in rows:
            parsed.append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]),
                    "pixel_acc": float(row["pixel_acc"]),
                    "mean_iou": float(row["mean_iou"]),
                }
            )
        series.append(
            {
                "name": display_name,
                "folder": folder,
                "color": color,
                "rows": parsed,
            }
        )
    return series


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=fnt)
    return right - left, bottom - top


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str,
) -> None:
    width, height = text_size(draw, text, fnt)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=fnt, fill=fill)


def nice_range(values: list[float], as_percent: bool) -> tuple[float, float, list[float]]:
    if as_percent:
        values = [v * 100 for v in values]
    vmin, vmax = min(values), max(values)
    if as_percent:
        step = 5.0
        low = max(0.0, (int(vmin // step) - 1) * step)
        high = min(100.0, (int(vmax // step) + 2) * step)
    else:
        step = 0.1
        low = max(0.0, (int(vmin / step) - 1) * step)
        high = (int(vmax / step) + 2) * step
    ticks = []
    current = low
    while current <= high + step / 2:
        ticks.append(round(current, 4))
        current += step
    return low, high, ticks


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    width: int = 4,
) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points[:: max(1, len(points) // 12)]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    if points:
        x, y = points[-1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    panel_box: tuple[int, int, int, int],
    series: list[dict[str, object]],
    metric: str,
    title: str,
    ylabel: str,
    as_percent: bool,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    left, top, right, bottom = panel_box
    axis_color = "#2b2d33"
    grid_color = "#dcded9"
    text_color = "#5d626b"

    all_rows = [row for item in series for row in item["rows"]]  # type: ignore[index]
    max_epoch = max(int(row["epoch"]) for row in all_rows)
    values = [float(row[metric]) for row in all_rows]
    y_min, y_max, y_ticks = nice_range(values, as_percent)

    plot_left, plot_top = left + 120, top + 66
    plot_right, plot_bottom = right - 70, bottom - 75
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=14,
        fill="#ffffff",
        outline="#ebe9df",
        width=2,
    )
    draw.text((left + 30, top + 20), title, font=fonts["panel_title"], fill=axis_color)

    def x_to_px(epoch: int) -> float:
        if max_epoch <= 1:
            return plot_left
        return plot_left + (epoch - 1) / (max_epoch - 1) * plot_width

    def y_to_px(value: float) -> float:
        if as_percent:
            value *= 100
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_height

    for tick in y_ticks:
        y = plot_bottom - (tick - y_min) / (y_max - y_min) * plot_height
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        label = f"{tick:.0f}" if as_percent else f"{tick:.1f}"
        draw_centered(draw, (plot_left - 45, y), label, fonts["tick"], text_color)

    x_ticks = [1, 10, 20, 30, 40, 50, max_epoch]
    for tick in x_ticks:
        x = x_to_px(tick)
        draw.line((x, plot_bottom, x, plot_bottom + 7), fill=axis_color, width=2)
        draw_centered(draw, (x, plot_bottom + 32), str(tick), fonts["tick"], text_color)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=3)

    draw_centered(draw, ((plot_left + plot_right) / 2, bottom - 26), "Epoch", fonts["axis"], axis_color)
    ylabel_img = Image.new("RGBA", (220, 38), (255, 255, 255, 0))
    ylabel_draw = ImageDraw.Draw(ylabel_img)
    ylabel_draw.text((0, 0), ylabel, font=fonts["axis"], fill=axis_color)
    rotated = ylabel_img.rotate(90, expand=True)
    draw.bitmap((left + 24, int((plot_top + plot_bottom - rotated.height) / 2)), rotated, fill=None)

    final_labels = []
    for item in series:
        rows = item["rows"]  # type: ignore[assignment]
        points = [(x_to_px(int(row["epoch"])), y_to_px(float(row[metric]))) for row in rows]  # type: ignore[index]
        draw_polyline(draw, points, str(item["color"]))

        last = rows[-1]  # type: ignore[index]
        last_value = float(last[metric])
        label_value = last_value * 100 if as_percent else last_value
        label = f"{label_value:.2f}" if as_percent else f"{label_value:.3f}"
        x, y = points[-1]
        final_labels.append(
            {
                "x": x,
                "y": y,
                "adjusted_y": y,
                "label": label,
                "color": str(item["color"]),
            }
        )

    final_labels.sort(key=lambda item: item["y"])
    label_gap = 24
    for idx in range(1, len(final_labels)):
        previous = final_labels[idx - 1]["adjusted_y"]
        if final_labels[idx]["adjusted_y"] < previous + label_gap:
            final_labels[idx]["adjusted_y"] = previous + label_gap
    if final_labels and final_labels[-1]["adjusted_y"] > plot_bottom - 8:
        shift = final_labels[-1]["adjusted_y"] - (plot_bottom - 8)
        for item in final_labels:
            item["adjusted_y"] -= shift
    if final_labels and final_labels[0]["adjusted_y"] < plot_top + 8:
        shift = (plot_top + 8) - final_labels[0]["adjusted_y"]
        for item in final_labels:
            item["adjusted_y"] += shift

    for item in final_labels:
        x = float(item["x"])
        y = float(item["y"])
        adjusted_y = float(item["adjusted_y"])
        color = str(item["color"])
        draw.line((x + 3, y, x + 10, adjusted_y), fill=color, width=2)
        draw.text((x + 12, adjusted_y - 12), str(item["label"]), font=fonts["small"], fill=color)


def write_summary(series: list[dict[str, object]]) -> None:
    with SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "epoch", "final_loss", "final_pixel_acc", "final_mean_iou"],
        )
        writer.writeheader()
        for item in series:
            last = item["rows"][-1]  # type: ignore[index]
            writer.writerow(
                {
                    "model": item["name"],
                    "epoch": last["epoch"],
                    "final_loss": f"{last['loss']:.6f}",
                    "final_pixel_acc": f"{last['pixel_acc']:.6f}",
                    "final_mean_iou": f"{last['mean_iou']:.6f}",
                }
            )


def draw_chart(series: list[dict[str, object]]) -> None:
    width, height = 1900, 1720
    bg = "#fbfbf7"
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": font(48, bold=True),
        "subtitle": font(25),
        "panel_title": font(31, bold=True),
        "axis": font(23),
        "tick": font(20),
        "small": font(19),
        "legend": font(22),
    }

    title_color = "#2b2d33"
    muted = "#62656d"
    draw.text((115, 48), "pet_fast_run_60 训练过程曲线", font=fonts["title"], fill=title_color)
    draw.text(
        (115, 113),
        "Loss, Pixel Accuracy and mIoU over 60 epochs for five semantic segmentation models",
        font=fonts["subtitle"],
        fill=muted,
    )

    legend_x, legend_y = 115, 166
    x = legend_x
    for name, _folder, color in MODELS:
        draw.line((x, legend_y + 13, x + 44, legend_y + 13), fill=color, width=6)
        draw.ellipse((x + 17, legend_y + 6, x + 29, legend_y + 18), fill=color)
        draw.text((x + 56, legend_y), name, font=fonts["legend"], fill=title_color)
        x += 315

    panels = [
        (90, 230, width - 90, 690),
        (90, 735, width - 90, 1195),
        (90, 1240, width - 90, 1700),
    ]
    for box, panel in zip(panels, PANELS):
        draw_panel(draw, box, series, *panel, fonts)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)


def main() -> None:
    series = read_metrics()
    write_summary(series)
    draw_chart(series)
    print(OUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
