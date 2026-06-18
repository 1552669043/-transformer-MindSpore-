# -*- coding: utf-8 -*-
"""Create a grouped bar chart from paper-reported ADE20K mIoU metrics."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "paper_figures"
CSV_PATH = OUT_DIR / "paper_ade20k_miou_metrics.csv"
PNG_PATH = OUT_DIR / "paper_ade20k_miou_grouped_bar.png"


DATA = [
    {
        "paper": "SETR",
        "config": "SETR-MLA",
        "ss_miou": 48.64,
        "ms_miou": 50.28,
        "params_m": 310.57,
        "source": "SSPT.pdf, Table 4",
        "note": "ADE20K val, T-Large, 160k iterations",
    },
    {
        "paper": "SegFormer",
        "config": "SegFormer-B5",
        "ss_miou": 51.0,
        "ms_miou": 51.8,
        "params_m": 84.7,
        "source": "SegFormer.pdf, Table 1",
        "note": "ADE20K val, MiT-B5; params = encoder 81.4M + decoder 3.3M",
    },
    {
        "paper": "Mask2Former",
        "config": "Swin-L-FaPN",
        "ss_miou": 56.4,
        "ms_miou": 57.7,
        "params_m": 217.0,
        "source": "Mask.pdf, Table V",
        "note": "ADE20K val, best semantic configuration in the paper",
    },
    {
        "paper": "OneFormer",
        "config": "best reported",
        "ss_miou": 58.3,
        "ms_miou": 58.8,
        "params_m": "",
        "source": "OneFormer.pdf, Table VI",
        "note": "SS: DiNAT-L 640 crop; MS: ConvNeXt-XL 640 crop",
    },
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


def write_csv() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper",
                "config",
                "ss_miou",
                "ms_miou",
                "params_m",
                "source",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(DATA)


def draw_chart() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    width, height = 1800, 1180
    margin_left, margin_right = 170, 90
    plot_top, plot_bottom = 190, 840
    plot_left, plot_right = margin_left, width - margin_right
    plot_width, plot_height = plot_right - plot_left, plot_bottom - plot_top
    y_max = 65.0

    bg = "#fbfbf7"
    axis = "#2b2d33"
    grid = "#d8d8d2"
    ss_color = "#3b7ea1"
    ms_color = "#ef8a3a"

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    title_font = font(44, bold=True)
    subtitle_font = font(25)
    label_font = font(25)
    small_font = font(21)
    tick_font = font(22)
    value_font = font(22, bold=True)

    draw.text((margin_left, 54), "四篇论文 ADE20K 语义分割 mIoU 对比", font=title_font, fill=axis)
    draw.text(
        (margin_left, 115),
        "Grouped bar chart of paper-reported validation results: SS = single-scale, MS = multi-scale",
        font=subtitle_font,
        fill="#62656d",
    )

    # Plot background.
    draw.rounded_rectangle(
        (plot_left - 10, plot_top - 8, plot_right + 8, plot_bottom + 8),
        radius=12,
        fill="#ffffff",
        outline="#ebe9df",
        width=2,
    )

    def y_to_px(value: float) -> float:
        return plot_bottom - (value / y_max) * plot_height

    # Grid, ticks, and axes.
    for tick in range(0, 70, 10):
        y = y_to_px(float(tick))
        draw.line((plot_left, y, plot_right, y), fill=grid, width=1)
        draw_centered(draw, (plot_left - 48, y), f"{tick}", tick_font, "#60636a")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis, width=3)

    y_label = "mIoU (%)"
    label_img = Image.new("RGBA", (180, 40), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), y_label, font=label_font, fill=axis)
    rotated = label_img.rotate(90, expand=True)
    img.paste(rotated, (38, int(plot_top + plot_height / 2 - rotated.height / 2)), rotated)

    group_count = len(DATA)
    group_width = plot_width / group_count
    bar_width = 92
    bar_gap = 24

    for idx, row in enumerate(DATA):
        cx = plot_left + group_width * (idx + 0.5)
        values = [("SS", row["ss_miou"], ss_color), ("MS", row["ms_miou"], ms_color)]
        left_start = cx - bar_width - bar_gap / 2
        for j, (_label, value, color) in enumerate(values):
            x0 = left_start + j * (bar_width + bar_gap)
            x1 = x0 + bar_width
            y0 = y_to_px(float(value))
            draw.rounded_rectangle((x0, y0, x1, plot_bottom), radius=8, fill=color)
            draw.text(
                (x0 + bar_width / 2 - 33, y0 - 35),
                f"{value:.2f}".rstrip("0").rstrip("."),
                font=value_font,
                fill=axis,
            )

        label_y = plot_bottom + 42
        draw_centered(draw, (cx, label_y), row["paper"], label_font, axis)
        draw_centered(draw, (cx, label_y + 35), f"({row['config']})", small_font, "#62656d")

    # Legend.
    legend_y = 126
    legend_x = plot_right - 450
    for label, color, offset in [
        ("SS 单尺度", ss_color, 0),
        ("MS 多尺度", ms_color, 190),
    ]:
        x = legend_x + offset
        draw.rounded_rectangle((x, legend_y, x + 34, legend_y + 22), radius=5, fill=color)
        draw.text((x + 46, legend_y - 3), label, font=small_font, fill=axis)

    source_lines = [
        "数据来源: SSPT.pdf Table 4; SegFormer.pdf Table 1; Mask.pdf Table V; OneFormer.pdf Table VI.",
        "OneFormer 采用论文中最佳报告值: SS 为 DiNAT-L 640 crop, MS 为 ConvNeXt-XL 640 crop。",
        f"CSV: {CSV_PATH.name}",
    ]
    y = 980
    for line in source_lines:
        draw.text((margin_left, y), line, font=small_font, fill="#666a72")
        y += 34

    img.save(PNG_PATH)


def main() -> None:
    write_csv()
    draw_chart()
    print(PNG_PATH)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
