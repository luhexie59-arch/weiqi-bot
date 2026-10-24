"""Shared visualisation and JSON export used by the GUI and command line."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scoring import count_area, BLACK, WHITE
from vision import BOARD_PIXELS, MARGIN, grid_coordinates


def _font(size):
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_board(board, confidence=None, neutral_points=(), photo=None) -> np.ndarray:
    score = count_area(board, neutral_points)
    n = len(board)
    if photo is None:
        image = Image.new("RGB", (BOARD_PIXELS, BOARD_PIXELS), "#ddba7e")
    else:
        image = Image.fromarray(cv2.cvtColor(photo.copy(), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    grid = grid_coordinates(n)
    spacing = grid[1] - grid[0]
    if photo is None:
        for p in grid:
            draw.line((grid[0], p, grid[-1], p), fill="#534128", width=2)
            draw.line((p, grid[0], p, grid[-1]), fill="#534128", width=2)
        stars = [2, 4, 6] if n == 9 else ([3, 6, 9] if n == 13 else [3, 9, 15])
        for r in stars:
            for c in stars:
                x, y = grid[c], grid[r]
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#534128")
    for r, y in enumerate(grid):
        for c, x in enumerate(grid):
            value = int(board[r, c])
            if value:
                radius = spacing * (.42 if photo is None else .24)
                box = (x - radius, y - radius, x + radius, y + radius)
                if photo is None:
                    draw.ellipse(box, fill="#171b21" if value == BLACK else "#fafafa",
                                 outline="#101820" if value == BLACK else "#6b7280", width=2)
                    if value == BLACK:
                        draw.arc((x - radius + 4, y - radius + 4, x + radius - 4, y + radius - 4),
                                 205, 280, fill="#535c66", width=2)
                else:
                    draw.ellipse(box, outline="#49ccf2" if value == BLACK else "#f376bc", width=3)
            else:
                owner = int(score.ownership[r, c])
                half = spacing * (.075 if owner == 0 else .13)
                colour = {0: "#80878b", BLACK: "#087e96", WHITE: "#cd448a"}[owner]
                draw.rectangle((x - half, y - half, x + half, y + half), fill=colour,
                               outline="#e2e8f0" if photo is not None else None)
            if confidence is not None and confidence[r, c] < .25:
                half = spacing * .42
                draw.rectangle((x - half, y - half, x + half, y + half), outline="#ff8c00", width=2)
    font = _font(14)
    columns = "ABCDEFGHJKLMNOPQRST"[:n]
    for i, p in enumerate(grid):
        draw.text((p, 10), columns[i], font=font, fill="#252d36", anchor="mt")
        draw.text((20, p), str(n - i), font=font, fill="#252d36", anchor="rm")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def result_dict(board, *, recognition=None, source=None, neutral_points=()) -> dict:
    result = {
        "schema_version": 1,
        "board_size": len(board),
        "source": str(source) if source else None,
        "rules": "stones_plus_single_colour_empty_regions; no_komi; no_dead_stone_inference",
        "legend": {"0": "empty/neutral", "1": "black", "2": "white"},
        "coordinate_convention": "matrices: row 0 at top, column 0 at left; points: [row, column], zero-based",
        "board": np.asarray(board).tolist(),
        "neutral_points": [list(p) for p in sorted(neutral_points)],
        "score": count_area(board, neutral_points).as_dict(),
    }
    if recognition is not None:
        result.update(corners=recognition.corners.tolist(),
                      confidence=recognition.confidence.tolist(),
                      grid_quality=recognition.grid_quality, warnings=recognition.warnings)
    return result


def export_result(base_path, board, *, recognition=None, source=None, neutral_points=()):
    """Write JSON, labelled diagram, and optional perspective-corrected photo."""
    base = Path(base_path)
    if base.suffix.lower() in (".json", ".png"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    data = result_dict(board, recognition=recognition, source=source, neutral_points=neutral_points)
    json_path = Path(str(base) + ".json")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    confidence = recognition.confidence if recognition is not None else None
    diagram = Image.fromarray(cv2.cvtColor(draw_board(board, confidence, neutral_points), cv2.COLOR_BGR2RGB))
    sheet = Image.new("RGB", (BOARD_PIXELS, BOARD_PIXELS + 150), "#f5f7fa")
    sheet.paste(diagram, (0, 150))
    draw = ImageDraw.Draw(sheet)
    score = count_area(board, neutral_points)
    draw.text((24, 12), "围棋当前占地 · 子数 + 独占空点", fill="#172536", font=_font(26))
    draw.text((24, 53), f"黑 {score.black_area} 点（子 {score.black_stones} + 空 {score.black_territory}）   "
              f"白 {score.white_area} 点（子 {score.white_stones} + 空 {score.white_territory}）   "
              f"中立 {score.neutral}", fill="#172536", font=_font(22))
    draw.text((24, 93), "青色方点：黑方空地   粉色方点：白方空地   灰色：中立   橙框：建议核对", fill="#45556b", font=_font(18))
    png_path = Path(str(base) + ".png")
    sheet.save(png_path)
    paths = [json_path, png_path]
    if recognition is not None:
        corrected_path = Path(str(base) + "_corrected.png")
        Image.fromarray(cv2.cvtColor(recognition.corrected, cv2.COLOR_BGR2RGB)).save(corrected_path)
        paths.append(corrected_path)
    return paths
