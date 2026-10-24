"""OpenCV board detection, projective correction and local stone recognition."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from scoring import BLACK, WHITE, EMPTY

BOARD_PIXELS = 900
MARGIN = 64


@dataclass
class Recognition:
    board: np.ndarray
    corrected: np.ndarray
    corners: np.ndarray
    confidence: np.ndarray
    grid_quality: float | None = None
    warnings: list[str] = field(default_factory=list)


def read_image(path: str | Path | BinaryIO) -> np.ndarray:
    """Read a path or binary stream, respecting camera EXIF orientation."""
    with Image.open(path) as image:
        if image.width * image.height > 60_000_000:
            raise ValueError("图片超过 6000 万像素，请先缩小。")
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image(path: str | Path, image: np.ndarray) -> None:
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)


def order_corners(points) -> np.ndarray:
    """Order a convex quadrilateral TL, TR, BR, BL (image orientation)."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("需要四个有效的 (x, y) 角点。")
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 100:
        raise ValueError("四角必须构成凸四边形，且不能重合或共线。")
    centre = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - centre[1], hull[:, 0] - centre[0])
    ordered = hull[np.argsort(angles)]
    return np.roll(ordered, -np.argmin(ordered.sum(axis=1)), axis=0).astype(np.float32)


def grid_coordinates(size: int) -> np.ndarray:
    return np.linspace(MARGIN, BOARD_PIXELS - MARGIN - 1, size)


def rectify(image: np.ndarray, corners) -> np.ndarray:
    corners = order_corners(corners)
    h, w = image.shape[:2]
    if np.any(corners < 0) or np.any(corners[:, 0] >= w) or np.any(corners[:, 1] >= h):
        raise ValueError("角点必须在图片范围内。")
    end = BOARD_PIXELS - MARGIN - 1
    target = np.float32([[MARGIN, MARGIN], [end, MARGIN], [end, end], [MARGIN, end]])
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image, matrix, (BOARD_PIXELS, BOARD_PIXELS),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _fit_axis(profile: np.ndarray, size: int) -> tuple[float, float, float]:
    """Fit a regularly spaced comb; require every line, not only a few peaks."""
    length = len(profile)
    baseline = float(np.percentile(profile, 45))
    scale = max(float(np.percentile(profile, 99)) - baseline, 0.035)
    signal = np.clip((profile - baseline) / scale, 0, 1.5)
    # Allow line width / interpolation error without moving the estimated line.
    signal = cv2.dilate(signal.astype(np.float32)[None, :], np.ones((1, 3), np.uint8))[0]
    indices = np.arange(size)
    best = (-1.0, 0.0, 0.0)
    for step in np.arange(length * .48 / (size - 1), (length - 6) / (size - 1), .35):
        starts = np.arange(3, length - 3 - step * (size - 1), 1.5)
        if not len(starts):
            continue
        positions = starts[:, None] + step * indices[None, :]
        sampled = signal[np.rint(positions).astype(int)]
        # Weak lines penalise harmonic fits and a crop containing only stones.
        values = sampled.mean(axis=1) * .6 + np.quantile(sampled, .2, axis=1) * .4
        k = int(np.argmax(values))
        candidate = float(values[k])
        if candidate > best[0]:
            best = (candidate, float(starts[k]), float(step))
    if best[0] <= 0:
        return 0.0, 0.0, (length - 1) / (size - 1)
    _, start, step = best
    # Refine below a pixel to avoid an accumulating last-line error.
    for s in np.arange(max(1, step - .5), step + .51, .05):
        starts = np.arange(max(1, start - 2), min(start + 2.01, length - 1 - s * (size - 1)), .2)
        if not len(starts):
            continue
        positions = starts[:, None] + s * indices[None, :]
        sampled = np.interp(positions.ravel(), np.arange(length), signal).reshape(-1, size)
        values = sampled.mean(axis=1) * .6 + np.quantile(sampled, .2, axis=1) * .4
        k = int(np.argmax(values))
        if values[k] > best[0]:
            best = (float(values[k]), float(starts[k]), float(s))
    return best


def _line_profiles(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 8)
    # Long kernels remove stones, text and most wood grain from line evidence.
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((65, 1), np.uint8))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, 65), np.uint8))
    return vertical.mean(axis=0) / 255, horizontal.mean(axis=1) / 255


def _candidate_quads(image: np.ndarray) -> list[np.ndarray]:
    h, w = image.shape[:2]
    candidates = [np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    # Board surface contours can provide the perspective even when the outer
    # grid is occluded. The lattice fit subsequently removes the wooden border.
    masks = [cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))]
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(threshold)
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:35]:
            if abs(cv2.contourArea(contour)) < w * h * .12:
                continue
            hull = cv2.convexHull(contour)
            polygon = cv2.approxPolyDP(hull, .025 * cv2.arcLength(hull, True), True)
            if len(polygon) == 4:
                candidates.append(order_corners(polygon.reshape(4, 2)))
    # Long line families give an alternative when a board has no visible rim.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 70,
                            minLineLength=min(h, w) * .22, maxLineGap=30)
    if lines is not None:
        vertical, horizontal = [], []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            line = np.cross([float(x1), float(y1), 1.], [float(x2), float(y2), 1.])
            if abs(dy) > abs(dx) * 1.4:
                x_mid = -(line[1] * h / 2 + line[2]) / line[0]
                vertical.append((x_mid, line))
            elif abs(dx) > abs(dy) * 1.4:
                y_mid = -(line[0] * w / 2 + line[2]) / line[1]
                horizontal.append((y_mid, line))
        if len(vertical) >= 4 and len(horizontal) >= 4:
            vertical.sort(key=lambda x: x[0])
            horizontal.sort(key=lambda x: x[0])
            # Extremes and trimmed extremes handle an occasional background edge.
            for trim in (0, .08):
                vi, hi = int(len(vertical) * trim), int(len(horizontal) * trim)
                left, right = vertical[vi][1], vertical[-1 - vi][1]
                top, bottom = horizontal[hi][1], horizontal[-1 - hi][1]
                points = []
                for a, b in ((left, top), (right, top), (right, bottom), (left, bottom)):
                    p = np.cross(a, b)
                    if abs(p[2]) < 1e-6:
                        break
                    points.append(p[:2] / p[2])
                if len(points) == 4:
                    try:
                        quad = order_corners(points)
                        if (quad[:, 0] >= 0).all() and (quad[:, 0] < w).all() and (quad[:, 1] >= 0).all() and (quad[:, 1] < h).all():
                            candidates.append(quad)
                    except ValueError:
                        pass
    unique = []
    for quad in candidates:
        if not any(np.max(np.abs(quad - old)) < 10 for old in unique):
            unique.append(quad)
    return unique[:16]


def detect_corners(image: np.ndarray, size: int = 19) -> tuple[np.ndarray, float]:
    if size not in (9, 13, 19):
        raise ValueError("支持 9、13、19 路棋盘。")
    h, w = image.shape[:2]
    ratio = min(1., 1100 / max(h, w))
    small = cv2.resize(image, None, fx=ratio, fy=ratio) if ratio < 1 else image
    side = 760
    target = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
    best = None
    for quad in _candidate_quads(small):
        transform = cv2.getPerspectiveTransform(quad, target)
        warped = cv2.warpPerspective(small, transform, (side, side))
        px, py = _line_profiles(warped)
        qx, x0, dx = _fit_axis(px, size)
        qy, y0, dy = _fit_axis(py, size)
        quality = min(qx, qy)
        lattice = np.float32([[x0, y0], [x0 + dx * (size - 1), y0],
                              [x0 + dx * (size - 1), y0 + dy * (size - 1)],
                              [x0, y0 + dy * (size - 1)]])
        corners = cv2.perspectiveTransform(lattice[None], np.linalg.inv(transform))[0] / ratio
        if best is None or quality > best[0]:
            best = (quality, corners)
    if best is None or best[0] < .40:
        raise ValueError("无法可靠定位完整网格。请选择正确路数，并使用“手动四角”标定最外侧四个交点。")
    return order_corners(best[1]), float(best[0])


def _search_offset_stone(lab: np.ndarray, x: int, y: int, step: float,
                         background: np.ndarray, sensitivity: float) -> tuple[int, float] | None:
    """Search candidate centres only inside +/- one third of a grid interval.

    Require a mostly uniform stone-sized disk, not just a dark/bright fragment.
    A neighbouring stone's edge can otherwise win a shifted colour comparison.
    Pale white stones additionally need a circular rim, since empty pale board
    pixels have essentially the same colour as the stone.
    """
    limit = int(np.floor(step / 3))
    offsets = np.unique(np.rint(np.linspace(-limit, limit, 5)).astype(int))
    radius = int(np.ceil(step * .55))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1] / step
    distance = np.hypot(xx, yy)
    disk = (distance > .12) & (distance < .31)
    background_chroma = np.linalg.norm(background[1:] - 128)
    # Rays away from grid-line directions distinguish a round rim from a cross.
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    angles = angles[(np.abs(np.cos(angles)) > .23) & (np.abs(np.sin(angles)) > .23)]
    radii = np.linspace(.34, .54, 12)
    ray_x = np.rint(radius + step * np.cos(angles)[:, None] * radii).astype(int)
    ray_y = np.rint(radius + step * np.sin(angles)[:, None] * radii).astype(int)
    best = None
    best_rank = -1.
    for dy in offsets:
        for dx in offsets:
            cx, cy = x + int(dx), y + int(dy)
            patch = lab[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
            pixels = patch[disk]
            middle = np.median(pixels, axis=0)
            delta = middle[0] - background[0]
            chroma_gain = max(0., background_chroma - np.linalg.norm(middle[1:] - 128))
            black_strength = -delta / 45 * sensitivity
            white_strength = max(delta / 28, (delta + chroma_gain * 1.6) / 34) * sensitivity
            black_coverage = float(np.mean(pixels[:, 0] < background[0] - 35 / sensitivity))
            white_pixels = (pixels[:, 0] - background[0]
                            + np.maximum(0., background_chroma - np.linalg.norm(pixels[:, 1:] - 128, axis=1)) * 1.6)
            white_coverage = float(np.mean(white_pixels > 28 / sensitivity))
            if background_chroma < 7 and np.linalg.norm(middle[1:] - 128) < 5 and middle[0] > 90:
                pale_coverage = float(np.mean(pixels[:, 0] > background[0] - 14))
                rim_coverage = float(np.mean(np.min(patch[ray_y, ray_x, 0], axis=1) < middle[0] - 20 / sensitivity))
                if pale_coverage >= .9 and rim_coverage >= .8:
                    white_coverage = max(white_coverage, pale_coverage)
                    white_strength = max(white_strength, 2 * rim_coverage)
            for label, strength, coverage in ((BLACK, black_strength, black_coverage),
                                               (WHITE, white_strength, white_coverage)):
                if strength <= 1.15 or coverage < .85:
                    continue
                certainty = float(np.clip((strength - 1) / 1.5, 0, 1))
                # Prefer complete disks and use distance only as a tie-breaker.
                rank = min(strength, 3.) * coverage - .015 * np.hypot(dx, dy) / step
                if rank > best_rank:
                    best_rank = rank
                    best = (label, min(certainty, coverage))
    return best


def classify_stones(corrected: np.ndarray, size: int, sensitivity: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """Compare off-line centre pixels against nearby diagonal board samples.

    Mask out grid lines and star points; robust percentiles tolerate small
    highlights. Chroma separation helps white stones on a warm wooden board.
    Uncertain intersections are re-examined within +/- one third of a grid
    interval on each axis. Confidence is a review aid, not a probability.
    """
    if size not in (9, 13, 19):
        raise ValueError("支持 9、13、19 路棋盘。")
    if not .5 <= sensitivity <= 2:
        raise ValueError("灵敏度必须在 0.5 到 2 之间。")
    grid = grid_coordinates(size)
    step = grid[1] - grid[0]
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB).astype(float)
    radius = int(step * .70)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1] / step
    distance = np.hypot(xx, yy)
    # Avoid cross lines and central hoshi dots, without sampling adjacent stones.
    centre_mask = (distance < .34) & (distance > .15) & (np.abs(xx) > .075) & (np.abs(yy) > .075)
    background_mask = (distance > .57) & (distance < .70) & (np.abs(xx) > .20) & (np.abs(yy) > .20)
    padding = radius + int(np.ceil(step / 3)) + 2
    padded = cv2.copyMakeBorder(lab, padding, padding, padding, padding, cv2.BORDER_REFLECT_101)
    board = np.zeros((size, size), dtype=np.uint8)
    confidence = np.zeros((size, size), dtype=float)
    narrow = ((np.abs(xx) < .055) & (np.abs(yy) < .30)) | ((np.abs(yy) < .055) & (np.abs(xx) < .30))
    thin_cross = ((np.abs(xx) < .5 / step) & (np.abs(yy) < .30)) | ((np.abs(yy) < .5 / step) & (np.abs(xx) < .30))
    uncertain = []
    for row, y in enumerate(grid):
        for col, x in enumerate(grid):
            x, y = round(x), round(y)
            cx, cy = x + padding, y + padding
            patch = padded[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
            centre = patch[centre_mask]
            background = patch[background_mask]
            # Nearby diagonals lie in gaps between stones. Median reduces the
            # effect of an adjacent stone shadow, grid remnants, or wood grain.
            bg = np.median(background, axis=0)
            middle = np.median(centre, axis=0)
            delta = middle[0] - bg[0]
            chroma_gain = np.linalg.norm(bg[1:] - 128) - np.linalg.norm(middle[1:] - 128)
            dark_fraction = np.mean(centre[:, 0] < bg[0] - 35 / sensitivity)
            black_strength = max(-delta / 45, dark_fraction / .65 if delta < -22 else 0)
            white_strength = max(delta / 28, (delta + max(0, chroma_gain) * 1.6) / 34)
            # A white disk also covers the grid across its centre. This catches
            # pure white stones on a very pale screenshot with little contrast.
            line_mask, trace = narrow, thin_cross
            if row in (0, size - 1) or col in (0, size - 1):
                valid = np.ones_like(distance, dtype=bool)
                if row == 0:
                    valid &= yy >= 0
                elif row == size - 1:
                    valid &= yy <= 0
                if col == 0:
                    valid &= xx >= 0
                elif col == size - 1:
                    valid &= xx <= 0
                line_mask, trace = narrow & valid, thin_cross & valid
            line_light = float(np.percentile(patch[:, :, 0][line_mask], 25))
            pale_surface = np.linalg.norm(bg[1:] - 128) < 7 and np.linalg.norm(middle[1:] - 128) < 5
            flat_white = pale_surface and middle[0] > 90 and delta > -12 and line_light > middle[0] - 14
            if black_strength * sensitivity > 1:
                label, strength = BLACK, black_strength * sensitivity
            elif white_strength * sensitivity > 1 or flat_white:
                label = WHITE
                strength = max(white_strength * sensitivity, 1.3 if flat_white else 0)
            else:
                label = EMPTY
                strength = max(black_strength, white_strength) * sensitivity
            board[row, col] = label
            confidence[row, col] = np.clip((strength - 1) / 1.5 if label else (1 - strength), 0, 1)
            # A median can hide a displaced disk or combine the brightness of
            # wood with the chroma of a black stone's highlight. Use pixel-level
            # disagreement to flag these apparently confident classifications.
            pixel_chroma = np.linalg.norm(centre[:, 1:] - 128, axis=1)
            white_pixels = centre[:, 0] - bg[0] + np.maximum(0., np.linalg.norm(bg[1:] - 128) - pixel_chroma) * 1.6
            white_fraction = np.mean(white_pixels > 28 / sensitivity)
            conflicting_colour = (label == WHITE and dark_fraction > .18) or (label == BLACK and white_fraction > .18)
            partial_stone = label == EMPTY and (dark_fraction > .18 or white_fraction > .18)
            partial_pale_rim = label == EMPTY and pale_surface and dark_fraction > .035
            # Outward rays were removed above: wood border is not a grid line
            # that a pale stone might have covered.
            partial_grid_cover = (label == EMPTY and pale_surface
                                  and np.mean(patch[:, :, 0][trace] > middle[0] - 14) > .18)
            if conflicting_colour or partial_stone or partial_pale_rim or partial_grid_cover:
                confidence[row, col] = min(confidence[row, col], .2)
            if confidence[row, col] < .25:
                uncertain.append((row, col, cx, cy, bg))
    # A second pass leaves confident initial classifications untouched.
    for row, col, x, y, bg in uncertain:
        candidate = _search_offset_stone(padded, x, y, step, bg, sensitivity)
        if candidate is not None and candidate[1] > confidence[row, col]:
            board[row, col], confidence[row, col] = candidate
    return board, confidence


def recognize(image: np.ndarray, size: int = 19, corners=None, sensitivity: float = 1.) -> Recognition:
    quality = None
    if corners is None:
        corners, quality = detect_corners(image, size)
    corrected = rectify(image, corners)
    board, confidence = classify_stones(corrected, size, sensitivity)
    warnings = []
    if quality is not None and quality < .65:
        warnings.append("网格匹配较弱，请核对四角和棋子；必要时手动标定。")
    uncertain = int(np.count_nonzero(confidence < .25))
    if uncertain:
        warnings.append(f"{uncertain} 个交点识别把握较低，请核对橙色标记。")
    return Recognition(board, corrected, order_corners(corners), confidence, quality, warnings)
