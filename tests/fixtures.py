"""Reproducible artificial photos; no external sample images are required."""
import cv2
import numpy as np

from vision import BOARD_PIXELS, MARGIN, grid_coordinates


def sample_board(size=19):
    board = np.zeros((size, size), np.uint8)
    # Separate black/white eyes, both edge and centre stones, neutral outside.
    for r, c in ((0, 0), (0, size - 1), (2, 3), (3, 2), (3, 4), (4, 3)):
        board[r, c] = 1
    for r, c in ((size - 1, 0), (size - 1, size - 1), (size - 4, size - 3),
                 (size - 3, size - 4), (size - 3, size - 2), (size - 2, size - 3)):
        board[r, c] = 2
    return board


def render_photo(board, shaded=False, pale=False, glare=False, offsets=None):
    n = len(board)
    side = BOARD_PIXELS
    colour = (244, 244, 244) if pale else (135, 185, 222)
    image = np.full((side, side, 3), colour, np.uint8)
    grid = grid_coordinates(n)
    step = grid[1] - grid[0]
    for p in grid:
        p = round(p)
        cv2.line(image, (MARGIN, p), (side - MARGIN - 1, p), (55, 65, 75), 2, cv2.LINE_AA)
        cv2.line(image, (p, MARGIN), (p, side - MARGIN - 1), (55, 65, 75), 2, cv2.LINE_AA)
    stars = [2, 4, 6] if n == 9 else ([3, 6, 9] if n == 13 else [3, 9, 15])
    for r in stars:
        for c in stars:
            cv2.circle(image, (round(grid[c]), round(grid[r])), 4, (55, 65, 75), -1)
    for r, c in np.argwhere(board != 0):
        dx, dy = (offsets or {}).get((int(r), int(c)), (0., 0.))
        point = (round(grid[c] + dx * step), round(grid[r] + dy * step))
        value = 27 if board[r, c] == 1 else 249
        cv2.circle(image, point, round(step * .44), (value, value, value), -1, cv2.LINE_AA)
        cv2.circle(image, point, round(step * .44), (20, 20, 20) if board[r, c] == 1 else (130, 130, 130), 1, cv2.LINE_AA)
        if glare and board[r, c] == 1:
            cv2.circle(image, (point[0] - round(step * .12), point[1] - round(step * .12)),
                       max(2, round(step * .065)), (225, 225, 225), -1, cv2.LINE_AA)
    if shaded:
        yy, xx = np.mgrid[:side, :side]
        lighting = .62 + .37 * xx / side + .05 * np.sin(yy / 140)
        rng = np.random.default_rng(123)
        image = np.clip(image.astype(float) * lighting[:, :, None] + rng.normal(0, 1.2, image.shape), 0, 255).astype(np.uint8)
    return image


def perspective_photo(board, offsets=None):
    raw = render_photo(board, shaded=True, glare=True, offsets=offsets)
    side = BOARD_PIXELS
    source = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
    target = np.float32([[160, 85], [990, 180], [1060, 950], [75, 1040]])
    transform = cv2.getPerspectiveTransform(source, target)
    photo = cv2.warpPerspective(raw, transform, (1160, 1120), borderValue=(52, 57, 63))
    first, last = MARGIN, side - MARGIN - 1
    corners = np.float32([[first, first], [last, first], [last, last], [first, last]])
    corners = cv2.perspectiveTransform(corners[None], transform)[0]
    return photo, corners
