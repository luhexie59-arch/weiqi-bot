"""Current-position area counting. No life/death search or move prediction."""
from dataclasses import dataclass
from collections import deque

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2


@dataclass
class Score:
    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    neutral: int
    ownership: np.ndarray

    @property
    def black_area(self) -> int:
        return self.black_stones + self.black_territory

    @property
    def white_area(self) -> int:
        return self.white_stones + self.white_territory

    def as_dict(self) -> dict:
        return {
            "black": {"stones": self.black_stones, "territory": self.black_territory,
                      "area": self.black_area},
            "white": {"stones": self.white_stones, "territory": self.white_territory,
                      "area": self.white_area},
            "neutral": self.neutral,
            "ownership": self.ownership.tolist(),
        }


def count_area(board, neutral_points=()) -> Score:
    """Flood-fill empty regions. Board edges are walls, not an extra colour.

    A region touching exactly one stone colour belongs to that colour. Regions
    touching both colours or no stones remain neutral. Existing stones count
    as occupied, even if a human might judge them dead or part of a seki.
    """
    board = np.asarray(board)
    if board.ndim != 2 or board.shape[0] != board.shape[1] or board.shape[0] < 2:
        raise ValueError("棋盘必须是至少 2×2 的方阵。")
    if not np.isin(board, [EMPTY, BLACK, WHITE]).all():
        raise ValueError("棋盘只能包含 0（空）、1（黑）、2（白）。")
    n = len(board)
    forced_neutral = set()
    for r, c in neutral_points:
        if not (0 <= r < n and 0 <= c < n) or board[r, c] != EMPTY:
            raise ValueError("中立标记必须位于棋盘内的空交点。")
        forced_neutral.add((int(r), int(c)))
    ownership = board.astype(np.uint8, copy=True)
    visited = board != EMPTY
    territories = {EMPTY: 0, BLACK: 0, WHITE: 0}
    for row in range(n):
        for col in range(n):
            if visited[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            region, neighbours = [], set()
            while queue:
                r, c = queue.popleft()
                region.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < n and 0 <= nc < n):
                        continue
                    value = int(board[nr, nc])
                    if value:
                        neighbours.add(value)
                    elif not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            owner = next(iter(neighbours)) if len(neighbours) == 1 else EMPTY
            if forced_neutral.intersection(region):
                owner = EMPTY
            territories[owner] += len(region)
            for r, c in region:
                ownership[r, c] = owner
    return Score(int(np.count_nonzero(board == BLACK)), int(np.count_nonzero(board == WHITE)),
                 territories[BLACK], territories[WHITE], territories[EMPTY], ownership)
