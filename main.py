"""Start the desktop app, or count a picture / saved board from the CLI."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scoring import count_area
from vision import read_image, recognize
from reporting import export_result


def main():
    parser = argparse.ArgumentParser(description="围棋图片计数：当前棋子 + 独占空点，不推演后续走法。")
    parser.add_argument("image", nargs="?", type=Path, help="棋盘图片；无参数启动图形界面")
    parser.add_argument("--size", type=int, choices=(9, 13, 19), default=19, help="棋盘路数，默认 19")
    parser.add_argument("--corners", nargs=8, type=float, metavar="XY",
                        help="四个最外侧网格交点 x1 y1 x2 y2 x3 y3 x4 y4；以 EXIF 转正后的原图像素为准")
    parser.add_argument("--sensitivity", type=float, default=1., help="棋子灵敏度 0.5～2，默认 1")
    parser.add_argument("--board", type=Path, help="从已有 JSON 的 board 矩阵计数，不进行图像识别")
    parser.add_argument("--neutral", nargs=2, type=int, action="append", default=[], metavar=("ROW", "COL"),
                        help="将指定空点所在空区标为中立；行列从 1 开始，可重复")
    parser.add_argument("--output", type=Path, default=Path("output/result"), help="输出文件名前缀，默认 output/result")
    args = parser.parse_args()
    if args.image is None and args.board is None:
        if args.corners or args.neutral:
            parser.error("--corners / --neutral 需要提供图片或 --board。")
        from gui import launch
        launch()
        return 0
    if args.image and args.board:
        parser.error("图片和 --board 请只选其一。")
    if args.board and args.corners:
        parser.error("--corners 只适用于图片。")
    try:
        recognition = None
        points = set()
        if args.board:
            data = json.loads(args.board.read_text(encoding="utf-8"))
            board = np.asarray(data["board"])
            if len(board) not in (9, 13, 19):
                raise ValueError("仅支持 9、13、19 路棋盘。")
            points = {tuple(p) for p in data.get("neutral_points", [])}
        else:
            image = read_image(args.image)
            corners = np.array(args.corners, dtype=np.float32).reshape(4, 2) if args.corners else None
            recognition = recognize(image, args.size, corners, args.sensitivity)
            board = recognition.board
        points.update((r - 1, c - 1) for r, c in args.neutral)
        score = count_area(board, points)
        paths = export_result(args.output, board, recognition=recognition, source=args.image or args.board,
                              neutral_points=points)
        print(f"黑方：{score.black_area} 点 = {score.black_stones} 子 + {score.black_territory} 空点")
        print(f"白方：{score.white_area} 点 = {score.white_stones} 子 + {score.white_territory} 空点")
        print(f"中立：{score.neutral} 点（不计入双方）")
        if recognition:
            for warning in recognition.warnings:
                print("提示：" + warning, file=sys.stderr)
        print("输出：" + ", ".join(str(p) for p in paths))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
