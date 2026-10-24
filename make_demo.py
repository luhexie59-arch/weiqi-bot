"""Generate an artificial camera-like example with a known correct answer."""
import json
from pathlib import Path

from reporting import export_result
from tests.fixtures import sample_board, perspective_photo
from vision import write_image


def main():
    target = Path(__file__).parent / "examples"
    target.mkdir(exist_ok=True)
    board = sample_board()
    image, corners = perspective_photo(board)
    write_image(target / "demo_photo.png", image)
    export_result(target / "expected", board)
    (target / "corners.json").write_text(json.dumps(corners.tolist(), indent=2), encoding="utf-8")
    print("已生成 examples/demo_photo.png 和 expected.json / expected.png。")


if __name__ == "__main__":
    main()
