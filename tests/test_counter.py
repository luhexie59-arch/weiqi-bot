import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from scoring import count_area
from vision import classify_stones, recognize, read_image, write_image, order_corners, grid_coordinates
from reporting import export_result
from tests.fixtures import sample_board, render_photo, perspective_photo


class ScoringTests(unittest.TestCase):
    def test_empty_board_and_mixed_open_region(self):
        board = np.zeros((9, 9), np.uint8)
        self.assertEqual(count_area(board).neutral, 81)
        board[0, 0], board[8, 8] = 1, 2
        score = count_area(board)
        self.assertEqual((score.black_area, score.white_area, score.neutral), (1, 1, 79))

    def test_eyes_and_corner_territory(self):
        board = sample_board(19)
        score = count_area(board)
        self.assertEqual((score.black_stones, score.white_stones), (6, 6))
        self.assertEqual((score.black_territory, score.white_territory), (1, 1))
        board = np.array([[0, 1, 2], [1, 1, 2], [2, 2, 2]])
        self.assertEqual(count_area(board).black_territory, 1)

    def test_no_dead_stone_inference(self):
        board = np.ones((5, 5), np.uint8)
        board[2, 2] = 2
        score = count_area(board)
        self.assertEqual((score.black_area, score.white_area, score.neutral), (24, 1, 0))

    def test_whole_region_neutral_override(self):
        board = np.zeros((9, 9), np.uint8)
        board[0, 0] = 1
        self.assertEqual(count_area(board).black_area, 81)
        score = count_area(board, [(4, 4)])
        self.assertEqual((score.black_area, score.neutral), (1, 80))

    def test_random_positions_conserve_area(self):
        rng = np.random.default_rng(48)
        for n in (9, 13, 19):
            for _ in range(25):
                score = count_area(rng.choice([0, 1, 2], (n, n), p=[.7, .15, .15]))
                self.assertEqual(score.black_area + score.white_area + score.neutral, n * n)

    def test_invalid_values_do_not_get_truncated(self):
        for board in ([[0, .5], [1, 2]], [[0, 3], [1, 2]], [[0, 1, 2]]):
            with self.assertRaises(ValueError):
                count_area(board)
        with self.assertRaises(ValueError):
            count_area([[1, 0], [0, 2]], [(0, 0)])


class VisionTests(unittest.TestCase):
    def test_classification_with_hoshi_edges_lighting_and_glare(self):
        for size in (9, 13, 19):
            for shaded in (False, True):
                with self.subTest(size=size, shaded=shaded):
                    expected = sample_board(size)
                    image = render_photo(expected, shaded=shaded, glare=True)
                    actual, confidence = classify_stones(image, size)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(np.isfinite(confidence).all())

    def test_pale_screenshot(self):
        expected = sample_board()
        for shaded in (False, True):
            actual, _ = classify_stones(render_photo(expected, pale=True, shaded=shaded), 19)
            np.testing.assert_array_equal(actual, expected)

    def test_dense_position_with_manual_corners(self):
        rng = np.random.default_rng(10)
        board = rng.choice([0, 1, 2], (19, 19), p=[.25, .4, .35]).astype(np.uint8)
        photo, corners = perspective_photo(board)
        actual = recognize(photo, 19, corners)
        np.testing.assert_array_equal(actual.board, board)

    def test_manual_perspective_correction(self):
        expected = sample_board()
        photo, corners = perspective_photo(expected)
        actual = recognize(photo, 19, corners)
        np.testing.assert_array_equal(actual.board, expected)

    def test_automatic_perspective_detection(self):
        expected = sample_board()
        photo, corners = perspective_photo(expected)
        actual = recognize(photo, 19)
        self.assertLess(float(np.max(np.linalg.norm(actual.corners - corners, axis=1))), 14.)
        np.testing.assert_array_equal(actual.board, expected)

    def test_automatic_flat_all_sizes(self):
        for size in (9, 13, 19):
            with self.subTest(size=size):
                expected = sample_board(size)
                actual = recognize(render_photo(expected), size)
                np.testing.assert_array_equal(actual.board, expected)

    def test_invalid_and_blank_images(self):
        with self.assertRaises(ValueError):
            recognize(np.full((800, 800, 3), 180, np.uint8))
        with self.assertRaises(ValueError):
            order_corners([[0, 0], [0, 0], [100, 100], [100, 0]])

    def test_unicode_image_and_export_roundtrip(self):
        board = sample_board(9)
        scratch = Path(__file__).resolve().parents[1] / "output" / "test-temp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "中文棋盘.png"
            write_image(path, render_photo(board))
            result = recognize(read_image(path), 9)
            paths = export_result(Path(directory) / "结果", result.board, recognition=result)
            self.assertTrue(all(p.is_file() for p in paths))
            data = json.loads(paths[0].read_text(encoding="utf-8"))
            np.testing.assert_array_equal(data["board"], board)
            self.assertEqual(data["score"]["black"]["area"], count_area(board).black_area)

    def test_camera_exif_orientation(self):
        scratch = Path(__file__).resolve().parents[1] / "output" / "test-temp"
        scratch.mkdir(parents=True, exist_ok=True)
        board = sample_board(13)
        image = Image.fromarray(cv2.cvtColor(render_photo(board), cv2.COLOR_BGR2RGB))
        exif = Image.Exif()
        exif[274] = 6  # Stored image is CCW; camera requests CW display rotation.
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "camera.jpg"
            image.transpose(Image.Transpose.ROTATE_90).save(path, quality=96, exif=exif)
            grid = grid_coordinates(13)
            corners = [[grid[0], grid[0]], [grid[-1], grid[0]],
                       [grid[-1], grid[-1]], [grid[0], grid[-1]]]
            actual = recognize(read_image(path), 13, corners)
            np.testing.assert_array_equal(actual.board, board)


if __name__ == "__main__":
    unittest.main()
