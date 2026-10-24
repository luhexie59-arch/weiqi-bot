"""Offset-stone regressions: whole-board comparisons catch duplicate counts."""
import unittest
from unittest.mock import patch

import numpy as np

import vision
from tests.fixtures import render_photo, sample_board, perspective_photo


class OffsetStoneTests(unittest.TestCase):
    def test_diagonal_offset_is_retried_instead_of_confidently_empty(self):
        for colour in (1, 2):
            with self.subTest(colour=colour):
                board = np.zeros((19, 19), np.uint8)
                board[7, 7] = colour
                photo = render_photo(board, offsets={(7, 7): (1/3, 1/3)})
                with patch('vision._search_offset_stone', return_value=None) as search:
                    initial, confidence = vision.classify_stones(photo, 19)
                self.assertEqual(initial[7, 7], 0)
                self.assertLess(confidence[7, 7], .25)
                self.assertTrue(search.called)
                actual, _ = vision.classify_stones(photo, 19)
                np.testing.assert_array_equal(actual, board)

    def test_both_colours_across_search_square_edges_corners_and_interior(self):
        positions = [(0, 0), (0, 18), (18, 0), (18, 18), (3, 3), (3, 9), (3, 15),
                     (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        directions = [(-1/3, -1/3), (1/3, -1/3), (-1/3, 1/3), (1/3, 1/3),
                      (-1/3, 0), (1/3, 0), (0, -1/3), (0, 1/3),
                      (.24, -.28), (-.19, .29), (.08, -.12)]
        offsets = {point: directions[i % len(directions)] for i, point in enumerate(positions)}
        for pale in (False, True):
            for shaded in (False, True):
                for reverse in (False, True):
                    with self.subTest(pale=pale, shaded=shaded, reverse=reverse):
                        board = np.zeros((19, 19), np.uint8)
                        for i, point in enumerate(positions):
                            board[point] = 1 + (i + reverse) % 2
                        image = render_photo(board, pale=pale, shaded=shaded, glare=True, offsets=offsets)
                        actual, confidence = vision.classify_stones(image, 19)
                        np.testing.assert_array_equal(actual, board)
                        self.assertTrue(np.isfinite(confidence).all())

    def test_adjacent_shifted_stones_are_not_counted_twice(self):
        offsets = {(7, 7): (1/3, 1/3), (7, 8): (1/3, 1/3)}
        for pale in (False, True):
            for colours in ((1, 1), (2, 2), (1, 2), (2, 1)):
                with self.subTest(pale=pale, colours=colours):
                    board = np.zeros((19, 19), np.uint8)
                    board[7, 7], board[7, 8] = colours
                    actual, _ = vision.classify_stones(render_photo(board, pale=pale, offsets=offsets), 19)
                    np.testing.assert_array_equal(actual, board)

    def test_empty_board_and_hoshi_are_not_promoted_to_stones(self):
        for size in (9, 13, 19):
            for pale in (False, True):
                for shaded in (False, True):
                    with self.subTest(size=size, pale=pale, shaded=shaded):
                        board = np.zeros((size, size), np.uint8)
                        actual, _ = vision.classify_stones(render_photo(board, pale=pale, shaded=shaded), size)
                        np.testing.assert_array_equal(actual, board)

    def test_confident_intersections_skip_expanded_search(self):
        board = np.zeros((19, 19), np.uint8)
        board[4, 4], board[12, 12] = 1, 2
        with patch('vision._search_offset_stone', wraps=vision._search_offset_stone) as search:
            actual, _ = vision.classify_stones(render_photo(board), 19)
        np.testing.assert_array_equal(actual, board)
        search.assert_not_called()

    def test_offset_stones_after_camera_perspective_correction(self):
        board = sample_board(19)
        offsets = {tuple(point): (1/3, 1/3) for point in np.argwhere(board != 0)}
        photo, corners = perspective_photo(board, offsets=offsets)
        for manual in (False, True):
            with self.subTest(manual=manual):
                result = vision.recognize(photo, 19, corners if manual else None)
                np.testing.assert_array_equal(result.board, board)

    def test_search_scales_with_nine_and_thirteen_line_boards(self):
        for size in (9, 13):
            for colour in (1, 2):
                with self.subTest(size=size, colour=colour):
                    board = np.zeros((size, size), np.uint8)
                    board[4, 4] = colour
                    photo = render_photo(board, shaded=True, offsets={(4, 4): (1/3, -1/3)})
                    actual, _ = vision.classify_stones(photo, size)
                    np.testing.assert_array_equal(actual, board)


if __name__ == '__main__':
    unittest.main()
