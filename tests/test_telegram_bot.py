"""Offline bot tests. No Telegram servers or real bot tokens are used."""
from io import BytesIO, StringIO
import logging
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
from PIL import Image
from telegram import Message, Update
from telegram.error import Conflict, NetworkError
from telegram.ext import ExtBot

import telegram_bot
from tests.fixtures import sample_board, perspective_photo

FAKE_TOKEN = "123456789:OFFLINE_TEST_TOKEN_NOT_REAL"
PNG = b"\x89PNG\r\n\x1a\n"


def incoming(bot, message_id=10, **content):
    return Update.de_json({
        "update_id": message_id,
        "message": {"message_id": message_id, "date": 1,
                    "chat": {"id": 123, "type": "private"}, **content},
    }, bot)


def photo_message(bot, message_id=10, **content):
    return incoming(bot, message_id, photo=[
        {"file_id": "large", "file_unique_id": "large-unique", "width": 1200, "height": 1200, "file_size": 1000},
        {"file_id": "small", "file_unique_id": "small-unique", "width": 100, "height": 100, "file_size": 100},
    ], **content)


class ImageProcessingTests(unittest.TestCase):
    def test_photo_bytes_to_annotated_png_and_exact_area_caption(self):
        photo, _ = perspective_photo(sample_board(19))
        ok, encoded = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertTrue(ok)
        picture, caption = telegram_bot.process_photo(encoded.tobytes())
        self.assertEqual(caption, "黑子：7\n白子: 7")  # Areas, not six stones.
        self.assertTrue(picture.startswith(PNG))
        with Image.open(BytesIO(picture)) as image:
            self.assertEqual(image.size, (900, 900))
        decoded = cv2.imdecode(np.frombuffer(picture, np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(np.any(np.all(decoded == (242, 204, 73), axis=2)))  # Cyan black-stone rings.

    def test_token_redacted_from_log_message_and_traceback(self):
        formatter = telegram_bot.TokenRedactingFormatter(FAKE_TOKEN)
        try:
            raise ValueError("https://example.test/bot" + FAKE_TOKEN)
        except ValueError:
            import sys
            record = logging.LogRecord("test", logging.ERROR, __file__, 1, "failed %s", (FAKE_TOKEN,), sys.exc_info())
        formatted = formatter.format(record)
        self.assertNotIn(FAKE_TOKEN, formatted)
        self.assertIn("[REDACTED]", formatted)


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = ExtBot(FAKE_TOKEN)
        self.remote_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"input")))
        self.get_file = AsyncMock(return_value=self.remote_file)
        self.reply_photo = AsyncMock()
        self.reply_text = AsyncMock()
        self.process = MagicMock(return_value=(PNG + b"annotation", "黑子：7\n白子: 7"))
        for target, value in (("telegram.ext.ExtBot.get_file", self.get_file),
                              ("telegram.Message.reply_photo", self.reply_photo),
                              ("telegram.Message.reply_text", self.reply_text),
                              ("telegram_bot.process_photo", self.process)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_highest_resolution_used_and_one_combined_reply(self):
        update = photo_message(self.bot)
        await telegram_bot.handle_photo(update, SimpleNamespace())
        self.assertEqual(self.get_file.await_args.kwargs["file_id"], "large")
        self.process.assert_called_once_with(bytearray(b"input"))
        self.reply_photo.assert_awaited_once()
        sent = self.reply_photo.await_args.kwargs
        self.assertEqual(sent["caption"], "黑子：7\n白子: 7")
        self.assertEqual(sent["photo"].input_file_content, PNG + b"annotation")
        self.assertNotIn("reply_markup", sent)
        self.reply_text.assert_not_awaited()

    async def test_each_album_photo_produces_its_own_result(self):
        for message_id in (10, 11):
            await telegram_bot.handle_photo(photo_message(self.bot, message_id, media_group_id="album"), SimpleNamespace())
        self.assertEqual(self.process.call_count, 2)
        self.assertEqual(self.reply_photo.await_count, 2)

    async def test_uncompressed_image_document_supported(self):
        update = incoming(self.bot, document={"file_id": "document", "file_unique_id": "doc-unique",
                                             "mime_type": "image/png", "file_size": 1000})
        await telegram_bot.handle_photo(update, SimpleNamespace())
        self.assertEqual(self.get_file.await_args.kwargs["file_id"], "document")
        self.reply_photo.assert_awaited_once()

    async def test_text_commands_and_nonimages_ignored(self):
        for content in ({"text": "/start"}, {"text": "hello"},
                        {"document": {"file_id": "pdf", "file_unique_id": "pdf-unique", "mime_type": "application/pdf"}}):
            await telegram_bot.handle_photo(incoming(self.bot, **content), SimpleNamespace())
        self.get_file.assert_not_awaited()
        self.reply_photo.assert_not_awaited()
        self.reply_text.assert_not_awaited()

    async def test_invalid_image_reports_failure_without_fake_counts(self):
        self.process.side_effect = ValueError("invalid board")
        with self.assertLogs("telegram_bot", logging.WARNING):
            await telegram_bot.handle_photo(photo_message(self.bot), SimpleNamespace())
        self.reply_photo.assert_not_awaited()
        self.reply_text.assert_awaited_once()
        self.assertIn("识别失败", self.reply_text.await_args.args[0])

    async def test_download_error_returns_short_failure(self):
        self.remote_file.download_as_bytearray.side_effect = NetworkError("offline")
        with self.assertLogs("telegram_bot", logging.WARNING):
            await telegram_bot.handle_photo(photo_message(self.bot), SimpleNamespace())
        self.process.assert_not_called()
        self.reply_photo.assert_not_awaited()
        self.reply_text.assert_awaited_once_with("图片下载失败，请稍后重发。")

    async def test_oversized_photo_rejected_before_download(self):
        update = incoming(self.bot, document={"file_id": "huge", "file_unique_id": "huge-unique",
                                             "mime_type": "image/jpeg", "file_size": telegram_bot.MAX_FILE_BYTES + 1})
        await telegram_bot.handle_photo(update, SimpleNamespace())
        self.get_file.assert_not_awaited()
        self.reply_text.assert_awaited_once()

    async def test_duplicate_polling_stops_this_instance(self):
        application = MagicMock()
        context = SimpleNamespace(error=Conflict("another instance"), application=application)
        with self.assertLogs("telegram_bot", logging.ERROR):
            await telegram_bot.handle_error(None, context)
        application.stop_running.assert_called_once()


class StartupTests(unittest.TestCase):
    def test_missing_token_exits_without_network_or_reusing_other_bots(self):
        with patch.dict(os.environ, {"TGBOT_ABETTERAIRCON_API_KEY": FAKE_TOKEN}, clear=True), \
                patch("telegram_bot.create_application") as build, patch("sys.stderr", new_callable=StringIO):
            self.assertEqual(telegram_bot.main(), 1)
            build.assert_not_called()

    def test_polling_configuration_and_token_selection_without_network(self):
        application = MagicMock()
        with patch.dict(os.environ, {"TGBOT_WEIQI_API_KEY": FAKE_TOKEN}, clear=True), \
                patch("telegram_bot.create_application", return_value=application) as build, \
                patch("telegram_bot.logging.basicConfig"):
            self.assertEqual(telegram_bot.main(), 0)
        build.assert_called_once_with(FAKE_TOKEN)
        arguments = application.run_polling.call_args.kwargs
        self.assertEqual(arguments["timeout"], 30)
        self.assertEqual(arguments["allowed_updates"], ["message"])
        self.assertFalse(arguments["drop_pending_updates"])

    def test_actual_application_has_only_image_handler(self):
        application = telegram_bot.create_application(FAKE_TOKEN)
        handlers = application.handlers[0]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(application.update_processor.max_concurrent_updates, 1)
        bot = application.bot
        self.assertTrue(handlers[0].check_update(photo_message(bot)))
        self.assertFalse(handlers[0].check_update(incoming(bot, text="/start")))


if __name__ == "__main__":
    unittest.main()
