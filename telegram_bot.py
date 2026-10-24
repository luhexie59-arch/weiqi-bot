"""19x19 Go photo bot: long polling, one annotated photo reply per image.

Token: TGBOT_WEIQI_API_KEY (or TELEGRAM_BOT_TOKEN). No commands or buttons.
"""
import asyncio
from io import BytesIO
import logging
import os
import signal
import sys

import cv2
from telegram import InputFile, Update
from telegram.error import Conflict, InvalidToken, TelegramError
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from reporting import draw_board
from scoring import count_area
from vision import read_image, recognize

LOGGER = logging.getLogger(__name__)
MAX_FILE_BYTES = 20 * 1024 * 1024


def process_photo(data: bytes | bytearray) -> tuple[bytes, str]:
    """CPU-only processing, with no Telegram calls or temporary photo files."""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("图片超过 20 MB。")
    with BytesIO(data) as source:
        image = read_image(source)
    result = recognize(image, size=19)
    score = count_area(result.board)
    annotated = draw_board(result.board, confidence=result.confidence, photo=result.corrected)
    success, encoded = cv2.imencode(".png", annotated)
    if not success:
        raise ValueError("无法生成标注图片。")
    caption = f"黑子: {score.black_area}\n白子: {score.white_area}"
    return encoded.tobytes(), caption


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Use only new messages: editing captions must not trigger another result.
    message = update.message
    if message is None:
        return
    if message.photo:
        media = max(message.photo, key=lambda photo: (photo.width * photo.height, photo.file_size or 0))
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        media = message.document
    else:
        return

    if (media.file_size or 0) > MAX_FILE_BYTES:
        await message.reply_text("图片过大，请压缩至 20 MB 以内后重发。")
        return
    try:
        remote_file = await media.get_file()
        data = await remote_file.download_as_bytearray()
    except TelegramError as exc:
        LOGGER.warning("Photo download failed, message_id=%s (%s)", message.message_id, type(exc).__name__)
        await message.reply_text("图片下载失败，请稍后重发。")
        return

    try:
        # Polling and networking remain responsive while OpenCV works. Incoming
        # photos are processed sequentially to bound CPU/RAM and preserve order.
        annotated, caption = await asyncio.to_thread(process_photo, data)
    except Exception as exc:
        LOGGER.warning("Photo recognition failed, message_id=%s (%s)", message.message_id, type(exc).__name__)
        await message.reply_text("识别失败，请发送清晰、完整的 19×19 棋盘照片。")
        return

    # One reply contains both the actual photo with annotations and the exact
    # two-line result. Do not send progress messages, keyboards or menus.
    await message.reply_photo(
        photo=InputFile(annotated, filename="go_count.png"),
        caption=caption,
        do_quote=True,
    )


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Avoid raw exception URLs: Telegram download/request URLs contain tokens.
    LOGGER.error("Telegram update failed (%s)", type(context.error).__name__)
    if isinstance(context.error, Conflict):
        LOGGER.error("Another polling instance is active. Stop the duplicate instance before restarting.")
        context.application.stop_running()


def create_application(token: str) -> Application:
    application = (
        Application.builder()
        .token(token)
        .concurrent_updates(False)
        .connect_timeout(15)
        .read_timeout(60)
        .write_timeout(60)
        .pool_timeout(15)
        .get_updates_connect_timeout(15)
        .get_updates_read_timeout(15)
        .build()
    )
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
    application.add_error_handler(handle_error)
    return application


class TokenRedactingFormatter(logging.Formatter):
    def __init__(self, token: str):
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.token = token

    def format(self, record):
        return super().format(record).replace(self.token, "[REDACTED]")


def main() -> int:
    token = os.environ.get("TGBOT_WEIQI_API_KEY", "").strip() or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("请先设置环境变量 TGBOT_WEIQI_API_KEY（或 TELEGRAM_BOT_TOKEN）。", file=sys.stderr)
        return 1
    handler = logging.StreamHandler()
    handler.setFormatter(TokenRedactingFormatter(token))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        application = create_application(token)
        LOGGER.info("Starting 19x19 Go photo bot (long polling). Ctrl+C to stop.")
        application.run_polling(
            timeout=30,
            allowed_updates=["message"],
            drop_pending_updates=False,
            bootstrap_retries=3,
            stop_signals=None if os.name == "nt" else (signal.SIGINT, signal.SIGTERM),
        )
    except InvalidToken:
        LOGGER.error("Bot token is invalid. Check TGBOT_WEIQI_API_KEY / TELEGRAM_BOT_TOKEN.")
        return 1
    except (TelegramError, OSError) as exc:
        LOGGER.error("Bot could not start (%s). Check network access and token configuration.", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
