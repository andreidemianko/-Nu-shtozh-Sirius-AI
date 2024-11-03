import logging
from telegram import Update, InputMediaPhoto
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes
from telegram.ext.filters import PHOTO
from PIL import Image
from io import BytesIO
import os
from vlm_model import generate_clothing_descriptions_and_search

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("T7335523574:AAF4ay5tJo0rOXFK869RbyEjJyOd1okG44s")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Привет! Отправь мне изображение одежды, и я опишу её и найду похожие товары.'
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(BytesIO(photo_bytes)).convert('RGB')

        descriptions, similar_images = generate_clothing_descriptions_and_search(image)

        response = "Найденные элементы одежды:\n" + "\n".join([f"• {desc}" for desc in descriptions])
        await update.message.reply_text(response)

        for desc, img_path in zip(descriptions, similar_images):
            with open(img_path, 'rb') as img_file:
                await update.message.reply_photo(img_file, caption=f"Похожие товары для: {desc}")

def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(PHOTO, handle_image))

    application.run_polling()

if __name__ == '__main__':
    main()