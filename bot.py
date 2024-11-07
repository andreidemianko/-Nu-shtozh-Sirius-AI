import logging
import nest_asyncio
from telegram import Update, InputMediaPhoto
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes
from telegram.ext.filters import PHOTO
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import requests
import os
from huggingface_hub import login

login("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")
BOT_TOKEN = "7335523574:AAFvi2gY4e5ZCRCvQI41AXFSg0DfMQFyvxY"

# Настройка Nest Asyncio для Colab;)
nest_asyncio.apply()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Настройка устройств и моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Настройка модели VLM
model_id_vlm = "google/paligemma-3b-mix-224"
model_vlm = PaliGemmaForConditionalGeneration.from_pretrained(model_id_vlm, torch_dtype=torch.bfloat16)
processor_vlm = PaliGemmaProcessor.from_pretrained(model_id_vlm)

def discribe_picture(input_image):
    promt = (
        "Create a detailed description of each item of clothing visible in the image. "
        "For each item, specify its type, color, pattern (if any), and style, as well as any relevant details like fit or unique features. "
        "The response should be formatted as a list, with each item description starting with a bullet point (•). "
        "If there is a person in the photo, do not describe them."
    )
    inputs = processor_vlm(text=promt, images=input_image, padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)
    model_vlm.to(device)
    inputs = inputs.to(dtype=model_vlm.dtype)

    with torch.no_grad():
        output = model_vlm.generate(**inputs, max_length=496)

    decoded_output = processor_vlm.decode(output[0], skip_special_tokens=True)
    
    if decoded_output.startswith(promt):
        decoded_output = decoded_output[len(promt):].strip()

    return decoded_output


# Загрузка и обработка датасета
dataset = load_dataset("ceyda/fashion-products-small", split="train")

# Функция для создания эмбеддингов изображений
def create_image_embeddings(dataset):
    image_embeddings = []
    image_paths = []

    for item in dataset:
        try:
            image_url = item["link"]
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                embedding = clip_model.get_image_features(**inputs)
                embedding /= embedding.norm(p=2, dim=-1, keepdim=True)
            image_embeddings.append(embedding.cpu().numpy())
            image_paths.append(image_url) 
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {image_url}: {e}")
    
    return np.vstack(image_embeddings), image_paths

# Создание или загрузка эмбеддингов для датасета
if os.path.exists("image_embeddings.npy") and os.path.exists("image_paths.npy"):
    image_embeddings = np.load("image_embeddings.npy")
    image_paths = np.load("image_paths.npy", allow_pickle=True)
else:
    subset = dataset.select(range(500))
    image_embeddings, image_paths = create_image_embeddings(subset)
    np.save("image_embeddings.npy", image_embeddings)
    np.save("image_paths.npy", image_paths)

# Создание индекса для быстрого поиска
dimension = image_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(image_embeddings)

# Функция для поиска наиболее подходящего изображения для каждого описания
def search_similar_images_by_description(descriptions, top_k=1):
    results = []
    
    for description in descriptions:
        inputs = clip_processor(text=[description], return_tensors="pt").to(device)
        
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs)
            text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)
        
        D, I = index.search(text_embedding.cpu().numpy(), top_k)
        similar_image = image_paths[I[0][0]]  # Получаем URL изображения с наивысшим баллом
        results.append((description, similar_image))
    
    return results


# Обработчик изображений
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            input_image = Image.open(BytesIO(photo_bytes)).convert('RGB')
            
            # Генерация описания с помощью VLM модели
            description = discribe_picture(input_image)
            await update.message.reply_text(f"Описание изображения: {description}")
            
            # Разделение описания на отдельные элементы, если это список строк
            descriptions_list = description.split(';')  # Измените разделитель в зависимости от формата выхода VLM
            
            # Поиск изображений, соответствующих каждому описанию
            search_results = search_similar_images_by_description(descriptions_list)
            
            for desc, img_url in search_results:
                await update.message.reply_text(f"Для описания \"{desc}\", найдено изображение:")
                await update.message.reply_photo(img_url)

        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}")
            await update.message.reply_text("Произошла ошибка при обработке изображения. Пожалуйста, попробуйте снова.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Привет! Отправь мне изображение одежды, и я найду похожие товары из ceyda/fashion-products-small.'
    )

def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(PHOTO, handle_image))

    logger.info("Бот запущен и готов к работе...")
    application.run_polling()

if __name__ == '__main__':
    main()
