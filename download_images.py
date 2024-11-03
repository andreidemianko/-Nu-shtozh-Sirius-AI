from datasets import load_dataset
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Папка для сохранения изображений
save_folder = 'laion_rvs_fashion_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Загрузка датасета LAION-RVS-Fashion
dataset = load_dataset("Slep/LAION-RVS-Fashion")

# Загрузка изображений из тренировочного набора
train_data = dataset["train"]

# Скачивание изображений и фильтрация по типу 'SIMPLE' (изолированные продукты)
for item in tqdm(train_data):
    if item["TYPE"] == "SIMPLE":
        url = item["URL"]
        product_id = item["PRODUCT_ID"]
        category = item["CATEGORY"]
        caption = item["blip2_caption1"]

        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            # Сохранение изображения с именем, включающим категорию и product_id
            img_path = os.path.join(save_folder, f"{category}_{product_id}.jpg")
            image.save(img_path)
        except Exception as e:
            print(f"Не удалось загрузить изображение по URL: {url}. Ошибка: {e}")
