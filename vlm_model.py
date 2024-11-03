import torch
from PIL import Image
import numpy as np
import faiss
import os
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, CLIPProcessor, CLIPModel
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"

# Настройка модели LLaVA-Onevision
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
vlm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
vlm_processor = AutoProcessor.from_pretrained(model_id)

# Настройка модели CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Путь к папке с изображениями
image_folder = 'laion_rvs_fashion_images'


# Функция для создания эмбеддингов изображений
def create_image_embeddings(image_folder):
    image_embeddings = []
    image_paths = []

    for root, dirs, files in os.walk(image_folder):
        for img_name in files:
            img_path = os.path.join(root, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = clip_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    embedding = clip_model.get_image_features(**inputs)
                    embedding /= embedding.norm(p=2, dim=-1, keepdim=True)
                image_embeddings.append(embedding.cpu().numpy())
                image_paths.append(img_path)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")

    if not image_embeddings:
        print("Ошибка: В папке нет подходящих изображений для обработки.")
        return np.array([]), []

    image_embeddings = np.vstack(image_embeddings)
    return image_embeddings, image_paths


# Проверка и создание эмбеддингов
if os.path.exists('image_embeddings.npy') and os.path.exists('image_paths.npy') and os.path.exists('faiss_index.index'):
    image_embeddings = np.load('image_embeddings.npy')
    image_paths = np.load('image_paths.npy', allow_pickle=True)
    index = faiss.read_index('faiss_index.index')
else:
    image_embeddings, image_paths = create_image_embeddings(image_folder)

    if image_embeddings.size == 0:
        raise ValueError("Ошибка: не удалось создать эмбеддинги, так как в папке нет подходящих изображений.")

    np.save('image_embeddings.npy', image_embeddings)
    np.save('image_paths.npy', image_paths)

    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(image_embeddings)
    faiss.write_index(index, 'faiss_index.index')


# Функция для генерации описаний и поиска похожих изображений
def generate_clothing_descriptions_and_search(image):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the clothing items in this image?"},
                {"type": "image"}
            ],
        }
    ]
    prompt = vlm_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = vlm_processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        output = vlm_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generated_text = vlm_processor.decode(output[0][2:], skip_special_tokens=True)

    descriptions_list = [line.strip('• ').strip() for line in generated_text.split("\n") if line.strip()]

    similar_images = []
    for desc in descriptions_list:
        inputs = clip_processor(text=[desc], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs)
            text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy()

        D, I = index.search(text_embedding, k=1)
        similar_image_path = image_paths[I[0][0]]
        similar_images.append(similar_image_path)

    return descriptions_list, similar_images

