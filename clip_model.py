from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)

import torch
from PIL import Image

image = [Image.open("docs/fashion-hippo.jpg")]
text = ["a hat", "a t-shirt", "shoes"]
processed = processor(text=text, images=image, padding='max_length', return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(processed['pixel_values'], normalize=True)
    text_features = model.get_text_features(processed['input_ids'], normalize=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
# [0.99990773, 0.00006382, 0.00002847]
