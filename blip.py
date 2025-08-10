from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os

# Load BLIP-2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Folder of images
image_folder = '/content/images'
results = []

# Generate captions
for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    image_path = os.path.join(image_folder, image_name)
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(images=raw_image, return_tensors="pt").to(model.device, torch.float16)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    results.append((image_name, caption))
    print(f"{image_name}: {caption}")

# Save to CSV/Excel
import pandas as pd
df = pd.DataFrame(results, columns=['Image ID', 'BLIP-2 Caption'])
df.to_excel("blip2_captions.xlsx", index=False)
