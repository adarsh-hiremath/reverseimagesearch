import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd

model, preprocess = torch.hub.load(
    'openai/clip', 'clip-VIT-B/32', pretrained=True)

image_dir = '/path/to/image/directory'


def generate_embedding(image_path):
    """Generate an embedding for an image."""
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    embedding = model.encode_image(image)
    return embedding


def generate_embeddings(image_dir):
    """Generate embeddings for all images in a directory."""
    df = pd.DataFrame(columns=['image_path', 'embedding'])
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            embedding = generate_embedding(image_path)
            df = df.append({'image_path': image_path,
                           'embedding': embedding}, ignore_index=True)
    print(df)
    return df


if __name__ == '__main__':
    generate_embeddings(image_dir)
