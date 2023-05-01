import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import requests
import certifi
import ftfy
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torchvision.transforms.functional as TF
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else \
         ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)

image_dir = "/Users/harsh/mercor/reverseimagesearch/images/"
target_image_path = "/Users/harsh/mercor/reverseimagesearch/target.webp"

def parse_url():
    # Read the content of the input text file
    with open(input_text_file, "r") as file:
        links = file.readlines()

    # Iterate through each link
    for link in links:
        file_url = link.strip()

        # Check if the URL is valid
        if file_url.startswith("https://di2ponv0v5otw.cloudfront.net"):
            # Get the file name from the URL
            file_name = file_url.split("/")[-1]

            # Set the path to save the downloaded image
            download_path = os.path.join(download_directory, file_name)

            # Download and save the image
            try:
                response = requests.get(file_url)
                with open(download_path, "wb") as download_file:
                    download_file.write(response.content)
                print(f"Downloaded '{file_name}' to '{download_path}'")
            except Exception as e:
                print(f"Error downloading '{file_name}': {e}")

def generate_embedding(image_path):
    """Generate an embedding for an image."""
    image = Image.open(image_path)
    image = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to(device)
    img_emb = model.get_image_features(image)
    return img_emb.cpu().detach()

def get_top_3_matches(target_embedding, embeddings, df):
    # convert the Pandas Series object to a list of PyTorch tensors
    embeddings_list = [embedding for embedding in embeddings.values]
    
    # append target embedding to the list
    embeddings_list.append(target_embedding)
    
    # convert the list of embeddings to a 2D tensor
    embeddings_tensor = torch.stack(embeddings_list)
    
    # flatten the tensor along the channel and spatial dimensions
    embeddings_tensor = embeddings_tensor.view(embeddings_tensor.shape[0], -1)
    
    # compute cosine similarity matrix between all embeddings
    cos_sim_matrix = cosine_similarity(embeddings_tensor)
    
    # get cosine similarity scores for target embedding
    scores = cos_sim_matrix[-1].tolist()
    scores.pop(-1)
    
    # get indices of top 3 matches
    top_3_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:30]
    
    # create list of top 3 matches and their cosine similarity scores
    matches = [(j, scores[j]) for j in top_3_indices]
    
    # get the image paths corresponding to the matches
    image_paths = [df.iloc[j]['image_path'] for j, _ in matches]
    
    return image_paths

def generate_embeddings(image_dir):
    """Generate embeddings for all images in a directory."""
    df = pd.DataFrame(columns=['image_path', 'embedding'])
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(image_dir, filename)
            embedding = generate_embedding(image_path)
            df = pd.concat([df, pd.DataFrame({'image_path':[image_path], 'embedding':[embedding]})])
    return df

df = generate_embeddings(image_dir)

# generate embedding for target image
target_embedding = generate_embedding(target_image_path)

# compute top 3 matches for target embedding
top_3_matches = get_top_3_matches(target_embedding, df['embedding'], df)

print(f"Top 3 matches for {target_image_path}:")
for image_path in top_3_matches:
    print(image_path)



