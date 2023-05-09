from fastapi import FastAPI, HTTPException
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from typing import List
import torch
from PIL import Image
from io import BytesIO
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from requestModel import RequestModel
import asyncio
import aiohttp
import ssl
import base64
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
app = FastAPI()

# Check for GPU availability.
if torch.cuda.is_available():
    device = "cuda"
else:
    print('working with cpu')
    device = "cpu"
data = []

# Load the model.
model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)


async def post_image_urls(image_urls: List[str]):
    endpoint = "https://download-streamed-images-62xutbvi7a-uc.a.run.app"
    headers = {"Content-Type": "application/json"}

    body = {
        "image_urls": image_urls
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, data=json.dumps(body)) as response:
            response_text = await response.text()
            print("Response arrived")
            return response_text




#Extract embedding of a particular image
def extract_embedding(image_bytes: BytesIO) -> torch.Tensor:
    """
        Extracts CLIP image features from image bytes. 

        Args:
        - image_bytes (BytesIO): A BytesIO object containing the image bytes.

        Returns:
        - A torch.Tensor object containing the image features.

        Raises:
        - ValueError: If there is an error opening the image.
    """
    try:
        image = Image.open(image_bytes)
    except Exception as e:
        raise ValueError(f"Error opening image: {e}")

    image = processor(
        text=None,
        images=image,
        return_tensors='pt'
    )['pixel_values'].to(device)
    embedding = model.get_image_features(image)
    return embedding
    

#Download image and generate embedding for target image
def gen_target_embedding(query: str) -> torch.Tensor:
    """
        Extracts the CLIP image features from the target image. 

        Args:
        - query (str): A URL string for the target image.

        Returns:
        - A torch.Tensor object containing the image features.

        Raises:
        - ValueError: If the query URL is empty or contains only whitespace, or if there is an error processing the image.
    """
    if not query.strip():
        raise ValueError("Query URL is empty or contains only whitespace.")

    try:
        # Read the image in as bytes without downloading.
        response = requests.get(query.strip(), timeout=10)
        response.raise_for_status()
        image_bytes = BytesIO(response.content)
        embedding = extract_embedding(image_bytes)
    except Exception as e:
        raise ValueError(f"Error processing {query}: {e}")
    return embedding




#Find matches of images according to image return final result
def get_matches(target_embedding: torch.Tensor, df: pd.DataFrame) -> List[str]:
    """
        Sorts and returns the top matches using the CLIP image features and simple cosine similarity.

        Args:
        - target_embedding (torch.Tensor): A torch.Tensor object containing the image features for the target image.
        - df (pd.DataFrame): A pandas DataFrame containing the image URLs and their corresponding embeddings.

        Returns:
        - A list of URL strings for the ranked images.

        Raises:
        - None.
    """
    # Get the embeddings from the dataframe
    embeddings = df['embedding']

    # Convert the embeddings to a list and append the target embedding.
    embeddings_list = [embedding for embedding in embeddings.values]
    embeddings_list.append(target_embedding)

    # Stack the embeddings and reshape the tensor.
    embeddings_tensor = torch.stack(embeddings_list)
    embeddings_tensor = embeddings_tensor.view(embeddings_tensor.shape[0], -1)

    # Calculate the cosine similarity matrix.
    cos_sim_matrix = cosine_similarity(embeddings_tensor.cpu().detach().numpy())

    # Get the scores for all the images except the target image.
    scores = cos_sim_matrix[-1].tolist()
    scores.pop(-1)

    # Sort the matches based on the scores.
    sorted_matches = sorted(
        range(len(scores)), key=lambda x: scores[x], reverse=True)

    # Get the image URLs for the sorted matches.
    matches = [(j, scores[j]) for j in sorted_matches]
    image_urls = [df.iloc[j]['image_url'] for j, _ in matches]

    image_urls_with_scores = []
    #make list of objects containing image url and score
    for match in matches:
        image_urls_with_scores.append({'image_url': df.iloc[match[0]]['image_url'], 'cosine_similarity_score': match[1]})

    return image_urls_with_scores

#Starting point
@app.post("/rank_images")
async def rank_images(request: RequestModel):
    """
        This function takes in a query URL and a list of image URLs, and returns a list of the image URLs sorted by their similarity to the target image specified by the query URL. 

        Args:
        - query (str): A URL string for the target image.
        - links (List[str]): A list of URL strings for the images to be ranked.

        Returns:
        - A dictionary with a single key "ranked_images" whose value is a list of URL strings for the ranked images.

        Raises:
        - HTTPException with status code 400 and a message detailing the error if any of the following occur:
            - The query URL is empty or contains only whitespace.
            - The list of image URLs is empty.
            - One of the image URLs is empty or contains only whitespace.
            - There is an error opening or processing an image.
    """
    try:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S:%SS")
        print("Start Time =", current_time)
        global data
        data = []
        bytesData = await post_image_urls(request.links)
        bytes_data_json = json.loads(bytesData)
        bytes_data_array = bytes_data_json['data']
        def process_image(image):
            base64_str = image['image_bytes']
            image_bytes = base64.b64decode(base64_str)
            image_bytes_io = BytesIO(image_bytes)
            embedding = extract_embedding(image_bytes_io)
            return {'image_url': image['image_url'], 'embedding': embedding}
        

    # Use ThreadPoolExecutor to extract embeddings in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image) for image in bytes_data_array]
            data = [future.result() for future in as_completed(futures)]


        print('All threads are done!')
        print(len(data))
        df = pd.DataFrame(data)

        #now we make target embedding
        target_embedding = gen_target_embedding(request.query)
        image_urls = get_matches(target_embedding, df)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S:%SS")
        print("Final End Time =", current_time)
        return {"ranked_images": image_urls}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/test')
async def test():
    return {"test": "success"}