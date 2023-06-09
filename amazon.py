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
app = FastAPI()

# Check for GPU availability.
if torch.cuda.is_available():
    device = "cuda"
else:
    print('working with cpu')
    device = "cpu"


# Load the model.
model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)


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


def generate_embeddings(links: List[str]) -> pd.DataFrame:
    """
        Generates the CLIP image features for all the images passed into the API. 

        Args:
        - links (List[str]): A list of URL strings for the images to be ranked.

        Returns:
        - A pandas DataFrame containing the image URLs and their corresponding embeddings.

        Raises:
        - ValueError: If the list of image URLs is empty, or if one of the image URLs is empty or contains only whitespace, or if there is an error processing an image.
    """
    if not links or len(links) == 0:
        raise ValueError("The list of image links is empty.")

    df = pd.DataFrame(columns=['image_url', 'embedding'])
    for link in links:
        if not link.strip():
            raise ValueError(
                "One of the image URLs is empty or contains only whitespace.")

        try:
            response = requests.get(link.strip(), timeout=10)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
            embedding = extract_embedding(image_bytes)
            df = pd.concat(
                [df, pd.DataFrame({'image_url': [link], 'embedding': [embedding.detach()]})])
        except Exception as e:
            raise ValueError(f"Error processing {link}: {e}")
    return df


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

    return image_urls


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
        print(request.query)
        print(request.links)
        df = generate_embeddings(request.links)
        target_embedding = gen_target_embedding(request.query)

        image_urls = get_matches(target_embedding, df)

        return {"ranked_images": image_urls}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/test')
async def test():
    return {"test": "success"}