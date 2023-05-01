
# API Endpoint for Image Queries

This project deploys an API endpoint using FastAPI and Amazon EC2 to accept two parameters: a query URL with images and a list of URLs corresponding to images. 

## Getting Started

To use this API endpoint, you need to send a POST request to the following URL:

http://[EC2_INSTANCE_IP_ADDRESS]:8000/rank_images

### Prerequisites

- Python 3.7 or higher installed
- FastAPI and Uvicorn packages installed
- An Amazon EC2 instance with a public IP address

### Installing

To install the required packages, run the following command:

pip install fastapi uvicorn

### Starting the Server

To start the server, navigate to the project directory and run the following command:

uvicorn main:app --host [EC2_INSTANCE_IP_ADDRESS] --port 8000

## Using the API

To use the API, you can send a POST request to the `/rank_images` endpoint with the following parameters:

- `query` (string): the URL of the query image
- `links` (list): a list of URLs corresponding to the images to compare against the query image

Example Request:

import requests

query_url = 'https://example.com/query.jpg'
image_urls = ['https://example.com/image1.jpg', 'https://example.com/image2.jpg']

response = requests.post('http://[EC2_INSTANCE_IP_ADDRESS]:8000/rank_images', json={'query': query_url, 'links': image_urls})

print(response.json())

Example Response:

```json
{
    "ranked_images": [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg"
    ]
}
