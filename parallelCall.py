import aiohttp
import asyncio
import ssl
import base64
import json
from io import BytesIO
from datetime import datetime
url = "https://stream-images-62xutbvi7a-uc.a.run.app/"

async def call_cloud_function(session, image_url):
    headers = {'Content-Type': 'application/json','Authorization':'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImM5YWZkYTM2ODJlYmYwOWViMzA1NWMxYzRiZDM5Yjc1MWZiZjgxOTUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTA4OTAyNjUyMjg4MTEyNjM1MjY1IiwiZW1haWwiOiJ0cml2ZWRpaGFyc2g0OUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXRfaGFzaCI6IjB1d0VIaG9sVjcydHFfS19UVXZjdGciLCJpYXQiOjE2ODMwMTc0NTIsImV4cCI6MTY4MzAyMTA1Mn0.R_VHMZqwAQ9i3H0GmFr_fQcsexr5OB7JfEli-Kdfb6oiVOPmmRlRiN77zWLC0ruWoo-qZNUt8N2wj4oN_uF_IeI8-LumJ-VEOu45fSiM4OFNbZO_JCbqRGoTjLUzruGfN-6wA8KqKu4uQtkQa5xACr_wI0kd9VJz7gPWPQ_f4oIjRHdP2uPfMdgv1yeRQjtHOZG9OUt97LcfUI5LePDjYtF6xt5tB90ptzVt7rWJKgFoK3ubJDvGcslB09-MihUtjfsgBJJGmSBPENH90nXQ3Wx1eoPKUfqYe_rqPGUYDGQSpLgvkKR1cUgIDoepoNCxUI2MnhKa3nexa8I_sASO-Q'}
    async with session.post(url, json={"image_url": image_url},headers=headers) as response:
        # read the actual bytes from the response
        return await response.text()

async def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S:%SS")
    print("Start Time =", current_time)
    # Create an SSL context to bypass SSL verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Read the image URLs from input.txt
    with open('input.txt', 'r') as f:
        image_urls = [line.strip() for line in f]
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = [call_cloud_function(session, image_url) for image_url in image_urls]
        results = await asyncio.gather(*tasks)
        # Print the results of each call
        for result in results:
            #convet result to json object
            json_result = json.loads(result)
            # Convert the string to a BytesIO object
            base64_str = json_result['image_bytes']
            # Decode the base64-encoded string to bytes
            image_bytes = base64.b64decode(base64_str)
            # Create a BytesIO object from the bytes
            image_bytes_io = BytesIO(image_bytes)
            
            
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S:%SS")
        print("End Time =", current_time)
            



# Run the main function
asyncio.run(main())