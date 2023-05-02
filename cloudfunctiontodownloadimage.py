import requests
from io import BytesIO
from flask import jsonify
import base64

def convert_url_to_bytes_stream(request):
    data = request.get_json()
    image_url = data['image_url']
    response = requests.get(image_url)
    bytes_stream = BytesIO(response.content)
    base64_bytes = base64.b64encode(bytes_stream.read())
    # Decode the bytes to a string before passing to jsonify
    base64_str = base64_bytes.decode('utf-8')
    return jsonify({'image_bytes': base64_str, 'image_url': image_url, 'status': 'success', 'size': len(base64_bytes)})
