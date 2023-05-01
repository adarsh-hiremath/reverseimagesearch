# Image Search API

This API allows you to rank a list of images according to their similarity to a given query image. To use this API, follow the instructions below.

## Calling the API

You can call this API using either Python or Curl.

### Python Example

Here's an example Python code snippet that shows how to call the API:

```python
import requests
import json

def main():
    url = "https://imagesearch.backend.mercor.io/rank_images"
    body = {
        "query": "https://nb.scene7.com/is/image/NB/bbw550bb_nb_02_i?$dw_detail_main_lg$&bgc=f1f1f1&layer=1&bgcolor=f1f1f1&blendMode=mult&scale=10&wid=1600&hei=1600",
        "links": [
            "https://di2ponv0v5otw.cloudfront.net/posts/2022/05/04/6272f4a13751f5ea760832fb/s_wp_6272fa5c941f175a1ce82807.webp",
            "https://di2ponv0v5otw.cloudfront.net/posts/2022/11/10/636da131046d74db9f3b6e49/s_wp_636da1b1dff94d691895db7f.webp"
        ]
    }

    headers = {
        'Content-Type': 'application/json'
    }
    res = requests.post(url, data=json.dumps(body), headers=headers)
    print(res.json())

if __name__ == "__main__":
    main()```
    
### Curl Example

Here's an example Curl command that shows how to call the API:

```bash
curl --location 'https://imagesearch.backend.mercor.io/rank_images' \
--header 'Content-Type: application/json' \
--data '{ "query":"https://nb.scene7.com/is/image/NB/bbw550bb_nb_02_i?$dw_detail_main_lg$&bgc=f1f1f1&layer=1&bgcolor=f1f1f1&blendMode=mult&scale=10&wid=1600&hei=1600", "links":[ "https://di2ponv0v5otw.cloudfront.net/posts/2022/05/04/6272f4a13751f5ea760832fb/s_wp_6272fa5c941f175a1ce82807.webp", "https://di2ponv0v5otw.cloudfront.net/posts/2022/11/10/636da131046d74db9f3b6e49/s_wp_636da1b1dff94d691895db7f.webp" ] }'
'''

Make sure to replace the `query` field with the URL of your query image and the `links` field with a list of URLs to your images to be ranked.

## Response

The API returns a JSON object containing a list of image URLs ranked in descending order of similarity to the query image.

