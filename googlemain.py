from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from typing import List
import torch
from PIL import Image
from io import BytesIO
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import aiohttp
import ssl
import base64
import json
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
            print(response)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
            embedding = extract_embedding(image_bytes)
            df = pd.concat([df, pd.DataFrame({'image_url': [link], 'embedding': [embedding.detach()]})])
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


async def rank_images(query: str, links: List[str]):
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
        df = callConcurrent(links)
        target_embedding = gen_target_embedding(query)

        image_urls = get_matches(target_embedding, df)

        return {"ranked_images": image_urls}
    except ValueError as e:
        return {"status_code":400, "detail":str(e)}

#@functions_framework.http
async def main():
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    # request_json = request.get_json(silent=True)
    # request_args = request.args
    # print(request_json)
    # print(request_args)

    query = 'https://nb.scene7.com/is/image/NB/bbw550bb_nb_02_i?$dw_detail_main_lg$&bgc=f1f1f1&layer=1&bgcolor=f1f1f1&blendMode=mult&scale=10&wid=1600&hei=1600'
    links = [
 "https://di2ponv0v5otw.cloudfront.net/posts/2022/05/04/6272f4a13751f5ea760832fb/s_wp_6272fa5c941f175a1ce82807.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/11/10/636da131046d74db9f3b6e49/s_wp_636da1b1dff94d691895db7f.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/05/17/6283ae38c693bd6a58c90b7e/s_wp_6283ae9d9c33781543188976.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2023/03/03/6402488bdbb0e72c27751303/s_wp_640248a492e491cc620af5e1.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/01/26/63d344aec9a22876ee878984/s_wp_63d344bdffb5d0480778770b.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/02/05/63dfddbe3676a105f3c4ea99/s_wp_63dfddbe3676a105f3c4ea9a.webp",
"https://di2ponv0v5otw.cloudfront.net/posts/2022/12/27/63ab38d232c1dc6122705815/s_63bf28b9a0aeb7974b9889c1.jpg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/02/02/63dc34ee8bb2e28b9c2bd469/s_63f448b9a0aeb7ffc98a442f.jpg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/22/644467589464f368279f7cb4/s_64446763f644e523964f80b7.jpg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/22/644466ac382db80ee4f6cfc3/s_644466b98d7a3cdff628d6d0.jpg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/13/6438395649d8d87fcacdc1a8/s_wp_6438395649d8d87fcacdc1a9.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/17/643d68341741be8489176d0a/s_wp_643d69ffa0e6c69cc7c5bf0c.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/18/643f4f4881078affa0b7caf6/s_wp_643f4f77b635f8c7ddf5238e.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2022/09/15/632383c42ca3080c96f83f26/s_wp_632383c54bd760ba964ebdbc.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/25/641f9bd9948fa025f014a253/s_wp_641f9bd9948fa025f014a254.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/01/28/63d5aa01452746969cc9edb7/s_wp_63d5aad6a0e6c6c21c61e92b.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/02/17/63f02d8717fb4b59b7e8628a/s_wp_63f02d8a17fb4b59b7e862b5.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/06/642edae54bd7602040e844ba/s_wp_642edbf9ffb5d07f4eb70ace.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2022/10/16/634c9aba8d7a3c24642b9753/s_wp_634c9ac6a0e6c649fb524183.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2022/05/04/627304ea45d73159d7eb8009/s_wp_627306ea3e732b9b3fb65b71.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/28/6423086ebc6e1c9acef06a80/s_wp_64230884382db83aff2f6937.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/11/643591aff8c5dace9921ede8/s_wp_643591b0f8c5dace9921edef.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/21/6442fd5cb142f314eb0b0eca/s_6442fd5e91e053a6d444d2d5.jpeg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/13/6438d36813d681fcdba4d08a/s_wp_6438d36b3b982a6100e58bf4.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/09/640a4d9fdbb0e7825df978cb/s_wp_640a5a4417e49c863b519ade.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/20/644178ab04166ddbece84723/s_wp_644178ab04166ddbece84724.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/21/644312ec3251873eadc0a19d/s_644312efc9a228a309e1f8c8.jpeg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/10/6434386050e2df7a235ed1f3/s_wp_643438641645f74facdf84e7.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/20/644218225c465ee2fa916b19/s_wp_644218fbff04841a9f6f56a5.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/02/10/63e6926e1645f73cb941b513/s_wp_63e6926e1645f73cb941b514.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/02/10/63e693c2dff94df22fab2fec/s_wp_63e6940117fb4bc2676f35be.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/24/641e1d80cfc66e45ee074705/s_wp_641e1d8483cbec93a8648ac0.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/29/64248406c5df6c4b482c237b/s_wp_64248406c5df6c4b482c237c.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/02/04/63df3abd8bb2e27b530773f6/s_wp_63df3b3856b2f8382c3a6ef6.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/03/64024cda7fc3ac9c17ebf268/s_wp_64024cda7fc3ac9c17ebf269.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/22/64444ae0e0f2ceae71aabb76/s_64444ae397b5d073e602c2bc.jpeg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/09/640a592a4bf9ff621a0c1837/s_wp_640a595effb5d0397e5d5357.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2022/10/17/634d90b05d686b9df2727886/s_wp_634d90c7382db83bd0f9580d.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/10/640b65a52fc59f0a79873ca8/s_wp_640b65a817e49cd1b8933928.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/17/643e03e717e49c236b84c19c/s_643e056802760bdfe1f32b78.jpg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/21/644332dc5c465ede40806b53/s_644332df3b982a854a54f05f.jpeg",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/03/23/641cd2eeade7b566997ab46b/s_wp_641cd2f1a0e6c6468cda15da.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2023/04/10/64342b1cacf46238695e59f8/s_wp_64342b1cacf46238695e59f9.webp",
 "https://di2ponv0v5otw.cloudfront.net/posts/2022/07/20/62d8410e7ec30c375e7d65c7/s_wp_62d8410f7ec30c375e7d65df.webp",
 "https://images1.vinted.net/t/03_0019f_gYzhikz3ocpcG915ymSgXqpS/f800/1677345143.jpeg?s=32d3bd61769448d9ebfab2547ed4d6ba2d190271",
 "https://images1.vinted.net/t/03_020d3_CMga1yefxzAt4PoSmSBqFpRq/f800/1677342605.jpeg?s=a1b68a2e880a1227912b57a6906ad5eb87843ebe",
 "https://images1.vinted.net/t/03_015a5_dZPoa2hoJPdC1Zmo1VCphoPS/f800/1676992414.jpeg?s=e9f73da56754d26f4f02a264b83616410cbf243a",
 "https://images1.vinted.net/t/02_025fb_R6ymgy6t63yjSka4kpWcSxjo/f800/1681766910.jpeg?s=78bcee0c87b2db0ef417e5a89c941f66aa625c6d",
 "https://images1.vinted.net/t/03_019cc_8jAzWNT6AxiiEt5MeUATvQNP/f800/1659170450.jpeg?s=76e4f371d18921a3a16e0e8cb51878f6eba81874",
 "https://images1.vinted.net/t/02_0031d_HgrnFCgKYJb4kkpbEkS5HhS3/f800/1679828198.jpeg?s=b55ced2efcadef24f46233b4e3bd8f266d0d4d51",
 "https://images1.vinted.net/t/01_01b41_5JWz7uPKrwCN1TBHvMi3NyCd/f800/1671756400.jpeg?s=81d37751e9d119d510bc0eed198a248434422a6d",
 "https://images1.vinted.net/t/02_00540_apAU7p5d6uYaimj2N5DTvsTp/f800/1673187734.jpeg?s=1056a1bda4b1923044b9d19294e490ffd553a113",
 "https://images1.vinted.net/t/02_00e56_ByBf5xVMeMm7r5zEXSf8AAL6/f800/1660410429.jpeg?s=177ccf1ddec6bddac01fb1a2d46ec50d0c316c56",
 "https://images1.vinted.net/t/03_007d3_M7zAYx6NLBqNJBYGzxr3BWZt/f800/1675972224.jpeg?s=565ef32d842daebec67b73b0300b186d1373f7e2",
 "https://images1.vinted.net/t/02_0222f_8yJfX6LLrVtW1oCpcuSX97hR/f800/1647362894.jpeg?s=d1dbc5bfc320037909223e509d16c4c190cfe532",
 "https://images1.vinted.net/t/02_00ab1_BStcnjat449pyU54S9QFhb56/f800/1663261528.jpeg?s=d363a8d82611a70c70787869977afe03d7a9a3e9",
 "https://images1.vinted.net/t/01_026eb_Vt2dVBgzkxwrYy5MCP95cmZJ/f800/1677284497.jpeg?s=3136157fbfa3e27d0b04ab565b526dd591d34e84",
 "https://images1.vinted.net/t/03_023a2_CDsDB1GJYgd9XND5kdFcHvYf/f800/1659097185.jpeg?s=0ce2950de8c22e73f73434a33c7ed48fb1bc1a30",
 "https://images1.vinted.net/t/03_0161a_YjswEFhH4bTPsVuhUL6xW7Eq/f800/1661372635.jpeg?s=4f13ec18ed93692ddc2bb008c60ce67a0bed7166",
 "https://images1.vinted.net/t/01_00acf_HAZyVMRRUiwmSc7S8UjAoji1/f800/1666048916.jpeg?s=3b89a1447e170b17a0d847e4548a9d4fd1548317",
 "https://images1.vinted.net/t/02_0122f_RgoYZFWi2k5yfsMegV34Hgrb/f800/1658864775.jpeg?s=6728e41e188620efed0e371d6b524ff5507aa260",
 "https://images1.vinted.net/t/03_024fa_PQa6qhMyq57dXngNvTGMEDDp/f800/1675134647.jpeg?s=9e0874f6d3485ea5ae95b00a17214bfc40c2fe6f",
 "https://images1.vinted.net/t/03_02087_ZEFJsTBrHJtJJZR3m44zP41j/f800/1634229319.jpeg?s=1f0509f4cee214dbf53820f1380a5ab0d14581cb",
 "https://images1.vinted.net/t/02_003ec_S76Sq1eSftpkGkJoyBRUgQ2D/f800/1654480865.jpeg?s=4463ddc678f4a97beb8a637f8f4aa2cfaf68ba78",
 "https://images1.vinted.net/t/02_0244c_t4c1VJcYh5S2fAT2PTjkKtdH/f800/1653074070.jpeg?s=1cc07ed9ef067607911fe453113b025a30f61ea9",
 "https://images1.vinted.net/t/01_00c0a_LWc6yCLSe8bqT4Vuu56VFYHA/f800/1639061577.jpeg?s=86dd323aa8840c1bab88aabef019b215bb0176a5",
 "https://images1.vinted.net/t/03_0121f_FkXW6Bv9o3m6fGRGGh5h4o6G/f800/1640763322.jpeg?s=7640a773ef424b9c41849f7d36d37e9210bcda4a",
 "https://cf-assets-thredup.thredup.com/assets/459482691/complimentary.jpg",
 "https://cf-assets-thredup.thredup.com/assets/479971944/complimentary.jpg",
 "https://cf-assets-thredup.thredup.com/assets/463632789/complimentary.jpg",
"https://cf-assets-thredup.thredup.com/assets/478344411/complimentary.jpg",
    ]
    # query = request_json['query']
    # links = request_json['links']
    # print('QUERY = ',query)
    # print('LINKS = ',links)
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)

    try:
        #res = loop.run_until_complete(rank_images(query, links))
        res = await rank_images(query, links)
        print('RES = ',res)
        return {"res": res}
    except Exception as e:
        print('EXCEPTION = ',e)
        return {"res": e.args}
    finally:
        #loop.close()
        print('FINALLY')


if __name__ == '__main__':
    #asyncio.run(main())
    main()


async def callConcurrent(links)-> pd.DataFrame:
    # Create an SSL context to bypass SSL verification
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    res=[]

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = [call_cloud_function(session, image_url) for image_url in links]
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
            embedding = extract_embedding(image_bytes_io)
            res.append({'image_url':json_result['image_url'],'embedding':embedding})
        return pd.DataFrame(res)


async def call_cloud_function(session, image_url):
    headers = {"Content-Type": "application/json","Authorization":"Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImM5YWZkYTM2ODJlYmYwOWViMzA1NWMxYzRiZDM5Yjc1MWZiZjgxOTUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTA4OTAyNjUyMjg4MTEyNjM1MjY1IiwiZW1haWwiOiJ0cml2ZWRpaGFyc2g0OUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXRfaGFzaCI6IkxyNnYxSVpJMDRSNVZPc013YlV4NGciLCJpYXQiOjE2ODMwMDI4NDcsImV4cCI6MTY4MzAwNjQ0N30.m-occDf5T3bgqN_5RLSNU3upmp3Qi_7bvlqyyjIYoIRY5mRL1fsuA1fALb6bf1HwtTir-TE7IGEgDryAKxu4iX4HEsS4sHDSBrSu05WA_zi76dBIJlO_KlCYmrFZwU1hdPCj7PpVAkhI_zQpwra3dDMMOkFl1sVQas5unvniVHXxsIaJFBjwQ8SJRFsI1N2_8zTg0bfi3FrFG3DQuaJGmdgPh3H_d85t7gEIgl9Ce0AOEa9mePZthhk7kZCoXQwZdcHnkPassFFDKpEzICXzYpl8a-5U6XYfGO4eiH07qOoA9TGn7qRHWDHDU4VzBlOHCFFrplxVyj_5IuqEUt1Y2w"}
    async with session.post('https://stream-images-62xutbvi7a-uc.a.run.app/', json={"image_url": image_url},headers=headers) as response:
        # read the actual bytes from the response
        return await response.text()
