import os
import json
import base64
from pinecone import Pinecone
from typing import List, Tuple
import math

from pinecone_index_utilities import _email_json_to_string, EMBEDDINGS_MODEL
from directories import IMAGE_TAG_SETS_FOLDER, IMAGES_FOLDER


def _get_email_tags_str(email_name: str, index_name: str) -> str:
    with open(os.path.join(IMAGE_TAG_SETS_FOLDER, index_name, email_name + ".json"), "r") as f:
        email_metadata_json = json.load(f)

    return _email_json_to_string(email_metadata_json)


def get_email_image_path(email_name: str) -> str | None:
    # Define possible image extensions to check
    possible_extensions = [
        '.png', '.jpg', '.jpeg', '.gif', '.bmp','.avif',
        '.tiff', '.tif', '.webp', '.svg', '.ico',
        '.heic', '.heif', '.raw', '.cr2', '.nef',
        '.arw', '.dng', '.psd', '.ai', '.eps'
    ]

    # Check which extension exists for this email
    for ext in possible_extensions:
        potential_path = os.path.join(IMAGES_FOLDER, email_name + ext)
        if os.path.exists(potential_path):
            return potential_path

    # If no matching file is found, return None or raise an exception
    # Option 1: Return None
    return None


def search_email(email_name: str, index_name: str, n_emails: int=4) -> Tuple[str, List[str]]:
    pc = Pinecone(os.getenv('PINECONE_API_KEY'))
    index = pc.Index(index_name)


    email_tags_str = _get_email_tags_str(email_name, index_name)

    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[email_tags_str],
        parameters={
            "input_type": "query"
        }
    )

    results = index.query(
        namespace=index_name,
        vector=query_embedding[0].values,
        top_k=n_emails,
        include_values=False,
        include_metadata=False
    )
    # TODO - get less
    most_similar_emails_paths = []
    for m in results['matches']:
        if m['id'] != email_name:
            most_similar_emails_paths.append(get_email_image_path(m['id']))

    return get_email_image_path(email_name), most_similar_emails_paths

def get_image_base64_from_path(image_path: str) -> str:
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def _get_embeddings_from_email_path(email_path: str, index_name: str) -> str:

    tags_dict_path = os.path.join(
        IMAGE_TAG_SETS_FOLDER,
        index_name,
        email_path.split("/")[-1].split(".")[0] + ".json"
    )

    with open(tags_dict_path, "r") as f:
        email_metadata_json = json.load(f)

    return _email_json_to_string(email_metadata_json)


def _single_email_display_component(email_image, width_pct):

    return f"""   
        <div style="width: {width_pct}%;">  
                <div style="height: 600px; overflow: hidden;">
                    <img src="data:image/jpeg;base64,{email_image}" style="width: 100%;">

                </div>      
            </div> """

    # TODO - add back in get embeddings??
    # < div > """ + \
    # _get_embeddings_from_email_path(path, index_name).replace("\n", "<br>") +\
    # """ < / div >

def _emails_row_component(email_images):
    width_pct = math.floor(100 / len(email_images))
    return f"""
            <div class="row" style="display: flex; flex-wrap: wrap; justify-content: space-between; width: 100%; align-items: flex-start;"> 
            {''.join(_single_email_display_component(email_img, width_pct) for email_img in email_images)}
            </div>"""

def _emails_row_outer_fix(email_images, title):
    return f"""<div style="display: flex; flex-direction: column; align-items: center; width: 95%;">    
        <div style="display: flex; flex-direction: column; gap: 20px;">  
            <h2>{title}</h2> 
            {_emails_row_component(email_images)}
        </div>"""

##    "<div class="row" style="display: flex; justify-content: space-between; align-items: flex-start;">
def show_emails_html_from_query(
        similar_emails_from_keywords_rag: List[str],
        similar_emails_from_image_embeddings,
) -> str:

    html_content = f"""  <div style="display: flex; flex-direction: column; gap: 20px;">   
        {_emails_row_outer_fix(similar_emails_from_keywords_rag, "Emails found, Keyword RAG Search")}
        {_emails_row_outer_fix(similar_emails_from_image_embeddings, "Emails found,  Image embeddings Search")} 
    </div> 
    """

    return html_content



def _get_emails_from_image_embeddings(query_text):
    from transformers import CLIPProcessor, CLIPModel

    # instantiate clip model...
    pc = Pinecone(os.getenv('PINECONE_API_KEY'))
    index = pc.Index("clip-email-index")

    model_id = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.to("cpu")

    # create_text_embeddings_function
    text_embedding = processor(text=query_text,
                               padding=True,
                               images=None,
                               return_tensors='pt').to("cpu")

    text_emb = model.get_text_features(**text_embedding)[0].tolist()



    results = index.query(
        namespace="ns1",
        vector=text_emb,
        top_k=5,
        include_values=False,
        include_metadata=True
    )

    return [
        os.path.join(IMAGES_FOLDER, m['id']) for m in results['matches']
    ]


def get_emails_from_query(
        email_query: str,
        index_name: str
) -> str:
    pc = Pinecone(os.getenv('PINECONE_API_KEY'))
    index = pc.Index(index_name)


    query_embedding = pc.inference.embed(
        model=EMBEDDINGS_MODEL,
        inputs=[email_query],
        parameters={
            "input_type": "query"
        }
    )

    results = index.query(
        namespace=index_name,
        vector=query_embedding[0].values,
        top_k=5,
        include_values=False,
        include_metadata=False
    )
    # TODO - get less
    most_similar_emails_paths = []
    for m in results['matches']:
        most_similar_emails_paths.append(get_email_image_path(m['id']))

    similar_emails_from_keywords_rag = [
        get_image_base64_from_path(path) for path in most_similar_emails_paths
    ]

    similar_emails_from_image_embeddings_paths = _get_emails_from_image_embeddings(email_query)


    similar_emails_from_image_embeddings = [
        get_image_base64_from_path(path) for path in similar_emails_from_image_embeddings_paths
    ]

    return show_emails_html_from_query(
        similar_emails_from_keywords_rag=similar_emails_from_keywords_rag,
        similar_emails_from_image_embeddings=similar_emails_from_image_embeddings,
    )


