import os
import json
import base64
from pinecone import Pinecone
import random

from pinecone_index_utilities import _email_json_to_string
from directories import IMAGE_TAG_SETS_FOLDER, IMAGES_FOLDER


def _get_email_tags_str(email_name, index_name):
    with open(os.path.join(IMAGE_TAG_SETS_FOLDER, index_name, email_name + ".json"), "r") as f:
        email_metadata_json = json.load(f)

    return _email_json_to_string(email_metadata_json)


def get_email_image_path(email_name):
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


def search_email(email_name, index_name, n_emails=4):
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

def get_random_emails():
    email_images = os.listdir(IMAGES_FOLDER)
    email_images = [os.path.join(IMAGES_FOLDER, d) for d in email_images if d.endswith(".png")]
    return random.sample(email_images, 4)

def get_image_base64_from_path(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def _get_embeddings_from_email_path(email_path, index):
    tags_dict_path = os.path.join(
        IMAGE_TAG_SETS_FOLDER,
        index,
        email_path.split("/")[-1].split(".")[0] + ".json"
    )

    with open(tags_dict_path, "r") as f:
        email_metadata_json = json.load(f)

    return _email_json_to_string(email_metadata_json)

def show_emails_html(my_email_path, most_similar_emails_paths, index_name, other_emails_title="Most similar emails"):


    html_content = f"""
    <div class="row" style="display: flex; justify-content: space-between; align-items: flex-start;">   
        <div style="display: flex; flex-direction: column; align-items: center; width: 25%;">
            <h2>Reference email</h2>
            <div>
            <div style="width: 90%; height: 600px; overflow: hidden;">
                <img src="data:image/jpeg;base64,{get_image_base64_from_path(my_email_path)}" style="width: 100%;">
            </div>   
        </div>   
            {_get_embeddings_from_email_path(my_email_path, index_name).replace("\n","<br>")}
            </div>
        <div style="display: flex; flex-direction: column; align-items: center; width: 75%;">      
            <h2>{other_emails_title}</h2>
            <div class="row" style="display: flex; flex-wrap: wrap; justify-content: space-between; width: 100%; align-items: flex-start;"> 
            {''.join(f"""   
            <div style="width: 30%;">  
                    <div style="height: 600px; overflow: hidden;">
                        <img src="data:image/jpeg;base64,{get_image_base64_from_path(path)}" style="width: 100%;">
                                  
                    </div>    
                    <div>
                    {_get_embeddings_from_email_path(path, index_name).replace("\n","<br>")}
                    </div>    
                </div> 
            """ for path in most_similar_emails_paths)}
            </div>
             
        </div>
    </div> 
    """

    return html_content


def get_and_show_most_similar_emails_html(my_email_name, index):
    email_path, most_similar_emails_paths = search_email(my_email_name, index)

    return show_emails_html(email_path, most_similar_emails_paths, index)


