import os
import json
import base64

from functools import cached_property
from pinecone import Pinecone
from typing import List
import math
from transformers import CLIPProcessor, CLIPModel

from pinecone_index_utilities import _email_json_to_string, EMBEDDINGS_MODEL
from directories import IMAGE_TAG_SETS_FOLDER, IMAGES_FOLDER

# TODO - does this need to be in mixin?
def get_image_base64_from_path(image_path: str) -> str:
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


class KeyWordRAGSearchHandler(object):

    def __init__(self, index_name):
        self.pc = Pinecone(os.getenv('PINECONE_API_KEY'))
        self.index_name = index_name
        self.index = self.pc.Index(self.index_name)


    def _get_email_tags_str(self, email_name: str) -> str:
        with open(os.path.join(IMAGE_TAG_SETS_FOLDER, self.index_name, email_name + ".json"), "r") as f:
            email_metadata_json = json.load(f)

        return _email_json_to_string(email_metadata_json)

    @staticmethod
    def _get_email_image_path(email_name: str) -> str | None:
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

    @staticmethod
    def _get_embeddings_from_email_path(email_path: str, index_name: str) -> str:

        tags_dict_path = os.path.join(
            IMAGE_TAG_SETS_FOLDER,
            index_name,
            email_path.split("/")[-1].split(".")[0] + ".json"
        )

        with open(tags_dict_path, "r") as f:
            email_metadata_json = json.load(f)

        return _email_json_to_string(email_metadata_json)

    def query_index(self, email_query):


        query_embedding = self.pc.inference.embed(
            model=EMBEDDINGS_MODEL,
            inputs=[email_query],
            parameters={
                "input_type": "query"
            }
        )

        results = self.index.query(
            namespace=self.index_name,
            vector=query_embedding[0].values,
            top_k=5,
            include_values=False,
            include_metadata=False
        )
        # TODO - get less
        most_similar_emails_paths = []
        for m in results['matches']:
            most_similar_emails_paths.append(self._get_email_image_path(m['id']))

        return [
            get_image_base64_from_path(path) for path in most_similar_emails_paths
    ]

class EmailDisplayHandler(object):

    @staticmethod
    def _single_email_display_component(email_image, width_pct):

        return f"""   
            <div style="width: {width_pct}%;">  
                <div style="height: 600px; overflow: hidden;">
                    <img src="data:image/jpeg;base64,{email_image}" style="width: 100%;">
                </div>      
            </div>"""

        # TODO - add back in get embeddings??
        # < div > """ + \
        # _get_embeddings_from_email_path(path, index_name).replace("\n", "<br>") +\
        # """ < / div >

    def _emails_row_component(self, email_images):
        width_pct = math.floor(100 / len(email_images))
        return f"""
                <div class="row" style="display: flex; flex-wrap: wrap; justify-content: space-between; width: 100%; align-items: flex-start;"> 
                {''.join(self._single_email_display_component(email_img, width_pct) for email_img in email_images)}
                </div>"""

    def _emails_row_outer_fix(self, email_images, title):
        return f"""<div style="display: flex; flex-direction: column; align-items: center; width: 95%;">    
            <div style="display: flex; flex-direction: column; gap: 20px;">  
                <h2>{title}</h2> 
                {self._emails_row_component(email_images)}
            </div>"""


    def display_emails_html_from_query(
        self,
        similar_emails_from_keywords_rag: List[str],
        similar_emails_from_image_embeddings,
    ) -> str:

        html_content = f"""  <div style="display: flex; flex-direction: column; gap: 20px;">   
            {self._emails_row_outer_fix(similar_emails_from_keywords_rag, "Emails found, Keyword RAG Search")}
            {self._emails_row_outer_fix(similar_emails_from_image_embeddings, "Emails found,  Image embeddings Search")} 
        </div> 
        """

        return html_content


class ImageEmbeddingsSearchHandler(object):

    INDEX_NAME = "clip-email-index"
    CLIP_MODEL =  "openai/clip-vit-base-patch32"

    def __init__(self):
        self.pc = Pinecone(os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index(self.INDEX_NAME)

    @cached_property
    def model(self):
        model = CLIPModel.from_pretrained(self.CLIP_MODEL)
        model.to("cpu")
        return model

    @cached_property
    def processor(self):
        return CLIPProcessor.from_pretrained(self.CLIP_MODEL)

    def get_emails_from_image_embeddings(self, query_text):

        # create_text_embeddings_function
        text_embedding = self.processor(
            text=query_text,
            padding=True,
            images=None,
            return_tensors='pt'
        ).to("cpu")

        text_emb = self.model.get_text_features(**text_embedding)[0].tolist()

        results = self.index.query(
            namespace="ns1",
            vector=text_emb,
            top_k=5,
            include_values=False,
            include_metadata=True
        )

        paths = [os.path.join(IMAGES_FOLDER, m['id']) for m in results['matches']]
        return [
            get_image_base64_from_path(path) for path in paths
        ]


def get_emails_from_query(
        email_query: str,
        index_name: str
) -> str:

    similar_emails_from_keywords_rag = KeyWordRAGSearchHandler(index_name).query_index(email_query)
    similar_emails_from_image_embeddings = ImageEmbeddingsSearchHandler().get_emails_from_image_embeddings(email_query)

    return EmailDisplayHandler().display_emails_html_from_query(
        similar_emails_from_keywords_rag=similar_emails_from_keywords_rag,
        similar_emails_from_image_embeddings=similar_emails_from_image_embeddings,
    )


