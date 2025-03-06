import os
import json
import base64

from functools import cached_property
from pinecone import Pinecone
from typing import List, Dict
import math
from transformers import CLIPProcessor, CLIPModel

from pinecone_index_utilities import _email_json_to_string, EMBEDDINGS_MODEL
from directories import IMAGE_TAG_SETS_FOLDER, IMAGES_FOLDER

def get_image_base64_from_path(image_path: str) -> str:
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


class KeyWordRAGSearchHandler:

    def __init__(self, index_name: str):

        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        self.pc = Pinecone(api_key)
        self.index_name = index_name
        self.index = self.pc.Index(self.index_name)
        self.image_tag_sets_folder = os.path.join(IMAGE_TAG_SETS_FOLDER, index_name)

    @staticmethod
    def _get_email_image_path(email_name: str) -> str:
        """
        Get path corresponding to email name. Email name is distinct, but there are a range of file formats.
        """

        possible_image_extensions = [
            '.png', '.jpg', '.jpeg', '.gif', '.bmp','.avif',
            '.tiff', '.tif', '.webp', '.svg', '.ico',
            '.heic', '.heif', '.raw', '.cr2', '.nef',
            '.arw', '.dng', '.psd', '.ai', '.eps'
        ]

        for ext in possible_image_extensions:
            potential_path = os.path.join(IMAGES_FOLDER, email_name + ext)
            if os.path.exists(potential_path):
                return potential_path

        raise Exception(f"No local file found for {email_name}")

    def _get_tags_for_email(self, email_path: str) -> str:

        tags_dict_path = os.path.join(
            self.image_tag_sets_folder,
            os.path.splitext(os.path.basename(email_path))[0] + ".json"
        )

        try:
            with open(tags_dict_path, "r") as f:
                return _email_json_to_string(json.load(f))
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found for email: {email_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in metadata file for email: {email_path}")


    def query(self, email_query: str, k: int=5) -> List[Dict]:

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
            top_k=k,
            include_values=False,
            include_metadata=False
        )

        most_similar_emails = []
        for match in results['matches']:

            # Here, we use the filename of the image as the ID in pinecone. That can then be used to grab both
            # the image itself, and the tags, both of which are stored locally.
            image_path = self._get_email_image_path(match['id'])
            most_similar_emails.append({
                "image": get_image_base64_from_path(image_path),
                "tags": self._get_tags_for_email(image_path)
            })

        return most_similar_emails

class ImageEmbeddingsSearchHandler(object):

    # DEFAULT_INDEX_NAME = "clip-email-index"
    CLIP_MODEL =  "openai/clip-vit-base-patch32"
    CLIP_INDEX_NAMESPACE = "ns1"

    def __init__(self, index_name):
        self.pc = Pinecone(os.getenv('PINECONE_API_KEY'))
        self.index_name = index_name
        self.index = self.pc.Index(self.index_name)

    @cached_property
    def _model(self):
        model = CLIPModel.from_pretrained(self.CLIP_MODEL)
        model.to("cpu")
        return model

    @cached_property
    def _processor(self):
        return CLIPProcessor.from_pretrained(self.CLIP_MODEL)

    def query(self, query_text: str) -> List[Dict]:

        text_embedding = self._processor(
            text=query_text,
            padding=True,
            images=None,
            return_tensors='pt'
        ).to("cpu")

        text_emb = self._model.get_text_features(**text_embedding)[0].tolist()

        results = self.index.query(
            namespace=self.CLIP_INDEX_NAMESPACE,
            vector=text_emb,
            top_k=5,
            include_values=False,
            include_metadata=True
        )

        image_file_paths = [os.path.join(IMAGES_FOLDER, m['id']) for m in results['matches']]
        return [
            {"image": get_image_base64_from_path(image_file_path)} for image_file_path in image_file_paths
        ]

class EmailHTMLDisplayHTMLRenderer:

    @staticmethod
    def _tags_display_component(email: Dict) -> str:
        if email.get("tags"):
            tags =  email["tags"].replace("\n", "<br>").replace("tags:","Tags used in index:")
            return f"""<div style="font-size:12px;"> {tags} </div> """
        else:
            return ""

    @staticmethod
    def _email_display_component(email: Dict) -> str:
        return f"""
                <div style="height: 600px; overflow: hidden;">
                    <img src="data:image/jpeg;base64,{email["image"]}" style="width: 100%;" />
                </div>"""


    def _single_email_display_wrapper(self, email: Dict, width_pct: int) -> str:

        return f"""   
            <div style="width: {width_pct}%;">   
                {self._email_display_component(email)}
                {self._tags_display_component(email)} 
            </div>"""


    def _emails_row_component(self, email_images: List[Dict]) -> str:
        width_pct = math.floor(100 / len(email_images))
        return f"""
                <div class="row" style="display: flex; flex-wrap: wrap; justify-content: space-between; width: 100%; align-items: flex-start;"> 
                {''.join(self._single_email_display_wrapper(email_img, width_pct) for email_img in email_images)}
                </div>"""

    def _emails_row_outer_div(self, email_images, title):
        return f"""<div style="display: flex; flex-direction: column; align-items: center; width: 95%;">    
            <div style="display: flex; flex-direction: column; gap: 20px;">  
                <h2>{title}</h2> 
                {self._emails_row_component(email_images)}
            </div>"""


    def display_emails_html_from_query(
        self,
        emails_from_keywords_rag: List[Dict[str,str]],
        emails_from_image_embeddings:  List[Dict[str,str]],
    ) -> str:

        html_content = f"""  <div style="display: flex; flex-direction: column; gap: 20px;">   
            {self._emails_row_outer_div(emails_from_keywords_rag, "Emails found, Keyword RAG Search")}
            {self._emails_row_outer_div(emails_from_image_embeddings, "Emails found,  Image embeddings Search")} 
        </div> 
        """

        return html_content



def display_emails_from_query(
    email_query: str,
    keyword_rag_index_name: str,
    clip_index_name: str
) -> str:


    emails_from_keywords_rag = KeyWordRAGSearchHandler(keyword_rag_index_name).query(email_query)
    emails_from_image_embeddings = ImageEmbeddingsSearchHandler(clip_index_name).query(email_query)

    return EmailHTMLDisplayHTMLRenderer().display_emails_html_from_query(
        emails_from_keywords_rag=emails_from_keywords_rag,
        emails_from_image_embeddings=emails_from_image_embeddings,
    )


