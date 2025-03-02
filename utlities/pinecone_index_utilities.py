
from pinecone import Pinecone, ServerlessSpec
import time
import os
import json
from langchain_pinecone import PineconeEmbeddings
from directories import IMAGE_TAG_SETS_FOLDER

from dotenv import load_dotenv

load_dotenv()



EMBEDDINGS_MODEL = 'multilingual-e5-large'

def _join_list_or_return_string(s):
    if type(s) == str:
        return ": " + s
    if type(s) == list:
        return ":\n - " + "\n - ".join(s)
    print("unable to reformat ", s)
    return ""

def _email_json_to_string(email_metadata_json):
    email_tags_str = ""

    for k, v in email_metadata_json.items():
        email_tags_str = email_tags_str + "\n\n" + k + _join_list_or_return_string(v)

    return (email_tags_str)


def create_index(index_name):

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    embeddings = PineconeEmbeddings(
        model=EMBEDDINGS_MODEL,
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )

    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    spec = ServerlessSpec(cloud=cloud, region=region)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimension,
            metric="cosine",
            spec=spec
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # See that it is empty
    print("Index before upsert:")
    print(pc.Index(index_name).describe_index_stats())
    print("\n")

def _get_data_to_embed(index_name):

    email_tags_dir = os.path.join(IMAGE_TAG_SETS_FOLDER, index_name)

    data_to_embed_list = []
    for email_tags_file in os.listdir(email_tags_dir):
        if email_tags_file.startswith("."):
            continue

        with open(os.path.join(email_tags_dir, email_tags_file), "r") as f:
            email_metadata_json = json.load(f)

        email_tags_str = _email_json_to_string(email_metadata_json)

        data_to_embed_list.append(
            {
                "id": email_tags_file.replace(".json", ""),
                "text": email_tags_str

            }
        )

    return data_to_embed_list

def get_embeddings_and_upsert(index_name):

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    data_to_embed_list = _get_data_to_embed(index_name)

    chunk_size = 50
    chunks = [data_to_embed_list[i:i + chunk_size] for i in range(0, len(data_to_embed_list), chunk_size)]

    records = []

    for chunk_to_embed in chunks:
        # Convert the text into numerical vectors that Pinecone can index
        embeddings_chunk = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d["text"] for d in chunk_to_embed],
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )

        for d, e in zip(chunk_to_embed, embeddings_chunk):
            records.append({
                "id": d["id"],
                "values": e["values"],
            })

    # TODO - do I need to check if index exists?
    index = pc.Index(index_name)
    # Upsert the records into the index

    index.upsert(
        vectors=records,
        namespace=index_name
    )

    print("Index after upsert:")
    print(pc.Index(index_name).describe_index_stats())
    print("\n")



