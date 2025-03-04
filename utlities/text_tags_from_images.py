
import io
import base64
import anthropic
import os
import json
from typing import List, Dict
from PIL import Image
import retry
import re
from pillow_avif import AvifImagePlugin
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.message import Message
from anthropic.types.messages.message_batch import MessageBatch
from datetime import datetime

from directories import IMAGES_FOLDER, IMAGE_TAG_SETS_FOLDER



CLIENT = anthropic.Anthropic()


def _resize_and_crop_image(
        img: Image,
        target_width:int=600,
        max_height:int=1400
) -> Image:

    # Calculate aspect ratio
    width, height = img.size
    aspect_ratio = height / width

    # Resize to target width while maintaining aspect ratio
    new_width = target_width
    new_height = int(new_width * aspect_ratio)
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Crop from the top if height exceeds max_height
    if new_height > max_height:
        # Crop from the top (keep the top portion)
        resized_img = resized_img.crop((0, 0, new_width, max_height))

    return resized_img


def _image_file_to_base64(image_filename: str) -> str:
    try:
        image_path = os.path.join(IMAGES_FOLDER, image_filename)
        image = _resize_and_crop_image(Image.open(image_path))

        img_buffer = io.BytesIO()

        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"error with rendering email {image_filename}: {e}")
        return base64.b64encode(b"").decode('utf-8')




def _create_tags_dictionary(message : Message, tags_to_ignore: List[str]) -> Dict:

    tags_dict = json.loads(message.content[0].text)
    return {key: tags_dict[key] for key in tags_dict.keys() if key not in tags_to_ignore}

def create_image_tags_single_image(image_file, data_extraction_prompt, tags_to_ignore = []):


    message = CLIENT.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0,
        system=data_extraction_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _image_file_to_base64(image_file),
                        },
                    },
                ]
            }
        ]
    )

    return _create_tags_dictionary(message)

def _name_for_anthropic_id(image_file_name):
    ID_CHARATER_LIMIT = 60
    name, ext = os.path.splitext(image_file_name)
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:ID_CHARATER_LIMIT]


def _create_requests_list(data_extraction_prompt: str, image_file_names_batch: List[str]) -> List[Request]:
    requests = [
        Request(
            custom_id=_name_for_anthropic_id(image_file_name),
            params=MessageCreateParamsNonStreaming(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system=data_extraction_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _image_file_to_base64(image_file_name),
                                },
                            },
                        ]
                    }
                ]
            )
        ) for image_file_name in image_file_names_batch
    ]

    return requests


def _wait_for_batch_to_finish(batch: MessageBatch) -> None:
    print(f"Waiting for batch {batch.id}")

    @retry.retry(exceptions=Exception, tries=10, delay=5, backoff=2, max_delay=120)
    def _check_batch_status():
        message_batch_status = CLIENT.messages.batches.retrieve(batch.id)
        print(f"Batch status: {message_batch_status}")

        if message_batch_status.processing_status != "ended":
            print(f"Batch {batch.id} still processing...")
            raise Exception("Batch not complete")

        return message_batch_status

    # Keep trying until batch is complete
    try:
        _check_batch_status()
        print(f"Batch {batch.id} finished. Grabbing data.")
    except Exception as e:
        print(f"Failed to complete batch after maximum retries: {e}")
        raise

def _extract_and_save_data_from_batch(batch: MessageBatch, tags_folder_file_path: str, tags_to_ignore) -> None:
    for result in CLIENT.messages.batches.results(
            batch.id,
    ):
        try:
            tags_dict = _create_tags_dictionary(result.result.message, tags_to_ignore)

            with open(os.path.join(tags_folder_file_path, result.custom_id + ".json"), 'w') as file:
                json.dump(tags_dict, file, indent=4)
        except Exception as e:
            print(e)

def _get_image_file_names(tags_folder_file_path: str) -> List[str]:

    files_already_made = os.listdir(tags_folder_file_path)


    image_file_names = os.listdir(IMAGES_FOLDER)
    image_file_names = [
        n for n in image_file_names if not n.startswith(".")

    ]

    image_file_names = [
        n for n in image_file_names if
        _name_for_anthropic_id(n) + ".json" not in files_already_made
    ]

    return image_file_names

def create_image_tags_full_dataset(
        index_name: str,
        data_extraction_prompt: str,
        batch_size: int=50,
        tags_to_ignore=[]
) -> None:
    tags_folder_file_path = os.path.join(IMAGE_TAG_SETS_FOLDER, index_name)

    if not os.path.isdir(tags_folder_file_path):
        os.mkdir(tags_folder_file_path)

    image_file_names = _get_image_file_names(tags_folder_file_path)
    print(f"{len(image_file_names)} files to create")

    message_batches = []
    for i in range(0, len(image_file_names), batch_size):
        print(f'batch starting with image {i}, {datetime.now()}')
        message_batch = CLIENT.messages.batches.create(
            requests=_create_requests_list(
                data_extraction_prompt=data_extraction_prompt,
                image_file_names_batch=image_file_names[i:i + batch_size]
            )
        )

        message_batches.append(message_batch)

    for batch in message_batches:
        _wait_for_batch_to_finish(batch)
        _extract_and_save_data_from_batch(
            batch=batch,
            tags_folder_file_path=tags_folder_file_path,
            tags_to_ignore=tags_to_ignore
        )