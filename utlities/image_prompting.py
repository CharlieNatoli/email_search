
import io
import base64
import anthropic
import os
import json
from PIL import Image
import time
import re
from pillow_avif import AvifImagePlugin
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from datetime import datetime

from directories import IMAGES_FOLDER, IMAGE_TAG_SETS_FOLDER


def _resize_and_crop_image(img, target_width=600, max_height=1400):
    # Open the image

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

    # Save the processed image
    return (resized_img)


def _image_file_to_base64(image_filename):
    image_path = os.path.join(IMAGES_FOLDER, image_filename)
    image = _resize_and_crop_image(Image.open(image_path))

    img_buffer = io.BytesIO()

    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


CLIENT = anthropic.Anthropic()



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
    tags_dict = json.loads(message.content[0].text)
    tags_dict = {key: tags_dict[key] for key in tags_dict.keys() if key not in tags_to_ignore}
    return tags_dict


def _name_for_anthropic_id(image_file_name):

    name, ext = os.path.splitext(image_file_name)
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:60]

# TODO - better way to structure this so its not waiting on all batches to be generated before sarting to extract?
# TODO - figure out to hook up anthropic-ized names with query and display stuff
#  TODO - does tags_to_ignore need to be here, or just in query creation part?
def create_image_tags_full_dataset(index_name, data_extraction_prompt, tags_to_ignore = [], batch_size=50):


    tags_folder_file_name = os.path.join(IMAGE_TAG_SETS_FOLDER, index_name)

    if not os.path.isdir(tags_folder_file_name):
        os.mkdir(tags_folder_file_name)

    files_already_made = os.listdir(tags_folder_file_name)


    image_file_names = os.listdir(IMAGES_FOLDER)
    image_file_names = [
        n for n in image_file_names if not n.startswith(".")

    ]

    image_file_names = [
        n for n in image_file_names if
        _name_for_anthropic_id(n) + ".json" not in files_already_made
    ]

    print(f"{len(image_file_names)} files to create")

    message_batches = []
    for i in range(0, len(image_file_names), batch_size):
        image_file_names_batch = image_file_names[i:i + batch_size]
        print(f'batch starting with image {i}, {datetime.now()}')


        requests = [
            Request(
                  custom_id= _name_for_anthropic_id(image_file_name),
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
            )  for image_file_name in image_file_names_batch
        ]

        message_batch = CLIENT.messages.batches.create(
            requests=requests
        )

        message_batches.append(message_batch)

    for batch in message_batches:
        print(f"checking batch {batch.id}")
        batch_finished = False
        while not batch_finished:
            print("  not finished. Waiting 10 seconds")
            time.sleep(10)
            message_batch_status = CLIENT.messages.batches.retrieve(
                batch.id
            )
            print(message_batch_status)
            batch_finished = message_batch_status.processing_status == "ended"

        print("  finished. Grabbing data")
        for result in CLIENT.messages.batches.results(
            batch.id,
        ):
            try:
                tags_dict = json.loads(result.result.message.content[0].text)
                tags_dict = {key: tags_dict[key] for key in tags_dict.keys() if key not in tags_to_ignore}

                with open(os.path.join(tags_folder_file_name, result.custom_id + ".json"), 'w') as file:
                    json.dump(tags_dict, file, indent=4)
            except Exception as e:
                print(e)

