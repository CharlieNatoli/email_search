
import io
import base64
import anthropic
import os
import json
from PIL import Image
from pillow_avif import AvifImagePlugin

from directories import IMAGES_FOLDER, IMAGE_TAG_SETS_FOLDER


def _resize_and_crop_image(input_path, target_width=600, max_height=1400):
    # Open the image
    img = Image.open(input_path)

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
    image_format = image_filename.split(".")[-1]
    image_path = os.path.join(IMAGES_FOLDER, image_filename)
    image = _resize_and_crop_image(Image.open(image_path))

    img_buffer = io.BytesIO()

    image.save(img_buffer, format=image_format)
    img_bytes = img_buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


CLIENT = anthropic.Anthropic()


def create_image_tags(image_file, data_extraction_prompt, tags_to_ignore = []):


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

