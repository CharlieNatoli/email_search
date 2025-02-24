
import io
import base64
import anthropic
import os
import json
from PIL import Image

from directories import IMAGES_FOLDER, IMAGE_TAG_SETS_FOLDER

def _pil_image_to_base64(pil_image):
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


CLIENT = anthropic.Anthropic()


def create_image_tags(image_file, data_extraction_prompt, tags_to_ignore = []):
    image_path = os.path.join(IMAGES_FOLDER, image_file)
    image = Image.open(image_path)

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
                            "data": _pil_image_to_base64(image),
                        },
                    },
                ]
            }
        ]
    )
    tags_dict = json.loads(message.content[0].text)
    tags_dict = {key: tags_dict[key] for key in tags_dict.keys() if key not in tags_to_ignore}
    return tags_dict

