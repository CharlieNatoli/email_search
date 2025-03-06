
import io
import base64
import anthropic
import os
import json
from typing import List, Dict
from PIL import Image
from tenacity import retry, stop_after_attempt, retry_if_result, wait_exponential
import re
from pillow_avif import AvifImagePlugin
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.message import Message
from anthropic.types.messages.message_batch import MessageBatch
from datetime import datetime


from directories import IMAGES_FOLDER, IMAGE_TAG_SETS_FOLDER



class BaseAnthropicPromptMixin:

    def __init__(self, model="claude-3-5-sonnet-20241022", max_tokens=1000, temperature=0):
        self.CLIENT = anthropic.Anthropic()
        self.claude_config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    @staticmethod
    def _resize_and_crop_image(
            img: Image,
            target_width: int = 600,
            max_height: int = 1400
    ) -> Image:

        width, height = img.size
        aspect_ratio = height / width

        new_width = target_width
        new_height = int(new_width * aspect_ratio)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        if new_height > max_height:
            resized_img = resized_img.crop((0, 0, new_width, max_height))

        return resized_img

    def _image_filename_to_base64(self, image_filename: str) -> str:
        try:
            image_path = os.path.join(IMAGES_FOLDER, image_filename)

            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return ""

            with Image.open(image_path) as image:
                resized_image = self._resize_and_crop_image(image)

                img_buffer = io.BytesIO()

                resized_image.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return img_base64
        except Exception as e:
            print(f"error with rendering email {image_filename}: {e}")
            return base64.b64encode(b"").decode('utf-8')


    @staticmethod
    def _create_tags_dictionary(message : Message, tags_to_ignore: List[str]) -> Dict:
        tags_dict = json.loads(message.content[0].text)
        return {key: tags_dict[key] for key in tags_dict.keys() if key not in tags_to_ignore}



class SingleImagePromptHandler(BaseAnthropicPromptMixin):

    def create_image_tags_single_image(
        self,
        image_file: str,
        data_extraction_prompt: str,
        tags_to_ignore: List[str] = None
    ) -> Dict:

        if tags_to_ignore is None:
            tags_to_ignore = []
        message = self.CLIENT.messages.create(
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
                                "data": self._image_filename_to_base64(image_file),
                            },
                        },
                    ]
                }
            ],
            **self.claude_config
        )

        return self._create_tags_dictionary(message, tags_to_ignore)

class BatchNotReadyException(Exception):
    pass

class KeywordRAGIndexCreator(BaseAnthropicPromptMixin):

    BATCH_SIZE = 50

    def __init__(self, index_name, data_extraction_prompt, tags_to_ignore: List[str] = None):


        super().__init__()
        self.index_name = index_name
        self.tags_folder_file_path = os.path.join(IMAGE_TAG_SETS_FOLDER, self.index_name)
        self.data_extraction_prompt = data_extraction_prompt

        if tags_to_ignore is None:
            tags_to_ignore = []
        self.tags_to_ignore = tags_to_ignore

        # assumed that all files here are images
        self.image_file_names = [
            image_file_name for image_file_name in os.listdir(IMAGES_FOLDER) if not image_file_name.startswith(".")
        ]

    @staticmethod
    def _name_for_anthropic_id(image_file_name: str) -> str:
        # IDs for batch items in Anthropic must be no longer than 64 characters
        # and must not use special characters other than "-" and "_"
        # Filenames are used as IDs, but must conform to these requirements (assumed all filenames are unique)
        id_character_limit = 64
        name, ext = os.path.splitext(image_file_name)
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:id_character_limit]

    def _create_requests_list(self, image_file_names_batch: List[str]) -> List[Request]:
        requests = []

        for image_file_name in image_file_names_batch:
            try:
                # Get image data first and skip if not available
                image_data = self._image_filename_to_base64(image_file_name)
                if not image_data:
                    print(f"Skipping {image_file_name} - could not convert to base64")
                    continue

                requests.append(
                    Request(
                        custom_id=self._name_for_anthropic_id(image_file_name),
                        params=MessageCreateParamsNonStreaming(
                            **self.claude_config,
                            system=self.data_extraction_prompt,
                            messages=[{
                                "role": "user",
                                "content": [{
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_data,
                                    },
                                }]
                            }]
                        )
                    )
                )
            except Exception as e:
                print(f"Error creating request for {image_file_name}: {e}")

        return requests

    def _make_tags_folder(self):
        if not os.path.isdir(self.tags_folder_file_path):
            os.mkdir(self.tags_folder_file_path)

    def _create_batches_in_anthropic(self):
        message_batches = []
        for i in range(0, len(self.image_file_names ), self.BATCH_SIZE):
            print(f'batch starting with image {i}, {datetime.now()}')
            message_batch = self.CLIENT.messages.batches.create(
                requests=self._create_requests_list(
                    image_file_names_batch=self.image_file_names [i:i +  self.BATCH_SIZE]
                )
            )

            message_batches.append(message_batch)
        return message_batches

    @staticmethod
    def _batch_status_not_complete(message_batch):
        return message_batch.processing_status != "ended"

    @retry(wait=wait_exponential(multiplier=1, min=2, max=300),
            stop=stop_after_attempt(5),
           retry=retry_if_result(_batch_status_not_complete))
    def _check_batch_status(self, batch: MessageBatch) -> MessageBatch:
        message_batch = self.CLIENT.messages.batches.retrieve(batch.id)

        status = message_batch.processing_status
        print(f"Batch {batch.id} status: {status}")
        return message_batch

    def _wait_for_batch_to_finish(self, batch: MessageBatch) -> bool:
        print(f"Waiting for batch {batch.id}")

        # Keep trying until batch is complete
        try:
            self._check_batch_status(batch)
            print(f"Batch {batch.id} finished successfully. Grabbing data.")
            return True
        except Exception as e:
            print(f"Batch {batch.id} ended without completion. Unable to process images.")
            return False

    def _extract_and_save_data_from_batch(self, batch: MessageBatch) -> Dict:

        for result in self.CLIENT.messages.batches.results(
                batch.id,
        ):
            try:
                tags_dict = self._create_tags_dictionary(result.result.message, self.tags_to_ignore)

                with open(os.path.join(self.tags_folder_file_path, result.custom_id + ".json"), 'w') as output_file:
                    json.dump(tags_dict, output_file, indent=4)
            except Exception as e:
                print(e)

    def create_image_tags_full_dataset(self) -> None:

        self._make_tags_folder()

        print(f"{len(self.image_file_names)} tag sets to create")
        message_batches = self._create_batches_in_anthropic()

        for batch in message_batches:
            if self._wait_for_batch_to_finish(batch):
                self._extract_and_save_data_from_batch(
                    batch=batch
                )
