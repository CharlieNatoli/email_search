

import os
import json
from directories import IMAGES_FOLDER, IMAGE_TAG_SETS_FOLDER
from image_prompting import  create_image_tags

def create_email_tags_dataset(index_name, prompt):

    tags_folder_file_name = os.path.join(IMAGE_TAG_SETS_FOLDER, index_name)
    if not os.path.isdir(tags_folder_file_name):
        os.chdir(tags_folder_file_name)

    for image_file in os.listdir(IMAGES_FOLDER):
        if ".png" not in image_file:
            continue

        json_file_name = image_file.replace(".png", ".json")

        if json_file_name in os.listdir(tags_folder_file_name):
            continue

        try:
            email_tags = create_image_tags(image_file, prompt)
        except:
            continue

        print("creating file")
        with open(os.path.join(tags_folder_file_name, json_file_name), 'w') as file:
            json.dump(email_tags, file, indent=4)
