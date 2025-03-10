import os


from dotenv import load_dotenv

load_dotenv()

PROJECT_BASE_PATH = os.environ.get("PROJECT_DATA_ROOT")

IMAGES_FOLDER = os.path.join(PROJECT_BASE_PATH, "example_email_images")
IMAGE_TAG_SETS_FOLDER = os.path.join(PROJECT_BASE_PATH, "image_tag_sets")

PROJECT_ASSETS_FOLDER = os.environ.get("PROJECT_ASSETS_FOLDER")