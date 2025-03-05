import os
import shutil


from utlities.generate_rag_keywords import _name_for_anthropic_id
from utlities.directories import IMAGES_FOLDER, PROJECT_BASE_PATH

old_images_dir = os.path.join(PROJECT_BASE_PATH, "example_email_images_OLD_NAMES")

# Get all files in the directory
files = os.listdir(old_images_dir)

# Common image file extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

# Process each file
for filename in files:
    file_path = os.path.join(old_images_dir, filename)

    # Skip directories
    if os.path.isdir(file_path):
        continue

    # Check if the file is an image
    is_image = False

    # Method 1: Check file extension
    _, ext = os.path.splitext(filename)
    # Check if the file is an image
    is_image = not file_path.startswith(".png")

    # If it's an image, create a renamed copy
    if is_image:
        # Get the new name
        base_name = os.path.basename(filename)
        new_name = _name_for_anthropic_id(base_name) + ext

        # Create the full output path
        output_path = os.path.join(IMAGES_FOLDER, new_name)

        # Copy the file with the new name
        shutil.copy2(file_path, output_path)
        print(f"Copied {filename} to {new_name}")
