import base64
import os
import logging

logger = logging.getLogger(__name__)


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


def llama_cpp_image_handler(image_url_or_path: str):
    if image_url_or_path.startswith("http"):
        return image_url_or_path
    else:
        if not os.path.exists(image_url_or_path):
            raise ValueError(f"Path {image_url_or_path} does not exist")
        return image_to_base64_data_uri(image_url_or_path)
