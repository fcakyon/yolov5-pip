import os
from pathlib import Path

import urllib.request

TEST_COCO_DATASET_URL = "https://github.com/fcakyon/yolov5-pip/releases/download/0.6/test-coco-dataset.zip"
TEST_COCO_DATASET_ZIP_PATH = "tests/data/test-coco-dataset.zip"


def download_from_url(from_url: str, to_path: str):

    Path(to_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(to_path):
        urllib.request.urlretrieve(
            from_url,
            to_path,
        )
        
def download_test_coco_dataset_and_unzip():
    if not os.path.exists(TEST_COCO_DATASET_ZIP_PATH):
        print("Downloading test coco dataset...")
        download_from_url(TEST_COCO_DATASET_URL, TEST_COCO_DATASET_ZIP_PATH)
        print("Download complete.")

    if not os.path.exists("tests/data/dataset.json"):
        print("Unzipping test coco dataset...")
        import zipfile
        with zipfile.ZipFile(TEST_COCO_DATASET_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall("tests/data")
        print("Unzip complete.")

if __name__ == "__main__":
    download_test_coco_dataset_and_unzip()