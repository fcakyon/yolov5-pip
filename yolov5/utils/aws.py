import concurrent.futures
import logging
import os
from pathlib import Path

import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from tqdm import tqdm

from yolov5.utils.general import colorstr

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

LOGGER = logging.getLogger(__name__)


def parse_s3_uri(s3_uri):
    # strip 's3://'
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]
    # parse bucket and key
    s3_components = s3_uri.split("/")
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = "/".join(s3_components[1:])
    return bucket, s3_key

def upload_file_to_s3(local_file, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    # parse s3 uri
    bucket, s3_key = parse_s3_uri(s3_file)
    # upload to s3
    try:
        s3.upload_file(local_file, bucket, s3_key)
        return True
    except FileNotFoundError:
        print(f"{colorstr('aws:')} S3 upload failed because local file not found: {local_file}")
        return False
    except NoCredentialsError:
        print(f"{colorstr('aws:')} AWS credentials are not set. Please configure aws via CLI or set required ENV variables.")
        return False

def upload_single_file(client, bucket, local_path, s3_path):
    try:
        client.head_object(Bucket=bucket, Key=s3_path)
        return 1
    except:
        #print("Uploading %s..." % s3_path)
        client.upload_file(local_path, bucket, s3_path)
        return 0

def upload_folder_to_s3(local_folder, s3_folder):
    client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    # parse s3 uri
    bucket, s3_key = parse_s3_uri(s3_folder)
    # enumerate local files recursively
    client_list = []
    bucket_list = []
    local_path_list = []
    s3_path_list = []
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full path
            s3_path = s3_key
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(s3_path, relative_path)

            # append
            client_list.append(client)
            bucket_list.append(bucket)
            local_path_list.append(local_path)
            s3_path_list.append(s3_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as conc_exec:
        already_exist_list = list(tqdm(
            conc_exec.map(upload_single_file, client_list, bucket_list, local_path_list, s3_path_list),
            total=len(s3_path_list),
            desc=f"{colorstr('aws:')} Uploading dataset to S3"
            )
        )
    num_already_exist = np.sum(np.array(already_exist_list))
    if num_already_exist > 0:
        LOGGER.warning(f"{colorstr('aws:')} Skipped {num_already_exist} items since they already exists.")
