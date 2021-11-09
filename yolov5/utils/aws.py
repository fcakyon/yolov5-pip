import os
from pathlib import Path
import logging
import boto3
from botocore.exceptions import NoCredentialsError
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

def upload_file_to_s3(local_file, s3_dir, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    # parse s3 uri
    bucket, s3_key = parse_s3_uri(s3_dir)
    # upload to s3
    try:
        s3_path = str(Path(s3_key) / s3_file)
        s3.upload_file(local_file, bucket, s3_path)
        return True
    except FileNotFoundError:
        print(f"{colorstr('aws:')} S3 upload failed because local file not found: {local_file}")
        return False
    except NoCredentialsError:
        print(f"{colorstr('aws:')} AWS credentials are not set. Please configure aws via CLI or set required ENV variables.")
        return False
