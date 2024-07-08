import json
import subprocess
from loguru import logger
import os
import requests
from collections import Counter
import boto3
from botocore.exceptions import ClientError

logger = logger.opt(colors=True)


def download_from_s3(bucket_name, s3_key, local_path):
    s3_client = boto3.client('s3')

    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f'<green>Successfully downloaded {s3_key} from {bucket_name}</green>')
    except ClientError as e:
        logger.error(f'<red>Error downloading from S3: {e}</red>')
        raise


def upload_to_s3(local_path, s3_key):
    bucket_name = os.environ['BUCKET_NAME']
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        logger.info(f'<green>Successfully uploaded {local_path} to s3://{bucket_name}/{s3_key}</green>')
    except ClientError as e:
        logger.error(f'<red>Error uploading to S3: {e}</red>')
        raise


def prediction_decode(prediction_summary):
    try:
        labels = prediction_summary['labels']
        classes = [label['class'] for label in labels]
        quantities = Counter(classes)
        response = "Prediction Summary:\n"
        response += [f"{key.capitalize()} - {value}\n" for key, value in quantities.items()]
        return response
    except (json.JSONDecodeError, KeyError) as e:
        print(f'Error decoding JSON response: {e}')
        return {}


def predict_image(yolo_service_url, img_path):
    # Upload the image to S3
    s3_key = os.path.basename(img_path)
    upload_to_s3(img_path, s3_key)

    # Send a request to the YOLO5 service for prediction
    response = requests.post(f'{yolo_service_url}/predict', params={'imgName': s3_key})
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def try_except():
    @staticmethod
    def attempt(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e
