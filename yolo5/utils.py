import subprocess
import logging
import os
import requests

logger = logging.getLogger(__name__)

def download_from_s3(bucket_name, s3_key, local_path):
    command = f'aws s3 cp s3://{bucket_name}/{s3_key} {local_path}'
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f'Successfully downloaded {s3_key} from {bucket_name}')
    except subprocess.CalledProcessError as e:
        logger.error(f'Error downloading from S3: {e}')
        raise

def upload_to_s3(local_path, s3_key):
    bucket_name = os.environ['BUCKET_NAME']
    command = f'aws s3 cp {local_path} s3://{bucket_name}/{s3_key}'
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f'Successfully uploaded {local_path} to {bucket_name}/{s3_key}')
    except subprocess.CalledProcessError as e:
        logger.error(f'Error uploading to S3: {e}')
        raise

def predict_image(bucket_name, yolo_service_url, img_path):
    # Upload the image to S3
    s3_key = os.path.basename(img_path)
    upload_to_s3(img_path, s3_key)

    # Send a request to the YOLO5 service for prediction
    response = requests.post(f'{yolo_service_url}/predict', params={'imgName': s3_key})
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


class TryExcept:
    @staticmethod
    def attempt(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e
