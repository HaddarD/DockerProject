import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
from pymongo import MongoClient
import os
import boto3
from botocore.exceptions import ClientError

logger = logger.opt(colors=True)

# Environment variables
images_bucket = os.getenv('BUCKET_NAME')

mongo_client = MongoClient(os.environ['MONGO_URI'])
db = mongo_client['predictions_db']
collection = db['predictions']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

# Initialize Flask
app = Flask(__name__)


def download_from_s3(bucket_name, s3_key, local_path):
    s3_client = boto3.client('s3')

    try:
        local_path = Path(local_path)
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
        local_file_path = str(local_path.resolve())
        s3_client.download_file(bucket_name, s3_key, local_file_path)
        logger.info(f'<green>Successfully downloaded {s3_key} from {bucket_name}</green>')
    except ClientError as e:
        logger.error(f'<red>Error downloading from S3: {e}</red>')
        raise


def upload_to_s3(local_path, s3_key):
    bucket_name = os.getenv('BUCKET_NAME')
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        image_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        logger.info(f'<green>Successfully uploaded {local_path} to s3://{bucket_name}/{s3_key}</green>')
        return image_url
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        raise

    # try:
    #     s3_client.upload_file(local_path, bucket_name, s3_key)
    #     logger.info(f'<green>Successfully uploaded {local_path} to s3://{bucket_name}/{s3_key}</green>')
    # except ClientError as e:
    #     logger.error(f'<red>Error uploading to S3: {e}</red>')
    #     raise


@app.route('/predict', methods=['POST'])
def predict():
    # img_name = request.args.get('imgName')
    if 'imgName' in request.args:
        img_name = request.args['imgName']
    else:
        return 'Error: imgName parameter is required', 400
    logger.info(f'Received image name: {img_name}')

    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')

    original_img_path = Path(f'images/{img_name}')
    download_from_s3(images_bucket, img_name, str(original_img_path))
    logger.info(f'Prediction: {prediction_id}. Download img completed')

    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=str(original_img_path),
        project='static/data',
        name=prediction_id,
        save_txt=True
    )
    logger.info(f'Prediction: {prediction_id}. done')

    predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')
    upload_to_s3(str(predicted_img_path), f'{prediction_id}/{img_name}')

    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(label[0])],
                'cx': float(label[1]),
                'cy': float(label[2]),
                'width': float(label[3]),
                'height': float(label[4]),
            } for label in labels]

        logger.info(f'Prediction: {prediction_id}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': str(original_img_path),
            'predicted-img_path': str(predicted_img_path),
            'labels': labels,
            'time': time.time()
        }

        try:
            collection.insert_one(prediction_summary)
            logger.info(f'Prediction: {prediction_id}. prediction summary stored in MongoDB')
        except Exception as e:
            logger.error(f'Error storing prediction summary in MongoDB: {e}')
            return f'Error storing prediction summary in MongoDB: {e}', 500

        return f'Prediction processed successfully:\n{prediction_summary}', 200
    else:
        return f'Prediction: {prediction_id}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
