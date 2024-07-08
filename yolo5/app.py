import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
from pymongo import MongoClient
import subprocess
import os

# Environment variables
images_bucket = os.environ['BUCKET_NAME']

mongo_client = MongoClient(os.environ['MONGO_URI'])
db = mongo_client['predictions_db']
collection = db['predictions']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

# Initialize Flask
app = Flask(__name__)


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
    command = f'aws s3 cp {local_path}/{s3_key} s3://{bucket_name}/{s3_key}'
    logger.info(command)
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f'Successfully uploaded {local_path}/{s3_key} to {bucket_name}/{s3_key}')
    except subprocess.CalledProcessError as e:
        logger.error(f'Error uploading to S3: {e}')
        raise


@app.route('/predict', methods=['POST'])
def predict():
    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')

    img_name = request.args.get('imgName')
    logger.info(img_name)

    original_img_path = f'static/data/{img_name}'
    download_from_s3(images_bucket, img_name, str(original_img_path))
    logger.info(f'Prediction: {prediction_id}/{original_img_path}. Download img completed')

    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=str(original_img_path),
        project='static/data',
        name=prediction_id,
        save_txt=True
    )
    logger.info(f'Prediction: {prediction_id}/{original_img_path}. done')

    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

    upload_to_s3(str(predicted_img_path), f'{img_name}')

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

        logger.info(f'Prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': str(original_img_path),
            'predicted-img_path': str(predicted_img_path),
            'labels': labels,
            'time': time.time()
        }

        try:
            collection.insert_one(prediction_summary)
            logger.info(f'Prediction: {prediction_id}/{original_img_path}. prediction summary stored in MongoDB')
        except Exception as e:
            logger.error(f'Error storing prediction summary in MongoDB: {e}')
            return f'Error storing prediction summary in MongoDB: {e}', 500

        return prediction_summary
    else:
        return f'Prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


    # TODO download img_name from S3, store the local image path in the original_img_path variable.
    #  The bucket name is provided as an env var BUCKET_NAME.

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).

        # TODO store the prediction_summary in MongoDB

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)

