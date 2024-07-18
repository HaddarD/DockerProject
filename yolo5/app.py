import time
from pathlib import Path
from flask import Flask, request, jsonify
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
    app.logger.info("Predict endpoint was hit")
    # img_name = request.args.get('imgName')
    if 'imgName' in request.args:
        img_name = request.args['imgName']
        app.logger.info(f"Received request for image: {img_name}")

    else:
        return 'Error: imgName parameter is required', 400
    logger.info(f'Received image name: {img_name}')

    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')

    original_img_path = Path(f'images/{img_name}')
    try:
        download_from_s3(images_bucket, img_name, str(original_img_path))
    except ClientError as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    logger.info(f'Prediction: {prediction_id}. Download img completed')

    try:
        run(
            weights='yolov5s.pt',
            data='data/coco128.yaml',
            source=str(original_img_path),
            project='static/data',
            name=prediction_id,
            save_txt=True
        )
    except Exception as e:
        logger.error(f'Error during prediction: {e}')
        return jsonify({"status": "error", "message": str(e)}), 500

    logger.info(f'Prediction: {prediction_id}. done')

    predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')
    try:
        upload_to_s3(str(predicted_img_path), f'{prediction_id}/{img_name}')
    except ClientError as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')
    logger.info(f"Looking for prediction summary at {pred_summary_path}")

    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            if not labels:
                logger.error(f'Prediction result is empty at {pred_summary_path}')
                return jsonify({"status": "error", "message": "Prediction result is empty"}), 404

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
            # Attempt to insert the document
            insert_result = collection.insert_one(prediction_summary)

            # Log the result of the insertion
            logger.info(f"Inserted ID: {insert_result.inserted_id}")

            # Retrieve the inserted document
            inserted_doc = collection.find_one({'_id': insert_result.inserted_id})

            # Create a new dict without the '_id' field
            response_summary = {k: v for k, v in inserted_doc.items() if k != '_id'}

            # Add the inserted ID as a string
            response_summary['mongo_id'] = str(insert_result.inserted_id)

            logger.info(f"Response summary: {response_summary}")

            return jsonify({
                "status": "success",
                "message": "Prediction Done Successfully :D",
                "result_path": response_summary
            }), 200

        except Exception as e:
            logger.error(f"Error during MongoDB operation: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
