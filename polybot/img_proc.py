from pathlib import Path
from matplotlib.image import imread, imsave
import random
import requests
import os
from loguru import logger
import boto3
from botocore.exceptions import ClientError
import json

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class Img:

    def __init__(self, path):
        """
        Do not change the constructor implementation
        """
        self.path = Path(path)
        self.data = rgb2gray(imread(path)).tolist()
        self.bucket_name = os.getenv('BUCKET_NAME')

    def save_img(self):
        """
        Do not change the below implementation
        """
        new_path = self.path.with_name(self.path.stem + '_filtered' + self.path.suffix)
        imsave(new_path, self.data, cmap='gray')
        return new_path

    def blur(self, blur_level=16):
        """
        Applies a blur filter on the image and saves it to be sent back to the user
        :return:
        """
        height = len(self.data)
        width = len(self.data[0])
        filter_sum = blur_level ** 2

        result = []
        for i in range(height - blur_level + 1):
            row_result = []
            for j in range(width - blur_level + 1):
                sub_matrix = [row[j:j + blur_level] for row in self.data[i:i + blur_level]]
                average = sum(sum(sub_row) for sub_row in sub_matrix) // filter_sum
                row_result.append(average)
            result.append(row_result)

        self.data = result

    def contour(self):
        """
        Applies a contour filter on the image and saves it to be sent back to the user
        :return:
        """
        for i, row in enumerate(self.data):
            res = []
            for j in range(1, len(row)):
                res.append(abs(row[j-1] - row[j]))

            self.data[i] = res

    def rotate(self):
        """
        Rotates the image and saves it to be sent back to the user
        :return:
        """
        if not self.data:
            raise RuntimeError("Image data is empty")
        self.data = [list(row) for row in zip(*self.data[::-1])]

    def salt_n_pepper(self, salt_prob=0.05, pepper_prob=0.05):
        """
        Applies a salt & pepper filter on the image and saves it to be sent back to the user
        :return:
        """
        height = len(self.data)
        width = len(self.data[0])
        for i in range(height):
            for j in range(width):
                rand = random.random()
                if rand < salt_prob:
                    self.data[i][j] = 255
                elif rand < salt_prob + pepper_prob:
                    self.data[i][j] = 0

    def concat(self, other_img, direction='/horizontal'):
        """
        merges 2 images into a collage either horizontally or vertically according to the user's choice and saves it to be sent back to the user
        :return:
        """
        if direction == '/horizontal':
            try:
                if len(self.data) != len(other_img.data):
                    raise RuntimeError("Images must have the same height for horizontal concatenation")
                self.data = [row1 + row2 for row1, row2 in zip(self.data, other_img.data)]
            except RuntimeError as e:
                error_message = str(e)
                return error_message, 500
        elif direction == '/vertical':
            try:
                if len(self.data[0]) != len(other_img.data[0]):
                    raise RuntimeError("Images must have the same width for vertical concatenation")
                self.data += other_img.data
            except RuntimeError as e:
                error_message = str(e)
                return error_message, 500
        else:
            return "Invalid direction for concatenation. Must be 'horizontal' or 'vertical'.", 500
        return "Ok", 200

    def segment(self):
        """
        Applies a segment filter on the image and saves it to be sent back to the user
        :return:
        """
        if not self.data:
            raise RuntimeError("Image data is empty")
        total_pixels = sum(sum(row) for row in self.data)
        average = total_pixels // (len(self.data) * len(self.data[0]))
        for i, row in enumerate(self.data):
            self.data[i] = [0 if pixel < average else 255 for pixel in row]

    def upload_and_predict(self, yolo_service_url, image_path, image_name):
        if not image_name:
            raise ValueError("Image name is empty")
        try:
            self.upload_to_s3(image_path, image_name)
            logger.info(f"Successfully uploaded {image_name} to S3")
        except Exception as e:
            logger.exception(f'<red>Error uploading image to S3: {e}</red>')
            raise

        logger.info(f"Starting prediction for image: {image_name}")
        logger.info(f"YOLO service URL: {yolo_service_url}")

        # Send a request to the YOLO5 service for prediction
        full_url = f'{yolo_service_url}/predict'
        logger.info(f"Sending prediction request to: {full_url}")
        try:
            response = requests.post(full_url, params={'imgName': image_name})
            response.raise_for_status()
            logger.info(f"Received response from YOLO5 service: {response.status_code}")
            return json.loads(response.text)
        except requests.RequestException as e:
            logger.exception(f"Error during YOLO5 service request: {e}")
            raise
        # response = requests.post(f'{yolo_service_url}/predict', params={'imgName': image_name})
        # if response.status_code == 200:
        #     return response.json()
        # else:
        #     response.raise_for_status()

    def upload_to_s3(self, image_path, image_name=None):
        if image_name is None:
            image_name = os.path.basename(image_path)

        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(image_path, self.bucket_name, image_name)
            image_url = f"https://{self.bucket_name}.s3.amazonaws.com/{image_name}"
            return image_url
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise

    # def download_from_s3(self, image_name, download_path=None):
    #     if download_path is None:
    #         download_path = image_name
    #
    #     s3_client = boto3.client('s3')
    #     try:
    #         s3_client.download_file(self.bucket_name, image_name, download_path)
    #     except ClientError as e:
    #         logger.error(f"Error downloading file from S3: {e}")
    #         raise
    #