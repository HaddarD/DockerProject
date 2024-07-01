# Use an official Python runtime as a parent image
FROM python:3.10-slim
# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#Source the .env file for variables
ARG PRIVATE_KEY
ENV PRIVATE_KEY=${PRIVATE_KEY}
# Define the command to run the bot.py
CMD ["python3", "app.py"]
