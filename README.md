# Fashion MNIST Project

This project aims to automate the sorting of returned items for an online shopping platform using machine learning. The system categorizes items based on their images and runs as a service that can be triggered in batches overnight.

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip (Python package installer)
- AWS CLI (for S3 access)

## Setup

1. **Clone the Repository**:
   ```sh
   git clone <repository_url>
   cd fashion_mnist_project
   ```

2. **Install Dependencies**:
   Install the required Python packages using pip:
   ```sh
   pip install -r requirements.txt
   ```

## Data Ingestion

1. **Upload Images to S3**:
   Ensure your images are in the `data/raw` folder. Run the following script to upload images to your S3 bucket:
   ```sh
   python data_ingestion/ingest.py
   ```
   Reference:
   ```python:data_ingestion/ingest.py
   startLine: 1
   endLine: 21
   ```

## Data Processing

1. **Download and Process Images**:
   Download images from S3, convert them to a standard format, and save them locally:
   ```sh
   python data_processing/process.py
   ```
   Reference:
   ```python:data_processing/process.py
   startLine: 1
   endLine: 63
   ```

## Model Training

1. **Train the Model**:
   Train the machine learning model using the processed images:
   ```sh
   python model_training/train.py
   ```
   Reference:
   ```python:model_training/train.py
   startLine: 1
   endLine: 47
   ```

## Model Serving

1. **Start the Flask API**:
   Serve the trained model using Flask:
   ```sh
   python model_serving/app.py
   ```
   Reference:
   ```python:model_serving/app.py
   startLine: 1
   endLine: 23
   ```

## Batch Prediction

1. **Run Batch Predictions**:
   Perform batch predictions on new images:
   ```sh
   python batch_prediction/batch_predict.py
   ```
   Reference:
   ```python:batch_prediction/batch_predict.py
   startLine: 1
   endLine: 18
   ```

## Testing the API

1. **Test with `curl`**:
   Use `curl` to test the API with a sample image:
   ```sh
   curl -X POST -F "file=@data/processed/test/0_1.png" http://127.0.0.1:5000/predict
   ```

2. **Test with Postman**:
   - Open Postman and create a new POST request.
   - Set the URL to `http://127.0.0.1:5000/predict`.
   - In the **Body** tab, select **form-data**.
   - Add a key named `file` and set the type to **File**.
   - Choose an image file to upload.
   - Send the request and check the response.

## Scheduling Batch Predictions

1. **Set Up a Cron Job**:
   To automate batch predictions, set up a cron job to run the script every night at midnight:
   ```sh
   crontab -e
   ```
   Add the following line:
   ```sh
   0 0 * * * /usr/bin/python3 /path_to_your_repo/batch_prediction/batch_predict.py
   ```

