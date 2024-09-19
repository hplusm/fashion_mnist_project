# Fashion MNIST Project

This project automates the sorting of returned items for an online shopping platform using machine learning. The system categorizes items based on their images and runs as a service that can be triggered in batches overnight.

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

3. **Start MLflow UI**:
   Before running any scripts, start the MLflow tracking server:
   ```sh
   mlflow ui
   ```
   This will start the MLflow UI on `http://localhost:5000`.

## Data Ingestion

1. **Upload Images to S3**:
   Ensure your images are in the `data/raw` folder. Run the following script to upload images to your S3 bucket:
   ```sh
   python data_ingestion/ingest.py
   ```

## Data Processing

1. **Download and Process Images**:
   Download images from S3, convert them to a standard format, and save them locally:
   ```sh
   python data_processing/process.py
   ```

## Model Training

1. **Train the Model**:
   Train the machine learning model using the processed images:
   ```sh
   python model_training/train.py
   ```
   This script will log training parameters, metrics, and the model to MLflow.

2. **View Training Results**:
   Open your web browser and navigate to `http://localhost:5000` to view your experiments, runs, and metrics in the MLflow UI.

## Model Serving

1. **Start the Flask API**:
   Serve the trained model using Flask:
   ```sh
   python model_serving/app.py
   ```
   The API will be available at `http://127.0.0.1:5001`.

## Testing the API

1. **Test with `curl`**:
   Use `curl` to test the API with a sample image:
   ```sh
   curl -X POST -F "file=@data/processed/test/0_1.png" http://127.0.0.1:5001/predict
   ```

2. **Test with Postman**:
   - Open Postman and create a new POST request.
   - Set the URL to `http://127.0.0.1:5001/predict`.
   - In the **Body** tab, select **form-data**.
   - Add a key named `file` and set the type to **File**.
   - Choose an image file to upload.
   - Send the request and check the response.

## Batch Prediction

1. **Run Batch Predictions**:
   Perform batch predictions on new images:
   ```sh
   python batch_prediction/batch_predict.py
   ```
   This script will log the total number of predictions and the average prediction value to MLflow.

2. **View Batch Prediction Results**:
   Check the MLflow UI again to see the logged metrics from the batch prediction run.

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

## MLflow Integration

This project uses MLflow for experiment tracking and model management. MLflow is integrated into the following scripts:

- `model_training/train.py`: Logs training parameters, metrics, and the trained model.
- `model_serving/app.py`: Automatically loads the latest trained model and logs predictions.
- `batch_prediction/batch_predict.py`: Logs batch prediction metrics.

You can use the MLflow UI to compare different runs, view performance metrics, and manage your models throughout the machine learning workflow.

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed.
2. Check that the MLflow UI is running before executing other scripts.
3. Verify that your AWS credentials are correctly set up for S3 access.
4. If the Flask API fails to start, ensure no other process is using port 5001.

For more detailed information about each script, refer to the comments within the code files.