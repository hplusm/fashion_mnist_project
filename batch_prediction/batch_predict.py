import os
import requests
import mlflow
import mlflow.keras

def batch_predict(image_folder, api_url):
    # Set up MLflow experiment
    mlflow.set_experiment("Fashion MNIST Batch Prediction")
    
    with mlflow.start_run():
        predictions = []
        # Iterate through all PNG files in the specified folder
        for filename in os.listdir(image_folder):
            if filename.endswith('.png'):
                file_path = os.path.join(image_folder, filename)
                # Open and send each image file to the API
                with open(file_path, 'rb') as img_file:
                    response = requests.post(api_url, files={'file': img_file})
                    try:
                        # Parse the JSON response
                        result = response.json()
                        predictions.append(result['predicted_class'])
                        print(f'Predictions for {filename}: {result}')
                    except requests.exceptions.JSONDecodeError:
                        # Handle cases where the response is not valid JSON
                        print(f'Failed to get JSON response for {filename}: {response.text}')
        
        # Log metrics to MLflow
        mlflow.log_metric("total_predictions", len(predictions))
        mlflow.log_metric("average_prediction", sum(predictions) / len(predictions))

if __name__ == "__main__":
    # Configuration for batch prediction
    image_folder = 'data/processed/test'  # Folder containing new images
    api_url = 'http://127.0.0.1:5001/predict'  # URL of the prediction API
    batch_predict(image_folder, api_url)