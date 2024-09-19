import os
import requests
import mlflow
import mlflow.keras

def batch_predict(image_folder, api_url):
    mlflow.set_experiment("Fashion MNIST Batch Prediction")
    
    with mlflow.start_run():
        predictions = []
        for filename in os.listdir(image_folder):
            if filename.endswith('.png'):
                file_path = os.path.join(image_folder, filename)
                with open(file_path, 'rb') as img_file:
                    response = requests.post(api_url, files={'file': img_file})
                    try:
                        result = response.json()
                        predictions.append(result['predicted_class'])
                        print(f'Predictions for {filename}: {result}')
                    except requests.exceptions.JSONDecodeError:
                        print(f'Failed to get JSON response for {filename}: {response.text}')
        
        mlflow.log_metric("total_predictions", len(predictions))
        mlflow.log_metric("average_prediction", sum(predictions) / len(predictions))

if __name__ == "__main__":
    image_folder = 'data/processed/test'  # Change this to the folder containing new images
    api_url = 'http://127.0.0.1:5000/predict'
    batch_predict(image_folder, api_url)