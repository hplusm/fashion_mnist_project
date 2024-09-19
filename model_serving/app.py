import os
import mlflow
import mlflow.keras
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import requests

app = Flask(__name__)

# Model Configuration
mlflow_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(mlflow_uri)
experiment_name = "Fashion MNIST Classification"

# Check MLflow server
try:
    response = requests.get(mlflow_uri)
    print(f"MLflow server status: {response.status_code}")
except requests.ConnectionError:
    print("MLflow server is not accessible")
    exit(1)

# Set up the experiment
try:
    mlflow.set_experiment(experiment_name)
    print(f"Successfully set experiment: {experiment_name}")
except Exception as e:
    print(f"Error setting experiment: {e}")
    exit(1)

# Model Loading
try:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time desc"],
        max_results=1
    )

    if runs:
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        print(f"Latest run ID: {run_id}")
        
        # Print the artifact URI
        artifact_uri = mlflow.get_run(run_id).info.artifact_uri
        print(f"Artifact URI: {artifact_uri}")
        
        # List contents of the artifact directory
        artifact_dir = artifact_uri.replace("file://", "")
        print(f"Contents of {artifact_dir}:")
        for root, dirs, files in os.walk(artifact_dir):
            level = root.replace(artifact_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")
        
        # Try to load the model
        try:
            model = mlflow.keras.load_model(f"runs:/{run_id}/model")
            print(f"Loaded model from run: {run_id}")
        except Exception as model_load_error:
            print(f"Error loading model: {model_load_error}")
            print("Attempting to load from artifact URI directly...")
            model_path = os.path.join(artifact_dir, "model")
            if os.path.exists(model_path):
                model = mlflow.keras.load_model(model_path)
                print("Successfully loaded model from artifact URI")
            else:
                print(f"Model directory not found at {model_path}")
    else:
        raise Exception("No runs found for the experiment")
except Exception as e:
    print(f"Error in model loading process: {e}")
    exit(1)

def preprocess_image(file):
    """
    Preprocess the input image for model prediction.
    
    Args:
        file: Input image file object.
    
    Returns:
        np.array: Preprocessed image array of shape (1, 28, 28).
    """
    img = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions on input images.
    
    Returns:
        JSON response with predicted class.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = preprocess_image(file)
    
    # Model Inference
    with mlflow.start_run(run_id=run_id):
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        
        # Log prediction to MLflow
        mlflow.log_metric("prediction", predicted_class)
    
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Using port 5001 to avoid conflict with MLflow UI