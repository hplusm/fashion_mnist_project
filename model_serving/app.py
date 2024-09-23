import os
import mlflow
import mlflow.keras
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import requests
import logging 
import time 
from collections import deque 
from threading import Lock 
from scipy.stats import ks_2samp
import psutil
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Model Configuration
mlflow_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(mlflow_uri)
experiment_name = "Fashion MNIST Classification"

# Check MLflow server status
try:
    response = requests.get(mlflow_uri)
    print(f"MLflow server status: {response.status_code}")
except requests.ConnectionError:
    print("MLflow server is not accessible")
    exit(1)

# Set up the experiment in MLflow
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

# Initialize request tracking and drift detection variables
request_times = deque(maxlen=100)  # Store last 100 request times 
request_lock = Lock() 

BASELINE_DISTRIBUTION = None
CURRENT_DISTRIBUTION = [0] * 10  # Assuming 10 classes for Fashion MNIST
DRIFT_THRESHOLD = 0.1  # Adjust this value based on your needs
DISTRIBUTION_WINDOW = 1000  # Number of predictions to consider for drift detection

def update_distribution(predicted_class):
    """
    Update the distribution of predicted classes and check for drift.
    
    Args:
        predicted_class: The class predicted by the model.
    """
    global CURRENT_DISTRIBUTION, BASELINE_DISTRIBUTION
    CURRENT_DISTRIBUTION[predicted_class] += 1
    if BASELINE_DISTRIBUTION is None and sum(CURRENT_DISTRIBUTION) >= DISTRIBUTION_WINDOW:
        BASELINE_DISTRIBUTION = CURRENT_DISTRIBUTION.copy()
        logger.info("Baseline distribution set")
    elif sum(CURRENT_DISTRIBUTION) >= DISTRIBUTION_WINDOW:
        check_for_drift()
        CURRENT_DISTRIBUTION = [0] * 10  # Reset after checking for drift

def check_for_drift():
    """
    Check for model drift using the Kolmogorov-Smirnov test.
    """
    global BASELINE_DISTRIBUTION, CURRENT_DISTRIBUTION
    current_dist = np.array(CURRENT_DISTRIBUTION) / sum(CURRENT_DISTRIBUTION)
    baseline_dist = np.array(BASELINE_DISTRIBUTION) / sum(BASELINE_DISTRIBUTION)
    
    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(current_dist, baseline_dist)
    
    logger.info(f"Current KS statistic: {ks_statistic}, p-value: {p_value}")
    
    if ks_statistic > DRIFT_THRESHOLD:
        logger.warning(f"Potential model drift detected. KS statistic: {ks_statistic}")
    
    # Update BASELINE_DISTRIBUTION with a moving average
    BASELINE_DISTRIBUTION = [0.9 * b + 0.1 * c for b, c in zip(BASELINE_DISTRIBUTION, CURRENT_DISTRIBUTION)]

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    
    Returns:
        JSON response with predicted class, processing time, and requests per minute.
    """
    start_time = time.time()
    logger.info("Received prediction request")

    with request_lock: 
        request_times.append(start_time) 

    if 'file' not in request.files:
        logger.error("No file provided in the request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = preprocess_image(file)
    
    # Model Inference
    try:
        with mlflow.start_run(run_id=run_id):
            predictions = model.predict(img)
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            
            # Log prediction to MLflow
            mlflow.log_metric("prediction", predicted_class)
        
        logger.info(f"Prediction made: {predicted_class}")
        
        # Update distribution for drift detection
        update_distribution(predicted_class)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Prediction request processed in {processing_time:.4f} seconds")

    return jsonify({
        'predicted_class': predicted_class, 
        'processing_time': processing_time,
        'requests_per_minute': calculate_requests_per_minute()
    })

def calculate_requests_per_minute():
    """
    Calculate the number of requests per minute.
    
    Returns:
        float: Requests per minute.
    """
    with request_lock:
        if len(request_times) < 2:
            return 0
        time_diff = request_times[-1] - request_times[0]
        if time_diff == 0:
            return 0
        return (len(request_times) - 1) / (time_diff / 60)

@app.route('/metrics', methods=['GET'])
def metrics():
    """
    Endpoint to get system metrics.
    
    Returns:
        JSON response with CPU usage, memory usage, and requests per minute.
    """
    global cpu_usage, memory_usage
    return jsonify({
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'requests_per_minute': calculate_requests_per_minute()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint for health check.
    
    Returns:
        JSON response with health status.
    """
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'}), 200

@app.route('/drift-status', methods=['GET'])
def drift_status():
    """
    Endpoint to get the current drift status.
    
    Returns:
        JSON response with KS statistic, p-value, and drift status.
    """
    global BASELINE_DISTRIBUTION, CURRENT_DISTRIBUTION
    if BASELINE_DISTRIBUTION is None:
        return jsonify({'status': 'Baseline not set yet'})
    
    current_dist = np.array(CURRENT_DISTRIBUTION) / sum(CURRENT_DISTRIBUTION)
    baseline_dist = np.array(BASELINE_DISTRIBUTION) / sum(BASELINE_DISTRIBUTION)
    ks_statistic, p_value = ks_2samp(current_dist, baseline_dist)
    
    return jsonify({
        'ks_statistic': float(ks_statistic),
        'p_value': float(p_value),
        'threshold': DRIFT_THRESHOLD,
        'status': 'Drift detected' if ks_statistic > DRIFT_THRESHOLD else 'No drift detected',
        'current_distribution': current_dist.tolist(),
        'baseline_distribution': baseline_dist.tolist()
    })

# Global variables for resource monitoring
cpu_usage = 0
memory_usage = 0

def update_resource_usage():
    """
    Update CPU and memory usage periodically.
    """
    global cpu_usage, memory_usage
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        time.sleep(5)  # Update every 5 seconds

# Start the resource monitoring thread
threading.Thread(target=update_resource_usage, daemon=True).start()

if __name__ == '__main__':
    logger.info("Starting the Flask application")
    app.run(debug=True, port=5001)  # Using port 5001 to avoid conflict with MLflow UI