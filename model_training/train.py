import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras import layers, models
from PIL import Image
import mlflow
import mlflow.keras

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            label = int(filename.split('_')[0])
            img = Image.open(os.path.join(folder, filename))
            img = np.array(img)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def main():
    mlflow.set_experiment("Fashion MNIST Classification")
    
    with mlflow.start_run() as run:
        train_folder = 'data/processed/train'
        test_folder = 'data/processed/test'
        
        train_images, train_labels = load_images_from_folder(train_folder)
        test_images, test_labels = load_images_from_folder(test_folder)
        
        train_images, test_images = train_images / 255.0, test_images / 255.0
        
        model = models.Sequential([
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ])
        
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        history = model.fit(train_images, train_labels, epochs=10)
        
        mlflow.log_param("epochs", 10)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss", "sparse_categorical_crossentropy")
        
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        
        mlflow.keras.log_model(model, "model")
        
        # Log the run ID
        run_id = run.info.run_id
        mlflow.log_param("run_id", run_id)
        
        print(f"Model trained and logged with run ID: {run_id}")

if __name__ == "__main__":
    main()