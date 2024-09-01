import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras import layers, models
from PIL import Image

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
    
    model.fit(train_images, train_labels, epochs=10)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    model.save('model.keras')
    print('Model saved to model.keras')

if __name__ == "__main__":
    main()