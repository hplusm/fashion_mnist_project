import os
import boto3
import numpy as np
from PIL import Image

# Initialize S3 client
s3 = boto3.client('s3')

def download_images_from_s3(bucket_name, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    objects = s3.list_objects_v2(Bucket=bucket_name)['Contents']
    for obj in objects:
        file_name = obj['Key']
        file_path = os.path.join(download_folder, file_name)
        s3.download_file(bucket_name, file_name, file_path)
        print(f'Downloaded {file_name} to {download_folder}')

def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        f.read(4) 
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        f.read(8)  
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_images(images, labels, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, (image, label) in enumerate(zip(images, labels)):
        img = Image.fromarray(image, 'L')
        img.save(os.path.join(output_folder, f'{label}_{i}.png'))
        print(f'Saved image {i} with label {label}')

if __name__ == "__main__":
    bucket_name = 'refund-item-images'
    download_folder = 'data/downloaded'
    output_folder = 'data/processed'
    
    download_images_from_s3(bucket_name, download_folder)
    
    train_images_path = os.path.join(download_folder, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(download_folder, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(download_folder, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(download_folder, 't10k-labels-idx1-ubyte')
    
    train_images = read_mnist_images(train_images_path)
    train_labels = read_mnist_labels(train_labels_path)
    test_images = read_mnist_images(test_images_path)
    test_labels = read_mnist_labels(test_labels_path)
    
    save_images(train_images, train_labels, os.path.join(output_folder, 'train'))
    save_images(test_images, test_labels, os.path.join(output_folder, 'test'))