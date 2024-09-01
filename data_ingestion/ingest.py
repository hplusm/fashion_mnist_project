import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')

def upload_files_to_s3(folder, bucket_name):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            try:
                print(f'Uploading {file_path} to {bucket_name}')
                s3.upload_file(file_path, bucket_name, filename)
                print(f'Uploaded {filename} to {bucket_name}')
            except Exception as e:
                print(f'Failed to upload {filename}: {e}')

if __name__ == "__main__":
    folder = 'data/raw'  
    bucket_name = 'refund-item-images'  
    upload_files_to_s3(folder, bucket_name)