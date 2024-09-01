import boto3
import os

def upload_to_s3(bucket_name: str, folder_path: str) -> None:
    """
    Uploads files from a local folder to an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_path (str): The local folder path containing files to upload.
    """
    s3 = boto3.client('s3')
    for root, _, files in os.walk(folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.relpath(local_path, folder_path)
            s3.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")

if __name__ == "__main__":
    BUCKET_NAME = 'refund-item-images'  
    FOLDER_PATH = 'data/raw'  
    upload_to_s3(BUCKET_NAME, FOLDER_PATH)