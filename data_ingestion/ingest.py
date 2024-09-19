import boto3
import os

def upload_to_s3(bucket_name: str, folder_path: str) -> None:
    """
    Uploads files from a local folder to an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_path (str): The local folder path containing files to upload.
    """
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Walk through the directory tree
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Construct the full local path of the file
            local_path = os.path.join(root, file)
            
            # Determine the S3 path (key) for the file
            s3_path = os.path.relpath(local_path, folder_path)
            
            # Upload the file to S3
            s3.upload_file(local_path, bucket_name, s3_path)
            
            # Print a confirmation message
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")

if __name__ == "__main__":
    # S3 bucket name where files will be uploaded
    BUCKET_NAME = 'refund-item-images'  
    
    # Local folder path containing files to upload
    FOLDER_PATH = 'data/raw'  
    
    # Call the upload function
    upload_to_s3(BUCKET_NAME, FOLDER_PATH)