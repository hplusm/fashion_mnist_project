# Data Ingestion

This module handles the ingestion of raw images into an S3 bucket.

## Scripts

- `ingest.py`: Uploads images from the `data/raw` folder to the specified S3 bucket.

## Usage

1. Ensure AWS CLI is configured with the necessary permissions.
2. Run the script:
   ```sh
   python data_ingestion/ingest.py
   ```

## Configuration

- Modify `BUCKET_NAME` and `FOLDER_PATH` in `ingest.py` as needed.