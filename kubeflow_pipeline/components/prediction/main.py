'''
Small wrapper around to perform prediction over a test dataset.
This code perform:

- Load model from GCP bucket
- Run prediction over test set
- Save confusion matrix inside the bucket
'''
import argparse
import logging
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from google.cloud import storage

# logger configuration
logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

def _get_blob_bucket(bucket_name, path, destination):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(path)
    blob.download_to_filename(destination)
    
    logging.info('Exoplanets Pipeline: Blob {} downloaded to {}'.format(path, destination))
    return destination

def _upload_blob_bucket(bucket_name, source_file_name, destination):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(destination)
    blob.upload_from_filename(source_file_name)
    
    logging.info('Exoplanets Pipeline: Blob {} uploaded to {}'.format(source_file_name,
                                                                      destination))
    
def prediction(bucket):
    
    logging.info('Exoplanets Pipeline - Prediction: Load model from {0} bucket'.format(bucket))
    # define input bucket-path of the model
    model_path = _get_blob_bucket(bucket, "model/xgb_model.pkl", "xgb_model.pkl")
    test_dmatrix_path = _get_blob_bucket(bucket, "preprocess/test_dmatrix.data", "test_dmatrix.data")
    
    model = joblib.load(model_path)
    dtest = xgb.DMatrix(test_dmatrix_path)
    
    logging.info('Exoplanets Pipeline - Prediction: Perform Prediction')
    y_pred = model.predict(dtest)
    y_pred_binary = np.around(y_pred)
    
    logging.info('Exoplanets Pipeline - Prediction: Calculare metrics')
    cm = confusion_matrix(dtest.get_label(), y_pred_binary)
    precision = precision_score(dtest.get_label(), y_pred_binary)
    recall = recall_score(dtest.get_label(), y_pred_binary)
    
    logging.info('Exoplanets Pipeline - Prediction: Confusion matrix: \n {0}'.format(cm))
    logging.info('Exoplanets Pipeline - Prediction: Precision {0}'.format(precision))
    logging.info('Exoplanets Pipeline - Prediction: Recall {0}'.format(recall))
    
    logging.info('Exoplanets Pipeline - Prediction: Complete')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_bucket",
        help = "GCS bucket where datasets will be loaded and pushed.",
        required = True
    )

    args = parser.parse_args()
    
    logging.info('Exoplanets Pipeline - Prediction: Running...')
    prediction(args.input_bucket)
