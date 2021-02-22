'''
Small wrapper around the xgboost train function.
This code perform:

- Load DMatrix from GCP bucket
- Run XGBoost model Training
- Save model inside the Bucket
'''
import numpy as np
import argparse
import logging
import joblib
import xgboost as xgb
from google.cloud import storage

# logger configuration
logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

def _get_blob_bucket(bucket_name, path, destination):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(path)
    blob.download_to_filename(destination)
    
    logging.info('Exoplanets Pipeline: Blob {} downloaded to {}'.format(path, 
                                                                        destination))
    return destination

def _upload_blob_bucket(bucket_name, source_file_name, destination):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(destination)
    blob.upload_from_filename(source_file_name)
    
    logging.info('Exoplanets Pipeline: Blob {} uploaded to {}'.format(source_file_name,
                                                                      destination))
def _xgb_parameters(scale_positive):
    
    parameters = {
    'eta': 0.05,
    'objective': 'binary:logistic',
    'max_depth': 7, # depth of the trees in the boosting process
    'min_child_weight': 1, 
    'colsample_bytree': 0.8,
    'scale_pos_weight' : scale_positive
    }
    
    return parameters

def trainer(bucket):
    
    logging.info('Exoplanets Pipeline - Trainer: Load DMatrices from {0} bucket'.format(bucket))
    # define input bucket-path of the data
    train_dmatrix_path = _get_blob_bucket(bucket, "preprocess/train_dmatrix.data", "train_dmatrix.data")
    eval_dmatrix_path = _get_blob_bucket(bucket, "preprocess/eval_dmatrix.data", "eval_dmatrix.data")
    
    dtrain = xgb.DMatrix(train_dmatrix_path)
    deval = xgb.DMatrix(eval_dmatrix_path)
    
    label, count = np.unique(dtrain.get_label(), return_counts=True)
    scale_positive = count[0] / count[1]
    watchlist = [(dtrain, 'train'), (deval, 'eval')]
    
    ESTIMATORS = 300
    model = xgb.train(_xgb_parameters(scale_positive), dtrain, ESTIMATORS, watchlist, early_stopping_rounds=10)
    
    logging.info('Exoplanets Pipeline - Trainer: Storing model on the bucket')
    joblib.dump(model, 'xgb_model.pkl') 
    _upload_blob_bucket(bucket, "xgb_model.pkl", "model/xgb_model.pkl")
    
    logging.info('Exoplanets Pipeline - Trainer: Job Completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_bucket",
        help = "GCS bucket where datasets will be loaded and pushed.",
        required = True
    )

    args = parser.parse_args()
    
    logging.info('Exoplanets Pipeline - Trainer: Running...')
    trainer(args.input_bucket)
