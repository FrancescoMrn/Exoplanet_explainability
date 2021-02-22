'''
Small wrapper around the preprocessing function.
This code perform:

- Data loading from local files (debug) or GCP bigquery
- Save data as DMatrix
'''
import os
import argparse
import logging
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
    
def _local_loader(path, label_column_index, skiprows=1, delimiter=','):
    '''
    Light-weight txt files data loader that uses numpy as backend
    '''
    dataset = {}
    raw = np.loadtxt(path, skiprows=1, delimiter=',')
    x = raw[:, label_column_index+1:]
    y = raw[:, label_column_index, np.newaxis] - 1. # -1 to report label in the range 0-1
    return x, y


def preprocess_data(bucket, local=True):
    LABEL_COLUMN_INDEX = 0
    
    logging.info('Exoplanets Pipeline - Preprocess: Paths creation with {0} bucket'.format(bucket))
    # define input bucket-path of the data
    train_dataset_path = _get_blob_bucket(bucket, "data/exoTrain.csv", "exoTrain.csv")
    test_dataset_path = _get_blob_bucket(bucket, "data/exoTest.csv", "exoTest.csv")
                              
    logging.info('Exoplanets Pipeline - Preprocess: Data import')
    x, y = _local_loader(path=train_dataset_path, label_column_index=LABEL_COLUMN_INDEX)
    # create an additonal evaluation split
    x_train, x_eval, y_train, y_eval = train_test_split(x, y,
                                                        stratify=y,
                                                        test_size=0.25)
    x_test, y_test = _local_loader(path=test_dataset_path, label_column_index=LABEL_COLUMN_INDEX)
    # save results as DMatrix for better performances on xgboost
    dtrain = xgb.DMatrix(x_train, label=y_train)
    deval = xgb.DMatrix(x_eval, label=y_eval)
    dtest = xgb.DMatrix(x_test, label=y_test)
    
    logging.info('Exoplanets Pipeline - Preprocess: Store data set as DMatrix')
    xgb.DMatrix.save_binary(dtrain, "train_dmatrix.data")
    xgb.DMatrix.save_binary(deval, "eval_dmatrix.data")
    xgb.DMatrix.save_binary(dtest, "test_dmatrix.data")
    
    # upload data on the storage
    _upload_blob_bucket(bucket, "train_dmatrix.data", "preprocess/train_dmatrix.data")
    _upload_blob_bucket(bucket, "eval_dmatrix.data", "preprocess/eval_dmatrix.data")
    _upload_blob_bucket(bucket, "test_dmatrix.data", "preprocess/test_dmatrix.data")
    
    logging.info('Exoplanets Pipeline - Preprocess: Completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_bucket",
        help = "GCS bucket where datasets will be loaded and pushed.",
        required = True
    )

    args = parser.parse_args()
    
    logging.info('Exoplanets Pipeline - Preprocess: Running...')
    preprocess_data(args.input_bucket)
