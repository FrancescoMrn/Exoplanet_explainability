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

# logger configuration
logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

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

    logging.info('Local data loader running')
    TRAIN_SET_PATH = "data/exoTrain.csv"
    TEST_SET_PATH = "data/exoTest.csv"
    x_train, y_train = _local_loader(path=TRAIN_SET_PATH, label_column_index=LABEL_COLUMN_INDEX)
    x_test, y_test = _local_loader(path=TEST_SET_PATH, label_column_index=LABEL_COLUMN_INDEX)


    # save results as DMatrix for better performances on xgboost
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    logging.info('Store data set as DMatrix')
    xgb.DMatrix.save_binary(dtrain, 'train_DMatrix.data')
    xgb.DMatrix.save_binary(dtest, 'test_DMatrix.data')
    logging.info('DMatrices saving completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        help = "GCS bucket where datasets will be loaded and pushed.",
        required = True
    )

    args = parser.parse_args()

    gs = "gs://"
    bucket = args.bucket if gs in args.bucket else os.path.join(gs, args.bucket)

    # define input bucket-path of the data
    data_dir = os.path.join(bucket, 'data')
    train_import_path = os.path.join(data_dir, "exoTrain.csv")
    test_import_path = os.path.join(data_dir, "exoTest.csv")

    # define output bucket-path of the processed data
    preprocess_dir = os.path.join(bucket, 'preprocess')
    train_preprocess_path = os.path.join(preprocess_dir, "train_dmatrix.data")
    test_preprocess_path = os.path.join(preprocess_dir, "test_dmatrix.data")
    logging.info('Exoplanets Pipeline - Preprocess')
    preprocess_data()
