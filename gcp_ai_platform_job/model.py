import numpy as np
import datetime, os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, BatchNormalization, Input, concatenate, Activation
from tensorflow.keras.optimizers import Adam

#gcp api
from google.cloud import storage

# parameters
LABEL_COLUMN_INDEX = 0

def _get_blob_bucket(bucket_name, path, destination):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(path)
    blob.download_to_filename(destination)
    
    return destination

def _upload_blob_bucket(bucket_name, source_file_name, destination):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob = bucket.blob(destination)
    blob.upload_from_filename(source_file_name)


def data_loader_txt(path, label_column_index, skiprows=1, delimiter=',', normalize=True):
    '''
    Light-weight txt files data loader that uses numpy as backend
    '''
    raw = np.loadtxt(path, skiprows=1, delimiter=',')
    x = raw[:, label_column_index+1:]
    y = raw[:, label_column_index, np.newaxis] - 1. # -1 to report label in the range 0-1
    
    if normalize: # standard normalization
        x = ((x - np.mean(x, axis=1).reshape(-1,1)) / np.std(x, axis=1).reshape(-1,1))

    return x, y

def expand_inputs(x_train, x_test):
    # expand one dimension to work with 1D CNN layer
    x_train_exp = np.expand_dims(x_train, axis=2)
    x_test_exp = np.expand_dims(x_test, axis=2)
    return x_train_exp, x_test_exp


def model_creation(input_shape):
    return tf.keras.models.Sequential([
        Conv1D(filters=8, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPool1D(strides=4),
        BatchNormalization(),
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        MaxPool1D(strides=4),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPool1D(strides=4),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPool1D(strides=4),
        Flatten(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])


def batch_generator(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    
    yes_idx = np.where(y_train[:,0] == 1.)[0]
    non_idx = np.where(y_train[:,0] == 0.)[0]
    
    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)
    
        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch


def train_and_evaluate(args):
    train_dataset_path = _get_blob_bucket(args["bucket"], "data/exoTrain.csv", "exoTrain.csv")
    test_dataset_path = _get_blob_bucket(args["bucket"], "data/exoTest.csv", "exoTest.csv")

    x_train, y_train = data_loader_txt(train_dataset_path, label_column_index=LABEL_COLUMN_INDEX) 
    x_test, y_test = data_loader_txt(test_dataset_path, label_column_index=LABEL_COLUMN_INDEX)

    # perform expansion of dimension
    x_train, x_test = expand_inputs(x_train, x_test)

    # define model architecture
    model = model_creation(input_shape=x_train.shape[1:])
 
    model.compile(optimizer=Adam(3e-5),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall()])

    logdir = os.path.join("gs://",args["bucket"], args["output-dir"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model.fit(batch_generator(x_train, y_train, args["batch_size"]), 
            epochs=args["num_epochs"], 
            validation_data=(x_test, y_test),
            steps_per_epoch=x_train.shape[1]//args["batch_size"],
            callbacks=[tensorboard_callback])

    # model saving inside log directory
    tf.saved_model.save(obj=model, export_dir=logdir)
