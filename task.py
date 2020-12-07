# This task.py script is used to input all the required argumetnts to the 
# model.py containing the architecture of the TF model. This code structure
# can be easily trained and deployed on GCP

import argparse
import json
import os
from model import train_and_evaluate

import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk"
    )
    parser.add_argument(
        "--train_data_path",
        help="Location of training data",
        required=True
    )
    parser.add_argument(
        "--eval_data_path",
        help="Location of evaluation data",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write checkpoints and export models",
        required=True
    )
    parser.add_argument(
        "--batch_size",
        help="Number of examples to compute gradient over.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs to train the model.",
        type=int,
        default=10
    )
    parser.add_argument(
        "--eval_steps",
        help="""Positive number of steps for which to evaluate model. Default
        to None, which means to evaluate until input_fn raises an end-of-input
        exception""",
        type=int,
        default=None
    )

    # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Run the training job
    train_and_evaluate(arguments)