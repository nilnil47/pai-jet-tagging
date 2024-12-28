"""A collection of utility functions that are used in the other files.
They are organised here to make the other files easier to read.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


def get_training_args() -> argparse.Namespace:
    """Get the arguments needed for the training script."""
    # This is the standard way to define command line arguments in python,
    # without using config files which you are welcome to do!
    # Here the argparse module can pick up command line arguments and return them
    # So if you type in `python train.py --num_mlp_layers 3` it will save the value 3
    # to `args.num_mlp_layers`.
    # This allows you to change the model without changing the code!
    # Each possible argument must be defined here.
    # Feel free to add more arguments as you see fit.

    # First we have to create a parser object
    parser = argparse.ArgumentParser()

    # Define the important paths for the project
    parser.add_argument(
        "--model_dir",  # How we access the argument when calling `python train.py ...`
        type=str,  # We must also define the type of argument, here it is a string
        default="models",  # The default value so you dont have to type it in every time
        help="Where to save trained models",  # A helpfull message
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="jets_tranformer_01",
        help="The name of the model",
    )
    
    parser.add_argument(
        "--norm_epsilon",
        type=float,
        default=1e-5,
        help="The value of epsilon for the normalization layers",
    )

    # Arguments for the network
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="The size of the transformer heads",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="The number of tranformer layers",
    )
    parser.add_argument(
        "--intermediate_dim",
        type=int,
        default=32,
        help="The size of tranformer mlp layer",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="The activation function, see https://keras.io/api/layers/activations/",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="The dropout rate",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="batch",
        help="The normalization type",
    )

    # Arguments for how to train the model
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="mse",
        help="The loss function, see https://keras.io/api/losses/",
    )
    parser.add_argument(
        "--network_type",
        type=str,
        default="transformer",
        help="The type of network to use",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="The maximum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
        help="The maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="The weight decay",
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=-1,
        help="The size of the dataset to use for each type. the total size of the dataset will be 3*data_size",
    )
    parser.add_argument(
        "--multi_class",
        type=bool,
        default=False,
        help="Is this a multi class classification problem",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="The optimizer, see https://keras.io/api/optimizers/",
    )

    # This now collects all arguments
    args = parser.parse_args()

    # Now we return the arguments
    return args


def get_evaluation_args() -> argparse.Namespace:
    """Load the arguments needed for the evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Where the saved model are stored",
    )
    parser.add_argument(
        "--re_evelaute",
        type=bool,
        default=False,
        help="Is we want to run the predict on the test set again",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        default="cnn_1,cnn_2,cnn_3",
        help="A comma separated list of model names to load and compare",
    )
    parser.add_argument(
        "--multi_class",
        type=bool,
        default=False,
        help="Is this a multi class classification problem",
    )
    parser.add_argument(
        "--important_args",
        type=str,
        default="max_epochs",
        help="A comma separated list of args to include in the plots",
    )

    args = parser.parse_args()
    args.model_names = args.model_names.split(",")
    args.important_args = args.important_args.split(",")
    return args


def plot_history(
    history: tf.keras.callbacks.History,
    output_path: Path,
    accuracy_key: str = "accuracy",
) -> None:
    """Plot the training history."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Valid")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(history.history[accuracy_key], label="Train")
    axes[1].plot(history.history[f"val_{accuracy_key}"], label="Valid")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.savefig(output_path)
    plt.close()
