"""Simple script to compare two models.

Ideally you would make this much more involved, but for now we are
just going to do two things:
- Print a table showing the accuracy for each model
- Plot the accuracy curves for each model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


from src.data import load_jets_data
from src.utils import get_evaluation_args

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def tranform_to_labels(out, y_test, multi_class) -> tuple[np.ndarray, np.ndarray]:
    if multi_class:
        return np.argmax(out, axis=1), y_test
    return (out > 0.5), y_test

def create_confusion_matrix(out, y_test, multi_class, model_name):
    # Create the confusion matrix for the unweighted model
    labels = [0, 1] if not multi_class else [0, 1, 2]
    cm = confusion_matrix(y_test, out, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Display the confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("confusion_matrix")
    fig.savefig(f"{args.model_dir}/{model_name}/test_confusion_matrix.eps")
    fig.savefig(f"{args.model_dir}/{model_name}/test_confusion_matrix.png")


# Gather the arguments
args = get_evaluation_args()

# For each model load the args used during training
model_args = []
for model_name in args.model_names:
    file_name = f"{args.model_dir}/{model_name}/args.yaml"
    with open(file_name) as f:
        model_args.append(yaml.safe_load(f))

# Pull out the important args
important_args = {
    a: [model_args[i][a] for i in range(len(args.model_names))] for a in args.important_args
}

# We dont actually need to reload the whole model, just its test outputs
accuracies = []
for model_name in args.model_names:
    out = np.load(f"{args.model_dir}/{model_name}/test_out.npy")
    y_test = np.load(f"{args.model_dir}/{model_name}/test_y.npy")
    ypred_labels, y_test = tranform_to_labels(out, y_test, args.multi_class)
    acc = np.mean(ypred_labels == y_test)
    accuracies.append(acc)
    create_confusion_matrix(ypred_labels, y_test, args.multi_class, model_name)


# Create a table showing: model name, important args, and accuracy
df = pd.DataFrame()
df["Model"] = args.model_names
for k, v in important_args.items():
    df[k] = v
df["Test Accuracy"] = accuracies
df.to_markdown("plots/accuracy.md", index=False)

val_accuracy = "val_accuracy" if not args.multi_class else "val_sparse_categorical_accuracy"

# Plot the validation accuracy curves
fig, ax = plt.subplots()
for model_name in args.model_names:
    history = pd.read_csv(f"{args.model_dir}/{model_name}/history.csv")

    if val_accuracy not in history:
        print(f"Skipping {model_name} as no validation accuracy")
        continue
    
    ax.plot(history[val_accuracy], label=model_name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig("plots/val_accuracy.png")
plt.close()

