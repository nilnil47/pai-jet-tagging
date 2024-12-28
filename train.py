from pathlib import Path  # This is a nice way to handle file paths in python

import numpy as np
import pandas as pd
import yaml  # This is a nice library for reading and writing yaml files
from tensorflow import keras

# Here we import code from other parts of our project
from src.data import load_jets_data
from src.models import get_model
from src.utils import get_training_args, plot_history

from keras.utils import plot_model


# Gather the arguments, look in the utils.py file to see how this works
args = get_training_args()

# 1) Load and prepare the data
# For neatness we offload all of this to a function in the src/data.py file
(x_train, h_train, y_train), (x_valid, h_valid, y_valid), (x_test, h_test, y_test) = load_jets_data(args.data_size, args.multi_class)


# 2) Define the model
# Here we create a very simple cnn using all of the arguments we defined earlier
# To construct the cnn we use a class defined in the src/models.py file
# Take a look at the package and make sure you understand everything there
model = get_model(
    model_type=args.network_type,
    input_shape=(x_train.shape[1:], h_train.shape[1:]),
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    intermediate_dim=args.intermediate_dim,
    norm_epsilon=args.norm_epsilon,
    dropout=args.dropout,
    multi_class=args.multi_class,
)

# To finalise building the model we must call it with some input
# Note that when doing the summary the output shapes are not defined
# This is because we are not using a fixed input shape
model([x_train[0:1], h_train[0:1]])
model.summary()

# 3) Compile the model
# Compile the model using the arguments we defined earlier
# Since this is a classification task it is nice to track the accuracy

if args.multi_class:
    accuracy_key = "sparse_categorical_accuracy"
    loss = keras.losses.SparseCategoricalCrossentropy()
else:
    accuracy_key = "accuracy"
    loss = keras.losses.BinaryCrossentropy()


model.compile(
    optimizer=keras.optimizers.get(
        {
            "class_name": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
    ),
    loss=loss,
    metrics=[accuracy_key],
)

# We don't just want to train the model once then throw it away, so lets save it
# Each model will get its own directory/folder names after it
output_dir = Path(args.model_dir) / args.model_name
output_dir.mkdir(exist_ok=True, parents=True)  # Create the directory

# It is also incredibly useful to save all of the arguments that went into this
# script so we can reproduce the results later
# We use yaml to save (and later load) the dictionary of arguments
with open(output_dir / "args.yaml", "w") as f:
    yaml.dump(vars(args), f)

plot_model(model, to_file=output_dir / "model.png", show_shapes=True, 
           expand_nested=True, show_trainable=True, show_layer_activations=True)
with open(output_dir / "model.md", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# 4) Train the model
if args.max_epochs == 0:
    # If max_epochs is 0, we are not training the model, just saving it
    model.save(output_dir / "best.keras")

else:
    history = model.fit(
        [x_train, h_train],
        y_train,
        validation_data=([x_valid, h_valid], y_valid),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=args.patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ModelCheckpoint(  # Note the callback here saves the model
                filepath=str(output_dir / "best.keras"),
                save_best_only=True,
            ),
        ],
    )

    # Save the training history with a plot to go along with it
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv")
    plot_history(history, output_dir / "history.png", accuracy_key)

# 5) Predict on the test set
# Instead of evaluating the model on the test set here, we will do it in a separate
# script: `eval.py` This is because we might want to evaluate the model in different
# ways or we might want to evaluate multiple models at once.
# This is also a good way to keep the code clean and modular
model.load_weights(output_dir / "best.keras")  # Load the best version of the model
test_out = model.predict([x_test, h_test])
np.save(output_dir / "test_out.npy", test_out)
np.save(output_dir / "test_y.npy", y_test)
