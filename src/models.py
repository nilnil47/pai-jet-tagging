import tensorflow as tf
from tensorflow import keras
import keras_nlp.layers

def get_jet_transofrmer_model(
    input_shape: tuple,
    num_heads: int,
    intermediate_dim: int,
    num_layers: int,
    norm_epsilon: float,
    dropout: float,
    multi_class: bool = False,
    **kwargs
):
    inputs_constituents = keras.Input(shape=input_shape[0])
    inputs_high_level = keras.Input(shape=tuple(input_shape[1]))
    output_high_level = keras.layers.LayerNormalization(epsilon=norm_epsilon)(inputs_high_level)
    output_high_level = keras.layers.Dropout(rate=dropout)(output_high_level)

    # Apply layer normalization and dropout to the embedding.
    outputs = keras.layers.LayerNormalization(epsilon=norm_epsilon)(inputs_constituents)
    outputs = keras.layers.Dropout(rate=dropout)(outputs)

    # Add a number of encoder blocks
    for _ in range(num_layers):
        outputs = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm_epsilon=norm_epsilon,
        )(outputs)

    # todo understand what to do with this - for now it is just binary classification
    outputs = keras.layers.GlobalAveragePooling1D()(outputs)
    outputs = keras.layers.Concatenate()([outputs, output_high_level])
    outputs = keras.layers.Dense(intermediate_dim, activation='relu')(outputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(outputs) if not multi_class else keras.layers.Dense(3, activation='softmax')(outputs)

    model = keras.Model([inputs_constituents, inputs_high_level], outputs)
    return model 

def get_jet_mlp_model(
    input_shape: tuple,
    intermediate_dim: int,
    num_layers: int,
    norm_epsilon: float,
    dropout: float,
    multi_class: bool = False,
    **kwargs
):
    inputs = keras.Input(shape=input_shape)
    outputs = keras.layers.Flatten()(inputs)
    outputs = keras.layers.LayerNormalization(epsilon=norm_epsilon)(outputs)
    outputs = keras.layers.Dropout(rate=dropout)(outputs)

    # Add a number of encoder blocks
    for i in range(num_layers):
        outputs = keras.layers.Dense(
            intermediate_dim, activation='relu'
        )(outputs) 

    # for the mlp we dont need a global average pooling, but to make it similar 
    # to the transformer model we will just create a dense that will output vector
    # of size 4
    outputs = keras.layers.Dense(
            input_shape[1], activation='relu'
        )(outputs)
    
    outputs = keras.layers.Dense(1, activation='sigmoid')(outputs) if not multi_class else keras.layers.Dense(3, activation='softmax')(outputs)

    model = keras.Model(inputs, outputs)
    return model 

def get_model(model_type: str, **kwargs):
    if model_type == 'transformer':
        return get_jet_transofrmer_model(**kwargs)
    elif model_type == 'mlp':
        return get_jet_mlp_model(**kwargs)
    else:
        raise ValueError(f"Model type {model_type} not supported")