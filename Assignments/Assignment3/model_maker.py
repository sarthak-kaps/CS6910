import tensorflow as tf
from tensorflow import keras

# TODO : add support for Dropout
def make_model(config, input_shape_encoder, input_shape_decoder) :
    encoder_inputs = tf.keras.Input(shape = (None, input_shape_encoder))

    encoder_outputs = encoder_inputs
    encoder_states = []
    for i in range(0, len(config.layer_dimensions)) :
        # if it is not the last LSTM unit we need to set return_sequence = true
        if i < len(config.layer_dimensions) - 1:
            if config.cell_type == "LSTM" :
                encoder_outputs, h, c = tf.keras.layers.LSTM(config.layer_dimensions[i], return_state = True, return_sequences = True)(encoder_outputs)
            elif config.cell_type == "GRU" :
                encoder_outputs, c = tf.keras.layers.GRU(config.layer_dimensions[i], return_state = True, return_sequences = True)(encoder_outputs)
            elif config.cell_type == "RNN" :
                encoder_outputs, c = tf.keras.layers.RNN(config.layer_dimensions[i], return_state = True, return_sequences = True)(encoder_outputs)
            else :
                raise ValueError("Invalid cell type {}")
        else :
            if config.cell_type == "LSTM" : 
                encoder_outputs, h, c = tf.keras.layers.LSTM(config.layer_dimensions[i], return_state = True)(encoder_outputs)
            elif config.cell_type == "GRU" :
                encoder_outputs, c = tf.keras.layers.GRU(config.layer_dimensions[i], return_state = True)(encoder_outputs)
            elif config.cell_type == "RNN" :
                encoder_outputs, c = tf.keras.layers.RNN(config.layer_dimensions[i], return_state = True)(encoder_outputs)
            else :
                raise ValueError("Invalid cell type {}")

        if config.cell_type == "LSTM" :
            encoder_states += [h, c]
        else :
            encoder_states += [c]

    decoder_inputs = tf.keras.Input(shape = (None, input_shape_decoder))

    decoder_outputs = decoder_inputs
    for i in range(0, len(config.layer_dimensions)) :
        if config.cell_type == "LSTM" :
            layer = tf.keras.layers.LSTM(config.layer_dimensions[i], return_state = True, return_sequences = True)

            decoder_outputs, _, _ = layer(decoder_outputs, initial_state = encoder_states[2 * i : 2 * (i + 1)])
        
        elif config.cell_type == "GRU" :
            layer = tf.keras.layers.GRU(config.layer_dimensions[i], return_state = True, return_sequences = True)

            decoder_outputs, _ = layer(decoder_outputs, initial_state = encoder_states[i : i + 1])
        
        else
            layer = tf.keras.layers.RNN(config.layer_dimensions[i], return_state = True, return_sequences = True)

            decoder_outputs, _ = layer(decoder_outputs, initial_state = encoder_states[i : i + 1])



    decoder_dense = tf.keras.layers.Dense(input_shape_decoder, activation = 'softmax')

    decoder_outputs = decoder_dense(decoder_outputs)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

