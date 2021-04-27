import tensorflow as tf
from tensorflow import keras

def make_cell(config, **kwargs) :
    if config.cell_type == "LSTM" :
        return tf.keras.layers.LSTM(**kwargs, dropout = config.dropout, recurrent_dropout = config.recurrent_dropout)
    elif config.cell_type == "GRU" :
        return tf.keras.layers.GRU(**kwargs, dropout = config.dropout, recurrent_dropout = config.recurrent_dropout)
    elif config.cell_type == "RNN" :
        return tf.keras.layers.RNN(**kwargs, dropout = config.dropout, recurrent_dropout = config.recurrent_dropout)
    else :
        raise ValueError("Invalid cell type {}", config.cell_type)
    

def make_model(config, input_shape_encoder, input_shape_decoder) :
    encoder_inputs = tf.keras.Input(shape = (None, input_shape_encoder))

    encoder_outputs = encoder_inputs
    encoder_states = []
    for i in range(0, len(config.layer_dimensions)) :
        # if it is not the last LSTM unit we need to set return_sequence = true
        if i < len(config.layer_dimensions) - 1:
            try :
                encoder_outputs, h, c = make_cell(config, units = config.layer_dimensions[i], return_state = True, return_sequences = True)(encoder_outputs)
                states = [h, c]
            except :
                encoder_outputs, c = make_cell(config, units = config.layer_dimensions[i], return_state = True, return_sequences = True)(encoder_outputs)
                states = [c]
        else :
            try :
                encoder_outputs, h, c = make_cell(config, units = config.layer_dimensions[i], return_state = True)(encoder_outputs)
                states = [h, c]
            except :
                encoder_outputs, c = make_cell(config, units = config.layer_dimensions[i], return_state = True)(encoder_outputs)
                states = [c]

        encoder_states.extend(list(states))
    
    decoder_inputs = tf.keras.Input(shape = (None, input_shape_decoder))

    decoder_outputs = decoder_inputs
    for i in range(0, len(config.layer_dimensions)) :
        factor = 1
        if config.cell_type == "LSTM" :
            factor *= 2
        
        layer = make_cell(config, units = config.layer_dimensions[i], return_state = True, return_sequences = True) 
        try :
            decoder_outputs, _, _ = layer(decoder_outputs, initial_state = encoder_states[factor * i : factor * (i + 1)])
        except :
            decoder_outputs, _ = layer(decoder_outputs, initial_state = encoder_states[factor * i : factor * (i + 1)])

    dropout_layer = tf.keras.layers.Dropout(config.dropout)
    
    decoder_outputs = dropout_layer(decoder_outputs)

    decoder_dense = tf.keras.layers.Dense(input_shape_decoder, activation = 'softmax')

    decoder_outputs = decoder_dense(decoder_outputs)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

