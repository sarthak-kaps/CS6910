from tensorflow.keras.layers import LSTM, GRU, RNN, Dense, Concatenate, TimeDistributed, Attention
from tensorflow.keras import Input, Model
from tensorflow.python.keras.backend import dropout


def make_cell(config, **kwargs):
    dispatch_dict = {"LSTM": LSTM, "GRU": GRU, "RNN": RNN}
    layer = dispatch_dict.get(config.cell_type)
    if(layer):
        return layer(**kwargs, dropout=config.dropout, return_state=True, recurrent_dropout=config.recurrent_dropout)
    else:
        raise ValueError("Invalid cell type {}", config.cell_type)


def make_model(config, input_shape_encoder, input_shape_decoder):

    encoder_inp = Input(shape=(None, input_shape_encoder))
    prev_layer_out = encoder_inp
    encoder_states = []
    for i in range(0, len(config.layer_dimensions)):
        return_seqs = (i < len(config.layer_dimensions) - 1)
        layer_output = make_cell(
            config, units=config.layer_dimensions[i],  return_sequences=return_seqs)(prev_layer_out)
        prev_layer_out = layer_output[0]
        states = list(layer_output[1:])
        encoder_states.extend(list(states))

    encoder_out = prev_layer_out
    decoder_inp = Input(shape=(None, input_shape_decoder))
    prev_layer_out = decoder_inp
    for i in range(0, len(config.layer_dimensions)):
        # Why do we always start with factor = 1?
        factor = (2**(config.cell_type == "LSTM"))
        layer = make_cell(
            config, units=config.layer_dimensions[i],  return_sequences=True)
        layer_output = layer(
            prev_layer_out, initial_state=encoder_states[factor * i: factor * (i + 1)])
        prev_layer_out = layer_output[0]

    print(encoder_out.shape, prev_layer_out.shape)

    if(config.attention):
        attn_layer = Attention(
            input_shape_encoder)
        attn_out,  attn_states = attn_layer([encoder_out, prev_layer_out])
        prev_layer_out = Concatenate(axis=-1)([prev_layer_out, attn_out])

    # Why 2 adding dropout here and below too?
    # prev_layer_out = Dropout(config.dropout)(prev_layer_out)

    fc_layer = Dense(input_shape_decoder, activation='softmax')
    softmax_time = TimeDistributed(fc_layer)
    final_output = softmax_time(prev_layer_out)

    model = Model(
        [encoder_inp, decoder_inp], final_output)

    return model


# class default_config:
#     cell_type = "LSTM"
#     layer_dimensions = [10]
#     attention = True
#     dropout = 0.1
#     recurrent_dropout = 0.1


# default_config = {
#     "cell_type": "LSTM",
#     "layer_dimensions": [10],
#     "attention": False,
#     "dropout": 0.1,
#     "recurrent_dropout": 0.1,
# }

# model = make_model(default_config, (100, 200), (100, 200))
