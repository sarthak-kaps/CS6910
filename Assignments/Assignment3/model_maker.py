from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (GRU, LSTM, Concatenate, Dense, Dropout,
                                     SimpleRNN, TimeDistributed)
from attention import AttentionLayer


def make_cell(config, **kwargs):
    dispatch_dict = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}
    layer = dispatch_dict.get(config.cell_type)
    if(layer):
        return layer(**kwargs, dropout=config.dropout, return_state=True, return_sequences=True, recurrent_dropout=config.recurrent_dropout)
    else:
        raise ValueError("Invalid cell type {}", config.cell_type)


def make_model(config, enc_timesteps, enc_vsize, dec_timesteps, dec_vsize):
    encoder_inp = Input(
        shape=(None, enc_vsize), name='encoder_inputs')
    # if dec_timesteps:
    #     decoder_inp = Input(
    #         shape=(dec_timesteps, dec_vsize), name='decoder_inputs')
    # else:
    decoder_inp = Input(
        shape=(None, dec_vsize), name='decoder_inputs')

    prev_layer_out = encoder_inp
    encoder_states = []
    for i in range(0, len(config.layer_dimensions)):
        layer_output = make_cell(
            config, units=config.layer_dimensions[i])(prev_layer_out)
        prev_layer_out = layer_output[0]
        states = list(layer_output[1:])
        encoder_states.extend(list(states))

    encoder_out = prev_layer_out
    prev_layer_out = decoder_inp
    for i in range(0, len(config.layer_dimensions)):
        factor = (2**(config.cell_type == "LSTM"))
        layer = make_cell(
            config, units=config.layer_dimensions[i])
        layer_output = layer(
            prev_layer_out, initial_state=encoder_states[factor * i: factor * (i + 1)])
        prev_layer_out = layer_output[0]

    if(config.attention):
        attn_layer = AttentionLayer()
        attn_out,  attn_states = attn_layer([encoder_out, prev_layer_out])
        prev_layer_out = Concatenate(axis=-1)([prev_layer_out, attn_out])

    # Why 2 adding dropout here and below too?
    prev_layer_out = Dropout(config.dropout)(prev_layer_out)

    fc_layer = Dense(dec_vsize, activation='softmax')
    softmax_time = TimeDistributed(fc_layer)
    final_output = softmax_time(prev_layer_out)

    model = Model(
        [encoder_inp, decoder_inp], final_output)

    return model


if __name__ == "__main__":
    class default_config:
        cell_type = "LSTM"
        layer_dimensions = [10]
        attention = True
        dropout = 0.1
        recurrent_dropout = 0.1

    model = make_model(default_config(), 20, 30, 20, 30)
    model.summary()
# model = make_model(default_config, (100, 200), (100, 200))
