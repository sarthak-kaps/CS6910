from tensorflow.keras.layers import LSTM, GRU, RNN, Dense, Concatenate, TimeDistributed, Dropout
from tensorflow.python.keras.layers.dense_attention import Attention
from attention import AttentionLayer
from tensorflow.keras import Input, Model
from tensorflow.python.keras.backend import dropout


def make_cell(config, **kwargs):
    dispatch_dict = {"LSTM": LSTM, "GRU": GRU, "RNN": RNN}
    layer = dispatch_dict.get(config.cell_type)
    if(layer):
        return layer(**kwargs, dropout=config.dropout, return_state=True, return_sequences=True, recurrent_dropout=config.recurrent_dropout)
    else:
        raise ValueError("Invalid cell type {}", config.cell_type)


def make_model(config, enc_timesteps, enc_vsize, dec_timesteps, dec_vsize):
    encoder_inp = Input(
        shape=(enc_timesteps, enc_vsize), name='encoder_inputs')
    if dec_timesteps:
        decoder_inp = Input(
            shape=(dec_timesteps, dec_vsize), name='decoder_inputs')
    else:
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
    output_layers = []
    for i in range(0, len(config.layer_dimensions)):
        factor = (2**(config.cell_type == "LSTM"))
        layer = make_cell(
            config, units=config.layer_dimensions[i])
        layer_output = layer(
            prev_layer_out, initial_state=encoder_states[factor * i: factor * (i + 1)])
        prev_layer_out = layer_output[0]
        output_layers.append(layer)

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

    latent_dims = config.layer_dimensions

    encoder_model = tf.keras.Model(encoder_inp, encoder_states)


    d_outputs = decoder_inp
    
    decoder_states_inputs = []
    decoder_states = []
    
    for i in range(len(latent_dims)):
        current_state_inputs = [tf.keras.Input(shape=(latent_dims[len(latent_dims) - i - 1],)) for _ in range(factor)]

        temp = output_layers[i](d_outputs, initial_state=current_state_inputs)

        d_outputs, cur_states = temp[0], temp[1:]

        decoder_states += cur_states
        decoder_states_inputs += current_state_inputs
   

    d_outputs = Dropout(config.dropout)(d_outputs)
    
    decoder_outputs = softmax_time(d_outputs)
    
    decoder_model = tf.keras.Model(
        [decoder_inp] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)


    return model, encoder_model, decoder_model


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
