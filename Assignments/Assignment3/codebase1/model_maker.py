from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout,
                                     SimpleRNN, TimeDistributed, Layer)
from tensorflow import expand_dims, reduce_sum, multiply, concat, split, nn, reduce_mean
from tensorflow import print as pr
from numpy import zeros


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):

        if(len(query.shape) == 2):
            # values : (batch_size, max_len, hidden_size)
            # query : (batch_size, hidden_size)
            query = expand_dims(query, 1)
            s1 = self.W1(query)
            # s1 : (batch_size, 1, units)
            # s2 : (batch_size, max_len, units)
            s2 = self.W2(values)
            # score : (batch_size, max_len, 1)
            score = self.V(nn.tanh(s1 + s2))
            # attention_weights : (batch_size, max_len, 1)
            attention_weights = nn.softmax(score, axis=1)
            # context_vector : (batch_size, max_len, hidden_size)
            context_vector = attention_weights * values
            # context_vector : (batch_size, hidden_size)
            context_vector = reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights
        else:
            # query : (batch_size, max_len2, hidden_size)
            # values : (batch_size, max_len1, hidden_size)
            query = expand_dims(query, 2)
            values = expand_dims(values, 1)

            s1 = self.W1(query)
            s2 = self.W2(values)
            # s1 : (batch_size, max_len2,1, units)
            # s2 : (batch_size, 1,max_len1, units)
            tsum = s1+s2
            # tsum : (batch_size, max_len2, max_len1, units)
            score = self.V(nn.tanh(tsum))
            # score : (batch_size, max_len2, max_len1, 1)
            attention_weights = nn.softmax(score, axis=2)
           # attention_weights : (batch_size, max_len2, max_len1,1)
            # values : (batch_size, 1,max_len1, hidden_size)
            context_vector = attention_weights * values
            # context_vector : (batch_size, max_len2, max_len1, hidden_size)
            context_vector = reduce_sum(context_vector, axis=2)
            # context_vector : (batch_Size, max_len2, hidden_size)
            return context_vector, attention_weights


def make_model(config, enc_timesteps, enc_vsize, dec_timesteps, dec_vsize):
    # Take inputs for enc and decoder
    enc_inputs = Input(shape=(enc_timesteps, enc_vsize),
                       name="Train_encoder_input")
    dec_inputs = Input(shape=(dec_timesteps, dec_vsize),
                       name="Train_decoder_input")

    # Imp variables
    nlayers = len(config.layer_dimensions)
    latent_dims = config.layer_dimensions
    dispatch_dict = {"LSTM": LSTM, "GRU": GRU, "RNN": SimpleRNN}
    factor = (2**(config.cell_type == "LSTM"))

    # Create enc layers
    enc_layers, enc_states = [], []
    prev_layer_out = enc_inputs
    for i in range(nlayers):
        # We don't need all the timesteps for last layer.
        return_sequences = True
        enc_lstm = dispatch_dict[config.cell_type](
            latent_dims[i], return_state=True, recurrent_dropout=config.recurrent_dropout, dropout=config.dropout, return_sequences=return_sequences, name=f"train_encoder_{config.cell_type}_{i+1}")
        enc_lstm_out = enc_lstm(prev_layer_out)
        enc_lstm_states, enc_lstm_out = concat(
            list(enc_lstm_out[1:]), axis=-1), enc_lstm_out[0]
        prev_layer_out = enc_lstm_out
        enc_layers.append(enc_lstm)
        enc_states.append(enc_lstm_states)

    # Add attention layer
    # Attention layer will take last encoder layer output and decoder input as input
    # We will combine the attention context with decoder input
    # Create decoder layers
    dec_layers, dec_outs, dec_states = [], [], []
    prev_layer_out = dec_inputs
    for i in range(nlayers):
        # Here, we want all the layers to output sequence, as we are using it in softmax
        dec_lstm = dispatch_dict[config.cell_type](
            latent_dims[i], return_state=True, return_sequences=True, recurrent_dropout=config.recurrent_dropout, dropout=config.dropout, name=f"train_decoder_{config.cell_type}_{i+1}")
        dec_lstm_out = dec_lstm(
            prev_layer_out, initial_state=split(enc_states[i], factor, -1))
        dec_lstm_states, dec_lstm_out = list(dec_lstm_out[1:]), dec_lstm_out[0]
        prev_layer_out = dec_lstm_out
        dec_layers.append(dec_lstm)
        dec_outs.append(dec_lstm_out)
        dec_states.append(dec_lstm_states)

    attn = BahdanauAttention(config.attention_shape)
    attn_context, attn_weights = attn(
        prev_layer_out, enc_lstm_out)
    if(config.attention):
        prev_layer_out = multiply(
            attn_context, prev_layer_out)

    # Add dropout layer
    dec_dropout = Dropout(config.dropout, name="dropout")
    dec_dropout_out = dec_dropout(prev_layer_out)

    # Add softmax layer.
    # Note: the timedistributed layer, this applies softmax for each timestep
    dec_dense = Dense(dec_vsize, activation="softmax",
                      name="train_softmax_dense")
    dec_time_distri = TimeDistributed(dec_dense, name="train_timedist_softmax")
    dec_output = dec_time_distri(dec_dropout_out)

    # Final model is ready
    full_model = Model([enc_inputs, dec_inputs],
                       dec_output, name="Train_model")

    # We can use the above layers to create the model for inference as well

    # Inference encoder model

    # Take input
    inf_enc_inputs = Input(shape=(enc_timesteps, enc_vsize),
                           name="Inference_encoder_inputs")

    # Use the encoder layers and collect final states
    inf_enc_states = []
    prev_layer_out = inf_enc_inputs
    for i in range(nlayers):
        inf_enc_lstm_out = enc_layers[i](prev_layer_out)
        # Note: we are concating states for ease, as LSTM outputs 2 states at once.
        inf_enc_lstm_states, inf_enc_lstm_out = concat(
            list(inf_enc_lstm_out[1:]), axis=-1), inf_enc_lstm_out[0]
        inf_enc_states.append(inf_enc_lstm_states)
        prev_layer_out = inf_enc_lstm_out

    # Inference encoder model is ready
    inf_enc_model = Model(
        inf_enc_inputs, [inf_enc_states, prev_layer_out], name="Inference_encoder_model")

    # Inference Decoder model

    # Take inputs
    # Note the 1 here, as we will pass 1 char after another for inference
    inf_dec_inputs = Input(shape=(1, dec_vsize),
                           name="Inference_decoder_inputs")
    inf_dec_init_states = [Input(shape=(factor*latent_dims[i],), name=f"Inference_decoder_init_states_{i+1}")
                           for i in range(nlayers)]
    inf_enc_out = Input(
        shape=(enc_timesteps, latent_dims[nlayers-1]), name="Inference_encoder_output")

    # Use the decoder layers from training and collect their states
    # These states will be passed for each char separately
    prev_layer_out = inf_dec_inputs
    inf_dec_states = []
    for i in range(nlayers):
        inf_dec_lstm_out = dec_layers[i](
            prev_layer_out, initial_state=split(inf_dec_init_states[i], factor, axis=-1))
        inf_dec_lstm_states, inf_dec_lstm_out = concat(list(
            inf_dec_lstm_out[1:]), axis=-1), inf_dec_lstm_out[0]
        prev_layer_out = inf_dec_lstm_out
        inf_dec_states.append(inf_dec_lstm_states)

    # Use the attention layer to get the context
    # prev_layer_out = reduce_sum(prev_layer_out, axis=1)
    attn_context, attn_weights = attn(
        prev_layer_out, inf_enc_out)
    if(config.attention):
        prev_layer_out = multiply(
            attn_context, prev_layer_out)

    # Add dropout
    inf_dec_dropout = dec_dropout(prev_layer_out)
    inf_dec_output = TimeDistributed(dec_dense)(
        inf_dec_dropout)

    # Inference decoder model is ready
    inf_dec_model = Model([inf_dec_inputs,
                           inf_dec_init_states, inf_enc_out], [inf_dec_output, inf_dec_states, attn_weights], name="Inference_decoder")

    return full_model, inf_enc_model, inf_dec_model


if __name__ == "__main__":
    class default_config:
        cell_type = "LSTM"
        layer_dimensions = [128]
        attention = True
        attention_shape = 64
        dropout = 0.1
        recurrent_dropout = 0.1

    full_model, inf_enc_model, inf_dec_model = make_model(
        default_config(), 20, 30, 21, 31)
    # print(full_model.summary())
    # print(inf_enc_model.summary())
    # print(inf_dec_model.summary()) 
