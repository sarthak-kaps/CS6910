import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, RNN, Dense, Concatenate, TimeDistributed, Dropout
from tensorflow.python.keras.layers.dense_attention import Attention
from attention import AttentionLayer
from tensorflow.keras import Input, Model
from tensorflow.python.keras.backend import dropout

# takes in a trained model and returns the encoder and decoder parts
def make_inference_model(model, config) :
    
    encoder_inp = model.layers[0].output
    prev_layer_output = encoder_inp
    encoder_states = []
    
    for i in range(0, len(config.layer_dimensions)):
        layer_output = model.layers[2 + 2 * i](prev_layer_output)
        prev_layer_output = layer_output[0]
        states = list(layer_output[1:])
        encoder_states.extend(list(states))
    
    encoder_model = tf.keras.Model(encoder_inp, encoder_states)


    decoder_inp = model.layers[1].output
    prev_layer_out = decoder_inp
    
    output_layers = []
    
    for i in range(0, len(config.layer_dimensions)):
        factor = (2**(config.cell_type == "LSTM"))
        layer = model.layers[3 + 2 * i]
        layer_output = layer(
            prev_layer_out, initial_state=encoder_states[factor * i: factor * (i + 1)])
        prev_layer_out = layer_output[0]
        output_layers.append(layer)
        
    if(config.attention):
        attn_layer = AttentionLayer()
        attn_out,  attn_states = attn_layer([encoder_out, prev_layer_out])
        prev_layer_out = Concatenate(axis=-1)([prev_layer_out, attn_out])


    prev_layer_out = model.layers[6](prev_layer_out)
    final_output = model.layers[7](prev_layer_out)

    latent_dims = config.layer_dimensions
    d_outputs = decoder_inp

    decoder_states_inputs = []
    decoder_states = []

    for i in range(len(latent_dims)):
        current_state_inputs = [tf.keras.Input(shape=(latent_dims[i],)) for _ in range(factor)]

        temp = output_layers[i](d_outputs, initial_state=current_state_inputs)

        d_outputs, cur_states = temp[0], temp[1:]

        decoder_states += cur_states
        decoder_states_inputs += current_state_inputs
    

    d_outputs = model.layers[6](d_outputs)

    decoder_outputs = model.layers[7](d_outputs)

    decoder_model = tf.keras.Model(
        [decoder_inp] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model
