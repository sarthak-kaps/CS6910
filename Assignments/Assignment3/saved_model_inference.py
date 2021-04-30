import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import encode_input
import model_maker
import model_maker_inference
import numpy as np
from tqdm import tqdm


class default_config:
    cell_type = "LSTM"
    layer_dimensions = [128, 64]
    attention = False
    dropout = 0.1
    recurrent_dropout = 0.1
    optimizer = "adam"
    batch_size = 128
    epochs = 10


config = default_config()


base_data_set_name = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled."

data_encoder = encode_input.one_hot_encoder(
    [base_data_set_name + "train.tsv", base_data_set_name + "dev.tsv", base_data_set_name + "test.tsv"], ["train", "valid", "test"])

[encoder_input_data, decoder_input_data,
    decoder_target_data, input_texts_dict, target_texts_dict] = data_encoder.vectorize_data()


model = keras.models.load_model("model2")


encoder_model, decoder_model = model_maker_inference.make_inference_model(
    model, config)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in data_encoder.input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in data_encoder.target_token_index.items())


def decode_sequence(input_seq, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, data_encoder.num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, data_encoder.target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    # Creating a list then using "".join() is usually much faster for string creation
    decoded_sentence = []
    while not stop_condition:
        to_split = decoder_model.predict([target_seq] + states_value)

        output_tokens, states_value = to_split[0], to_split[1:]

        # Sample a token
#         print(output_tokens)
        sampled_token_index = np.argmax(output_tokens[0, 0])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == '\n' or len(decoded_sentence) > data_encoder.max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, data_encoder.num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

    return "".join(decoded_sentence)


accuracy = 0
input_seqs = encoder_input_data["valid"]
target_sents = target_texts_dict["valid"]
for seq_index in tqdm(range(len(encoder_input_data["valid"]))):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = input_seqs[seq_index]
    decoded_sentence = decode_sequence(
        input_seq, encoder_model, decoder_model)[:-1]
    target_sentence = target_sents[seq_index][0][1:-1]

    if(decoded_sentence == target_sentence):
        accuracy += 1

    if(seq_index < 20):
        print("-")
        print("Input sentence:",
              input_texts_dict["valid"][seq_index: seq_index + 1])
        print("Decodd sentence:", decoded_sentence)
        print("Target sentence:", target_sentence)

accuracy /= len(encoder_input_data["valid"])
print("Accuracy : ", accuracy)
