import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import encode_input
import model_maker
import model_maker_inference
import numpy as np

# Wandb default config
config_defaults = {
    "epochs": 5,
    "batch_size": None,
    "layer_dimensions": [128],
    "cell_type": "LSTM",
    "dropout": 0.1,
    "recurrent_dropout": 0.1,
    "embedding_size": 16,
    "optimizer": "adam",
    "attention": False,
}

# Initialize the project
wandb.init(project='assignment3',
           group='First Run',
           config=config_defaults)

# config file used for the current run
config = wandb.config


wandb.run.name = f"cell_type_{config.cell_type}_layer_org_{config.layer_dimensions}_embd_size_{config.embedding_size}_drpout_{config.dropout}_rec-drpout_{config.recurrent_dropout}_bs_{config.batch_size}_opt_{config.optimizer}"


base_data_set_name = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled."

data_encoder = encode_input.one_hot_encoder(
    [base_data_set_name + "train.tsv", base_data_set_name + "dev.tsv", base_data_set_name + "test.tsv"], ["train", "valid", "test"])


# encoder_input_data, decoder_input_data, decoder_target_data are dictionaries and contain the encoder input
# decoder input and the target data all in one hot encoding format for the train, valid and test datasets.
# Example -
# To obtain the encoder input for train data do encoder_input_data["train"]
# To obtain the decoder input for validation data do decoder_input_data["valid"]
[encoder_input_data, decoder_input_data,
    decoder_target_data, input_texts_dict, target_texts_dict] = data_encoder.vectorize_data()

# construct the model from the given configurations
model = model_maker.make_model(
    config, data_encoder.max_encoder_seq_length, data_encoder.num_encoder_tokens, data_encoder.max_decoder_seq_length, data_encoder.num_decoder_tokens)


# compile the model
model.compile(
    optimizer=config.optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# fit the model
model.fit(
    x=[encoder_input_data["train"], decoder_input_data["train"]],
    y=decoder_target_data["train"],
    batch_size=config.batch_size,
    epochs=config.epochs,
    # validation_data=([encoder_input_data["valid"],
    #                   decoder_input_data["valid"]], decoder_target_data["valid"]),
    callbacks=[WandbCallback()]
)

# Save model
model.save(wandb.run.name)

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


for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data["valid"][seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model)
    print("-")
    print("Input sentence:",
          input_texts_dict["valid"][seq_index: seq_index + 1])
    print("Decoded sentence:", decoded_sentence)
