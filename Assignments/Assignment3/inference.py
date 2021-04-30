import numpy as np
from tqdm import tqdm
from wandb.keras import WandbCallback

import encode_input
import model_maker
import model_maker_inference
import wandb

# Wandb default config
config_defaults = {
    "epochs": 1,
    "batch_size": 128,
    "layer_dimensions": [16],
    "cell_type": "LSTM",
    "dropout": 0.1,
    "recurrent_dropout": 0.1,
    "optimizer": "adam",
    "attention": False,
}

# Initialize the project
wandb.init(project='assignment3',
           group='First Run',
           config=config_defaults)

# config file used for the current run
config = wandb.config


wandb.run.name = f"cell_type_{config.cell_type}_layer_org_{''.join([str(i) for i in config.layer_dimensions])}_drpout_{int(config.dropout*100)}%_rec-drpout_{int(config.recurrent_dropout*100)}%_bs_{config.batch_size}_opt_{config.optimizer}"


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

print(model.summary())

# compile the model
model.compile(
    optimizer=config.optimizer, loss="categorical_crossentropy",  metrics=['accuracy']
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
        if type(states_value) != list:
            to_split = decoder_model.predict([target_seq]+[states_value])
        else:
            to_split = decoder_model.predict([target_seq]+states_value)
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


input_seqs = encoder_input_data["valid"][:20]
target_sents = target_texts_dict["valid"][:20]

acc = 0
for seq_index in tqdm(range(len(input_seqs))):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = input_seqs[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(
        input_seq, encoder_model, decoder_model)[:-1]
    target_sentence = target_sents[seq_index:seq_index+1][0][1:-1]
    if(seq_index < 20):
        wandb.log({f"input_{seq_index}": input_seq, f"output_{seq_index}": decoded_sentence,
                   f"target_{seq_index}": target_sentence})
    if(str(target_sentence) == str(input_seq)):
        acc += 1

acc /= len(input_seqs)
print("\n", acc)
wandb.log({"val_accuracy": acc})
