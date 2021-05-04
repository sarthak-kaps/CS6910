import numpy as np
from tensorflow.python.ops.gen_array_ops import edit_distance
from tqdm import tqdm
from wandb.keras import WandbCallback

import encode_input
import model_maker
import beam_search
import wandb
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
val_samples = 100
train_samples = 1000000


# Wandb default config
config_defaults = {
    "epochs": 3,
    "batch_size": 128,
    "layer_dimensions": [128, 128],
    "cell_type": "LSTM",
    "dropout": 0.1,
    "recurrent_dropout": 0.1,
    "optimizer": "adam",
    "beam_width": 3,
    "attention": False,
    "attention_shape": 256
}

# Initialize the project
wandb.init(project='assignment3',
           group='Without attention 1',
           config=config_defaults)

# config file used for the current run
config = wandb.config


wandb.run.name = f"cell_type_{config.cell_type}_layer_org_{config.layer_dimensions}_drpout_{int(config.dropout*100)}%_rec-drpout_{int(config.recurrent_dropout*100)}%_bs_{config.batch_size}_opt_{config.optimizer}_beam_{config.beam_width}"


base_data_set_name = "../dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled."

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
model, inf_enc_model, inf_dec_model = model_maker.make_model(
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

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in data_encoder.input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in data_encoder.target_token_index.items())


def decode_sequence(input_seq, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value, enc_out = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, data_encoder.num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, data_encoder.target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    # Creating a list then using "".join() is usually much faster for string creation
    decoded_sentence = ""
    while not stop_condition:
        to_split = decoder_model.predict(
            [target_seq, states_value, enc_out])
        output_tokens, states_value, attn_weights = to_split[0], list(
            to_split[1:-1]), to_split[-1]

        # Sample a token
#         print(output_tokens)
        sampled_token_index = np.argmax(output_tokens[0, 0])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += (sampled_char)

        # Exit conition: either hit max length
        # or find stop character.
        if sampled_char == '\n' or len(decoded_sentence) > data_encoder.max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, data_encoder.num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

    return decoded_sentence


def editDistance(str1, str2, m, n):
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):

            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace

    return dp[m][n]


input_seqs = encoder_input_data["valid"]
target_sents = target_texts_dict["valid"]
input_texts = input_texts_dict["valid"]
n = len(input_seqs)
val_avg_edit_dist = 0

if config.beam_width > 1:
    # make the beam search object
    bs = beam_search.BeamSearch(
        config.beam_width, data_encoder.max_decoder_seq_length, data_encoder.target_token_index)
    val_avg_edit_dist = 0
    log_table = []
    val_acc = 0
    for seq_index in tqdm(range(val_samples)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = input_seqs[seq_index:seq_index+1]
        decoded_sentence = bs.apply(inf_enc_model, inf_dec_model, input_seq)
        target_sentence = str(target_sents[seq_index:seq_index+1][0][1:-1])
    #     print(input_texts_dict["valid"][seq_index], target_sentence)
        decoded_sentence = "".join(decoded_sentence[0].characters[1:-1])

        edit_dist = editDistance(decoded_sentence, target_sentence, len(
            decoded_sentence), len(target_sentence))/len(target_sentence)
        val_avg_edit_dist += edit_dist
        if(seq_index < 20):
            log_table.append(
                [input_texts[seq_index], decoded_sentence, target_sentence, edit_dist])
            print({f"input_{seq_index}": input_texts[seq_index], f"output_{seq_index}": decoded_sentence,
                   f"target_{seq_index}": target_sentence, f"edit_distance_{seq_index}": edit_dist})

    val_avg_edit_dist /= val_samples
    val_acc /= val_samples
    wandb.log({"val_avg_edit_dist": val_avg_edit_dist, "val_avg_acc": val_acc})

    wandb.log({"Validation log table": wandb.Table(data=log_table,
                                                   columns=["Input", "Prediction", "Target", "Edit-dist"])})


else:
    val_avg_edit_dist = 0
    log_table = []
    val_acc = 0
    for seq_index in tqdm(range(val_samples)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = input_seqs[seq_index:seq_index+1]
        decoded_sentence = str(decode_sequence(
            input_seq, inf_enc_model, inf_dec_model)[:-1])
        target_sentence = str(target_sents[seq_index:seq_index+1][0][1:-1])
        edit_dist = editDistance(decoded_sentence, target_sentence, len(
            decoded_sentence), len(target_sentence))/len(target_sentence)
        val_avg_edit_dist += edit_dist
        if(seq_index < 20):
            log_table.append(
                [input_texts[seq_index], decoded_sentence, target_sentence, edit_dist])
            print({f"input_{seq_index}": input_texts[seq_index], f"output_{seq_index}": decoded_sentence,
                   f"target_{seq_index}": target_sentence, f"edit_distance_{seq_index}": edit_dist})
    wandb.log({"Validation log table": wandb.Table(data=log_table,
                                                   columns=["Input", "Prediction", "Target", "Edit-dist"])})

    val_avg_edit_dist /= val_samples
    val_acc /= val_samples
    wandb.log({"val_avg_edit_dist": val_avg_edit_dist, "val_avg_acc": val_acc})
