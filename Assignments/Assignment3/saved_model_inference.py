import numpy as np
from tensorflow import keras
from tqdm import tqdm
from tensorflow import reshape
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import encode_input
import wandb


base_data_set_name = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled."

data_encoder = encode_input.one_hot_encoder(
    [base_data_set_name + "train.tsv", base_data_set_name + "dev.tsv", base_data_set_name + "test.tsv"], ["train", "valid", "test"])

[encoder_input_data, decoder_input_data,
    decoder_target_data, input_texts_dict, target_texts_dict] = data_encoder.vectorize_data()


model = keras.models.load_model("model2/train")
inf_enc_model = keras.models.load_model("model2/inf-enc")
inf_dec_model = keras.models.load_model("model2/inf-dec")


# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in data_encoder.input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in data_encoder.target_token_index.items())


def decode_sequence(input_seq, encoder_model, decoder_model):
    attention_plot = np.zeros((input_seq.shape[1], input_seq.shape[1]))
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
    decoded_sentence = []
    i = 0
    while not stop_condition:
        to_split = decoder_model.predict(
            [target_seq, states_value, enc_out])
        output_tokens, states_value, attn_weights = to_split[0], list(
            to_split[1:-1]), to_split[-1]

        attention_weights = reshape(attn_weights, (-1, ))
        attention_plot[i] = attention_weights.numpy()

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
        i += 1

    return "".join(decoded_sentence), attention_plot


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

# function for plotting the attention weights


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


input_seqs = encoder_input_data["valid"]
target_sents = target_texts_dict["valid"]
n = len(input_seqs)
val_avg_edit_dist = 0
for seq_index in tqdm(range(500)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = input_seqs[seq_index:seq_index+1]
    decoded_sentence, attention_plot = str(decode_sequence(
        input_seq, inf_enc_model, inf_dec_model)[:-1])
    target_sentence = str(target_sents[seq_index:seq_index+1][0][1:-1])
    edit_dist = editDistance(decoded_sentence, target_sentence, len(
        decoded_sentence), len(target_sentence))/len(target_sentence)
    val_avg_edit_dist += edit_dist
    if(seq_index < 5):
        plot_attention(attention_plot, target_sentence, decoded_sentence)
    if(seq_index < 20):
        wandb.log({f"input_{seq_index}": input_seq, f"output_{seq_index}": decoded_sentence,
                   f"target_{seq_index}": target_sentence, f"edit_distance_{seq_index}": edit_dist})

val_avg_edit_dist /= 500

wandb.log({"val_avg_edit_dist": val_avg_edit_dist})
