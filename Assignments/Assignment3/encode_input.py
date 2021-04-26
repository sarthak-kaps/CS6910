import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# convert files into csv for better readability


def clean_files():
    base_file_name = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled."
    for filename_extension in ["train", "test", "dev"]:
        f = open("temp", mode='wt', encoding='utf-8')
        f.write("Hi\tEn\tLexicons\n")

        filename = base_file_name + filename_extension + ".tsv"
        data_file = open(filename, mode='rt', encoding='utf-8')
        for line in data_file.readlines():
            f.write(line)

        data = pd.read_table("temp")
        data.to_csv(filename_extension + ".csv")

# Vectorize the data.


def vectorize_data(data_path):

    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
        for line in lines[:len(lines) - 1]:
            input_text, target_text, _ = line.split("\t")
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)

    input_token_index = dict([(char, i)
                              for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i)
                               for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0
    return encoder_input_data, decoder_input_data, decoder_target_data

# vectorize_data(
#     data_path="dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")
