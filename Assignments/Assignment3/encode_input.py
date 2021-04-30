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


class one_hot_encoder:
    # some encoder parameters
    num_encoder_tokens = 0
    num_decoder_tokens = 0
    max_encoder_seq_length = 0
    max_decoder_seq_length = 0

    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []

    def __init__(self, data_paths, file_labels):
        assert(len(data_paths) == len(file_labels))
        self.data_paths = data_paths
        self.file_labels = file_labels

    # Vectorize the data.
    # Does one hot encoding

    def vectorize_data(self):

        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()

        combined_file = open("encode_data.tsv", mode='wt', encoding='utf-8')

        file_lengths = []

        for filename in self.data_paths:
            with open(filename, mode='rt', encoding='utf-8') as ff:
                lines = ff.readlines()
                file_lengths.append(len(lines))
                for line in lines:
                    combined_file.write(line)

        combined_file.close()

        with open("encode_data.tsv", "r", encoding="utf-8") as f:
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

        input_characters = sorted(list(input_characters)) + ['\n']
        target_characters = sorted(list(target_characters)) + ['\n']
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in target_texts])

        print("Number of samples:", len(input_texts))
        print("Number of unique input tokens:", self.num_encoder_tokens)
        print("Number of unique output tokens:", self.num_decoder_tokens)
        print("Max sequence length for inputs:", self.max_encoder_seq_length)
        print("Max sequence length for outputs:", self.max_decoder_seq_length)

        input_token_index = dict([(char, i)
                                  for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i)
                                   for i, char in enumerate(target_characters)])

        self.encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens), dtype="float32"
        )
        self.decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype="float32"
        )
        self.decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype="float32"
        )

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, input_token_index[char]] = 1.0
                # I think this line will increase error as we are passing the whole word into RNN
                # But, after the \n char, everything will be 0.
                # I think we should make \n default after the word end
            self.encoder_input_data[i, t + 1:, input_token_index["\n"]] = 1.0
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1,
                                             target_token_index[char]] = 1.0
            self.decoder_input_data[i, t + 1:, target_token_index["\n"]] = 1.0
            self.decoder_target_data[i, t:, target_token_index["\n"]] = 1.0

        curr_length = 0
        encoder_input_datas = {}
        decoder_input_datas = {}
        decoder_target_datas = {}
        input_texts_dict = {}
        target_texts_dict = {}

        for (lengths, file_label) in zip(file_lengths, self.file_labels):
            shuffler = np.random.permutation(lengths)
            encoder_input_datas[file_label] = self.encoder_input_data[curr_length: curr_length +
                                                                      lengths, :, :][shuffler]
            decoder_input_datas[file_label] = self.decoder_input_data[curr_length: curr_length +
                                                                      lengths, :, :][shuffler]
            decoder_target_datas[file_label] = self.decoder_target_data[curr_length: curr_length +
                                                                        lengths, :, :][shuffler]
            input_texts_dict[file_label] = np.array(input_texts[curr_length: curr_length +
                                                                lengths])[shuffler]
            target_texts_dict[file_label] = np.array(target_texts[curr_length: curr_length +
                                                                  lengths])[shuffler]
            curr_length += lengths

        # for ease of access later

        self.input_token_index = input_token_index
        self.target_token_index = target_token_index

        return encoder_input_datas, decoder_input_datas, decoder_target_datas, input_texts_dict, target_texts_dict
