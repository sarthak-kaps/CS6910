import tensorflow as tf

import unicodedata
import os
import io
import numpy as np

np.random.seed(100)


def preprocess_sentence(w):
    w = " ".join(list(w))
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples, shuffle=True):
    lines = io.open(path, encoding='UTF-8').read().strip().split("\n")
    if(shuffle):
        np.random.shuffle(lines)
    word_pairs = [[preprocess_sentence(w) for w in line.split('\t')][:2]
                  for line in lines[:num_examples]]

    return zip(*word_pairs)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None, shuffle=True):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print(f'{t} ----> {lang.index_word[t]}')


def load_tensors(dataset_type, num_examples, shuffle=True):

    path_to_file = f"../dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.{dataset_type}.tsv"

    # Try experimenting with the size of that dataset
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
        path_to_file, num_examples, shuffle)

    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_targ, max_length_inp
