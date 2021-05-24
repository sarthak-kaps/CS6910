import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from matplotlib.font_manager import FontProperties

import wandb


def evaluate(inp, encoder, decoder, BATCH_SIZE, max_length_targ, targ_lang):
    attention_plots = []
    enc_output, enc_hidden = encoder(inp)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims(
        [targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
    batchpreds = []
    for t in range(max_length_targ):
        output = decoder(dec_input, dec_hidden, enc_output)
        predictions, dec_hidden = output[0], output[1]
        attention_plot = tf.reshape(output[2], (inp.shape[0], -1))
        predicted_id = tf.argmax(predictions, axis=1).numpy()
        batchpreds.append(predicted_id)
        attention_plots.append(attention_plot)
        dec_input = tf.expand_dims(predicted_id, 1)
    batchpreds = np.array(batchpreds).T
    attention_plots = np.transpose(np.array(attention_plots), [1, 0, 2])
    batch_results = ["".join([targ_lang.index_word[t] for t in batchpreds[i] if (
        targ_lang.index_word[t] > "<end>")]) for i in range(inp.shape[0])]
    return batch_results, attention_plots


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    hindi_font = FontProperties(
        fname="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf")
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict)
    ax.set_yticklabels([''] + predicted_sentence, fontproperties=hindi_font,
                       fontdict=fontdict)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    wandb.log({"Attention weights": plt})
    plt.show()


def editDistance(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j    # Min. operations = j
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
    return dp[m][n]
