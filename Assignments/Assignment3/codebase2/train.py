import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
import wandb
import io
from tqdm import tqdm

from data_process import load_tensors, preprocess_sentence, create_dataset
from model_create import BahdanauAttention, Decoder, Encoder
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Source Han Sans TW',
                                   'sans-serif',
                                   'Lohit Devanagari'  # fc-list :lang=hi family
                                   ]


config_defaults = {
    "epochs": 10,
    "batch_size": 128,
    "layer_dimensions": [128],
    "cell_type": "LSTM",
    "dropout": 0.1,
    "recurrent_dropout": 0.1,
    "optimizer": "adam",
    "attention": False,
    "attention_shape": 16,
    "embedding_dim": 64
}

# Initialize the project
wandb.init(project='assignment3',
           group='trial with attention',
           config=config_defaults)


config = wandb.config


# Load dataset
input_tensor, target_tensor, inp_lang, targ_lang, max_length_targ, max_length_inp = load_tensors(
    "train", 50000)

# Set variables
BUFFER_SIZE = len(input_tensor)
BATCH_SIZE = config.batch_size
steps_per_epoch = len(input_tensor)//BATCH_SIZE
embedding_dim = config.embedding_dim
units = config.layer_dimensions[0]
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
VAL_SAMPLES = 1000

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# Create model
encoder = Encoder(vocab_inp_size, config)
decoder = Decoder(vocab_tar_size, config)

# Set loss and optimizer
optimizer = tf.keras.optimizers.Adam(0.1)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder

            output = decoder(
                dec_input, dec_hidden, enc_output)
            predictions, dec_hidden = output[0], output[1]
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


# Training loop
EPOCHS = config.epochs
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0.0
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss
        if batch % 100 == 0:
            print(
                f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

    wandb.log({"loss": total_loss, "epoch": epoch, "time": time.time()-start})
    print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    enc_out, enc_hidden = encoder(inputs)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        if(config.attention):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()
        else:
            predictions, dec_hidden = decoder(dec_input,
                                              dec_hidden,
                                              enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            if(config.attention):
                return result, sentence, attention_plot
            else:
                return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    if(config.attention):
        return result, sentence, attention_plot
    else:
        return result, sentence

# function for plotting the attention weights


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence,
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


# Validation Loop


valid_file_name = "../dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"

# Validation Loop
lines = io.open(valid_file_name,
                encoding='UTF-8').read().strip().split("\n")

np.random.shuffle(lines)
log_table = []

num_log = 20
num_attn_log = 1

val_edit_dist = 0.0
val_acc = 0
for i, line in enumerate(lines[:VAL_SAMPLES]):
    targ, inp = line.split("\t")[:2]
    if config.attention:
        result, sentence, attention_plot = evaluate(inp)
    else:
        result, sentence = evaluate(inp)
    predicted_sent = "".join(result.split(' ')[:-2])
    edit_dist = editDistance(predicted_sent, targ,
                             len(predicted_sent), len(targ))/len(targ)
    val_edit_dist += edit_dist
    if(targ == predicted_sent):
        val_acc += 1
    if (i < num_log):
        log_table.append([inp, predicted_sent, targ, edit_dist])
        print(
            f'Input: {inp}, Prediction: {predicted_sent}, Target:{targ}, Edit-dist: {edit_dist}')

    if ((config.attention) and (i < num_attn_log)):
        attention_plot = attention_plot[:len(result.split(' ')),
                                        :len(sentence.split(' '))]
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))

val_edit_dist /= VAL_SAMPLES
val_acc /= VAL_SAMPLES

wandb.log({"Validation log table": wandb.Table(data=log_table,
                                               columns=["Input", "Prediction", "Target", "Edit-dist"])})
wandb.log({"val_avg_edit_dist": val_edit_dist, "val_avg_acc": val_acc})
