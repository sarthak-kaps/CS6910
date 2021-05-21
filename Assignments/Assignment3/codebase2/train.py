import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from matplotlib.font_manager import FontProperties

import wandb
from data_process import load_tensors
from model_create import Decoder, Encoder

config_defaults = {
    "epochs": 5,
    "batch_size": 64,
    "layer_dimensions": [128],
    "cell_type": "GRU",
    "dropout": 0.1,
    "recurrent_dropout": 0,
    "optimizer": "adam",
    "attention": True,
    "attention_shape": 128,
    "embedding_dim": 128
}


# Initialize the project
wandb.init(project='assignment3',
           group='attention_exp1',
           config=config_defaults)


config = wandb.config


wandb.run.name = f"cell_type_{config.cell_type}_layer_org_{'_'.join([str(i) for i in config.layer_dimensions])}_drpout_{int(config.dropout*100)}%_rec-drpout_{int(config.recurrent_dropout*100)}%_bs_{config.batch_size}_opt_{config.optimizer}"


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

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# Create model
encoder = Encoder(vocab_inp_size, config)
decoder = Decoder(vocab_tar_size, config)


# Set loss and optimizer
optimizer = tf.keras.optimizers.Adam()
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


def evaluate(inp):
    attention_plots = [
        np.zeros((max_length_targ, max_length_inp))]*(inp.shape[0])
    enc_output, enc_hidden = encoder(inp)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims(
        [targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
    batchpreds = []
    for t in range(max_length_targ):
        output = decoder(dec_input, dec_hidden, enc_output)
        predictions, dec_hidden = output[0], output[1]
        attention_plots[:][t] = tf.reshape(output[2], (inp.shape[0], -1))
        predicted_id = tf.argmax(predictions, axis=1).numpy()
        batchpreds.append(predicted_id)
        dec_input = tf.expand_dims(predicted_id, 1)
    batchpreds = np.array(batchpreds).T
    batch_results = []
    for i in range(inp.shape[0]):
        curr_result = ""
        for t in batchpreds[i]:
            curr_result += targ_lang.index_word[t] + ' '
            if targ_lang.index_word[t] == "<end>":
                break
        batch_results.append(curr_result)
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


log_table = []

num_log = 20
num_attn_log = 1

val_edit_dist = 0.0
val_acc = 0

# Load dataset
test_input_tensor, test_target_tensor, test_inp_lang, test_targ_lang, test_max_length_targ, test_max_length_inp = load_tensors(
    "dev", 5000)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

val_acc = 0
for (batch, (inpu, target)) in enumerate(dataset.take(steps_per_epoch)):
    batch_results, attention_plots = evaluate(inpu)
    for i in range(target.shape[0]):
        predicted_sent = "".join(batch_results[i].split(' ')[:-2])
        target_np = np.array(target)
        inpu_np = np.array(inpu)
        targ = "".join([targ_lang.index_word[c]
                        for c in target_np[i] if(c > 2)])
        inp = "".join([inp_lang.index_word[c]
                       for c in inpu_np[i] if(c > 3)])
    # if(batch % 100) == 0:
        edit_dist = editDistance(predicted_sent, targ,
                                 len(predicted_sent), len(targ))/len(targ)
        val_edit_dist += edit_dist
        if(targ == predicted_sent):
            val_acc += 1
        if (i + batch*BATCH_SIZE < num_log):
            log_table.append([inp, predicted_sent, targ, edit_dist])
        if ((config.attention) and (i + batch*BATCH_SIZE < num_attn_log)):
            attention_plot = attention_plots[i][:len(batch_results[i]),
                                                :len(targ)+2]
            plot_attention(attention_plot, [
                           '<start>'] + list(targ) + ['<end>'], list("".join(batch_results[i].split(' ')[:-1])) + ["<end>"])

    print(
        f'Input: {inp}, Prediction: {predicted_sent}, Target:{targ}, Edit-dist: {edit_dist}')

VAL_SAMPLES = len(test_input_tensor)

val_edit_dist /= VAL_SAMPLES
val_acc /= VAL_SAMPLES

wandb.log({"Validation log table": wandb.Table(data=log_table,
                                               columns=["Input", "Prediction", "Target", "Edit-dist"])})
wandb.log({"val_avg_edit_dist": val_edit_dist, "val_avg_acc": val_acc})
