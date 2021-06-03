import tensorflow as tf
import encode_input
import beam_search_fast
import numpy as np
from tqdm import tqdm
import wandb

# Wandb default config
config_defaults = {
    "epochs": 10,
    "batch_size": 128,
    "layer_dimensions": [128, 128],
    "cell_type": "LSTM",
    "dropout": 0.1,
    "recurrent_dropout": 0.1,
    "optimizer": "adam",
    "attention": False,
    "attention_shape": 256
}

# Initialize the project
wandb.init(project='assignment3',
           group='Best Model Test',
           config=config_defaults)

# config file used for the current run
config = wandb.config

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


model_name = input("Enter name of model folder : ")

# load saved model
#model = tf.keras.models.load_model("Good_model_no_beam_search/train")
#print(model.summary())

inf_enc_model = tf.keras.models.load_model(model_name + "/inf-enc")
inf_dec_model = tf.keras.models.load_model(model_name + "/inf-dec")

wandb.run.name = model_name

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in data_encoder.input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in data_encoder.target_token_index.items())

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
            if j == 0:
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


input_seqs = encoder_input_data["test"]
target_sents = target_texts_dict["test"]
input_texts = input_texts_dict["test"]
n = len(input_seqs)
val_avg_edit_dist = 0
log_table = []
test_acc = 0
BATCH_SIZE = 64
beam_width = int(input("Enter beam width : "))

predictions_vanilla = open("predictions_vanilla" + model_name, 'w')
for seq_index in tqdm(range(0, n, BATCH_SIZE)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    # make the beam search object
    num_inputs = min(n, seq_index + BATCH_SIZE) - seq_index
    bs = beam_search_fast.BeamSearch(beam_width, data_encoder.max_decoder_seq_length, data_encoder.target_token_index, num_inputs)
    input_seq = input_seqs[seq_index:min(n, seq_index + BATCH_SIZE)]
    decoded_sentences = bs.apply(inf_enc_model, inf_dec_model, input_seq)
    
    decoded_sentences = ["".join(e[0].characters[1:-1]) for e in decoded_sentences]
    target_sentences = [str(target_sents[i : i + 1][0][1:-1]) for i in range(seq_index, min(n, seq_index + BATCH_SIZE))]
    edit_distances = []
    for i in range(0, len(decoded_sentences)) :
      edit_dist = editDistance(decoded_sentences[i], target_sentences[i], len(
          decoded_sentences[i]), len(target_sentences[i]))/len(target_sentences[i])
      val_avg_edit_dist += edit_dist
      edit_distances.append(edit_dist)
      if(decoded_sentences[i] == target_sentences[i]):
          test_acc += 1
    
    for i in range(seq_index, min(n, seq_index + BATCH_SIZE)) :
        predictions_vanilla.write(input_texts[i] + " | " + decoded_sentences[i - seq_index] + " | " + target_sentences[i - seq_index] + '\n')

    if(seq_index < BATCH_SIZE):
        for i in range(seq_index, min(n, seq_index + BATCH_SIZE)) :
          log_table.append(
              [input_texts[i], decoded_sentences[i - seq_index], target_sentences[i - seq_index], edit_distances[i - seq_index]])
          print({f"input_{i}": input_texts[i], f"output_{i}": decoded_sentences[i - seq_index],
                 f"target_{i}": target_sentences[i - seq_index], f"edit_distance_{i}": edit_distances[i - seq_index]})

wandb.log({"Validation log table": wandb.Table(data=log_table,
                                               columns=["Input", "Prediction", "Target", "Edit-dist"])})
    

#val_avg_edit_dist /= val_samples
#val_acc /= val_samples
#wandb.log({"val_avg_edit_dist": val_avg_edt_dist, "val_avg_acc": val_acc})

print("Test accuracy is : ", test_acc / n)
wandb.log({"Test accuracy" : test_acc / n})
