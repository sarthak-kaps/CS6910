import model_maker
import encode_input
from wandb.keras import WandbCallback
import wandb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Wandb default config
config_defaults = {
    "epochs": 5,
    "batch_size": None,
    "layer_dimensions": [128],
    "cell_type": "LSTM",
    "dropout": 0.1,
    "recurrent_dropout": 0.1,
    "embedding_size": 16,
    "optimizer": "adam",
    "attention": True
}


# Initialize the project
wandb.init(project='assignment3',
           group='First Run',
           config=config_defaults)

# config file used for the current run
config = wandb.config


# config = def_config()

wandb.run.name = f"cell_type_{config.cell_type}_layer_org_{config.layer_dimensions}_embd_size_{config.embedding_size}_drpout_{config.dropout}_rec-drpout_{config.recurrent_dropout}_opt_{config.optimizer}"


base_data_set_name = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled."

data_encoder = encode_input.one_hot_encoder(
    [base_data_set_name + "train.tsv", base_data_set_name + "dev.tsv", base_data_set_name + "test.tsv"], ["train", "valid", "test"])


# encoder_input_data, decoder_input_data, decoder_target_data are dictionaries and contain the encoder input
# decoder input and the target data all in one hot encoding format for the train, valid and test datasets.
# Example -
# To obtain the encoder input for train data do encoder_input_data["train"]
# To obtain the decoder input for validation data do decoder_input_data["valid"]
[encoder_input_data, decoder_input_data,
    decoder_target_data] = data_encoder.vectorize_data()

# construct the model from the given configurations
model = model_maker.make_model(
    config, data_encoder.max_encoder_seq_length, data_encoder.num_encoder_tokens, data_encoder.max_decoder_seq_length, data_encoder.num_decoder_tokens)


# compile the model
model.compile(
    optimizer=config.optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# fit the model
model.fit(
    x=[encoder_input_data["train"], decoder_input_data["train"]],
    y=decoder_target_data["train"],
    # batch_size=config.batch_size,
    epochs=config.epochs,
    validation_data=([encoder_input_data["valid"],
                      decoder_input_data["valid"]], decoder_target_data["valid"]),
    callbacks=[WandbCallback()]
)

# Save model
model.save(wandb.run.name)
