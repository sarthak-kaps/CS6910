import tensorflow as tf
import numpy as np


class FeedBack(tf.keras.Model):

    def __init__(self, config, dec_timesteps, dec_vsize):
        super().__init__()
        self.config = config
        self.lstm_cell = tf.keras.layers.LSTMCell(
            self.config.layer_dimension, recurrent_dropout=config.recurrent_dropout)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(
            self.lstm_cell, return_state=True, return_sequences=True, unroll=True)
        self.dense = tf.keras.layers.Dense(dec_vsize)
        self.dec_timesteps = dec_timesteps
        self.dec_vsize = dec_vsize

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)
        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        i = 0
        while(True):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            print(x.shape)
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            print(x.shape)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)
            if((np.argmax(prediction) == 0) or (i == self.dec_timesteps)):
                break

            i += 1

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


if __name__ == "__main__":

    class default_config:
        cell_type = "LSTM"
        layer_dimension = 10
        attention = True
        dropout = 0.1
        recurrent_dropout = 0.1

    f = FeedBack(default_config(), 11, 12)
    a = np.ones((1, 11, 12))
    p = f(a)
    # print(p)
    print(p.shape)
