import tensorflow as tf

dispatch_dict = {"LSTM": tf.keras.layers.LSTM,
                 "GRU": tf.keras.layers.GRU, "RNN": tf.keras.layers.SimpleRNN}


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, config):
        super(Encoder, self).__init__()
        self.config = config
        self.batch_sz = self.config.batch_size
        self.layer_dimensions = self.config.layer_dimensions
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, self.config.embedding_dim)

        self.model = []

        for i in range(0, len(self.layer_dimensions)):
            self.model.append(dispatch_dict[config.cell_type](self.layer_dimensions[i],
                                                              return_sequences=True,
                                                              return_state=True,
                                                              recurrent_initializer='glorot_uniform'))

    def call(self, x):
        x = self.embedding(x)

        for i in range(0, len(self.model)):
            hidden = [tf.zeros((x.shape[0], self.layer_dimensions[i]))] * \
                (2**(self.config.cell_type == "LSTM"))
            layer_out = self.model[i](x, initial_state=hidden)
            x, state = layer_out[0], layer_out[1:]
            state, x = tf.concat(
                list(layer_out[1:]), axis=-1), layer_out[0]
        output = x
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):

        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size,  config):
        super(Decoder, self).__init__()
        self.config = config
        self.batch_sz = self.config.batch_size
        self.layer_dimensions = self.config.layer_dimensions
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, self.config.embedding_dim)

        self.model = []
        for i in range(0, len(self.layer_dimensions)):
            self.model.append(dispatch_dict[config.cell_type](self.layer_dimensions[i],
                                                              return_sequences=True,
                                                              return_state=True,
                                                              recurrent_initializer='glorot_uniform'))

        self.fc = tf.keras.layers.Dense(vocab_size)

        if(config.attention):
            # used for attention
            self.attention = BahdanauAttention(self.config.attention_shape)

    def call(self, x, hidden, enc_output):

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # enc_output shape == (batch_size, max_length, hidden_size)
        if self.config.attention:
            context_vector, attention_weights = self.attention(
                hidden, enc_output)
            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        else:
            x = tf.concat([tf.expand_dims(hidden, 1), x], axis=-1)

        # passing the concatenated vector through the model
        for i in range(0, len(self.model)):
            layer_out = self.model[i](x)
            state, x = tf.concat(
                list(layer_out[1:]), axis=-1), layer_out[0]

        output = x
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        if(self.config.attention):
            return x, state, attention_weights
        else:
            return x, state
