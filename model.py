import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dims, enc_units, batch_size, dropout):
        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dims)
        self.enc_units = enc_units
        self.batch_size = batch_size

        self.lstm_1 = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences = True,
            return_state = True,
            dropout = dropout,
            recurrent_initializer = 'glorot_uniform'
        )

        self.lstm_2 = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences = True,
            return_state = True,
            dropout = dropout,
            recurrent_initializer = 'glorot_uniform'
        )

    def call(self, x, pre_state):
        x = self.embedding(x)
        x, h_state_1, c_state_1 = self.lstm_1(x, initial_state=pre_state[0])
        output, h_state_2, c_state_2 = self.lstm_2(x, initial_state=pre_state[1])
        state = [[h_state_1, c_state_1], [h_state_2, c_state_2]]

        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size,self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_size, self.enc_units))