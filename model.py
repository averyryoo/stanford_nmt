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

class Attention(tf.keras.Model):
    def __init__(self, enc_units, method='concat'):
        super(Attention, self).__init__()
        self.method = method
        self.W = tf.keras.layers.Dense(enc_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_h_t, enc_h_s):

        if self.method == 'concat':
            score = self.V(tf.nn.tanh(self.W_a(dec_h_t + enc_h_s)))
        elif self.method == 'general':
            score = tf.matmul(self.W(enc_h_s), dec_h_t, transpose_b=True)
        elif self.method == 'dot':
            score = tf.matmul(enc_h_s, dec_h_t, transpose_b=True) 

        # a_t shape == (batch_size, seq_len, 1)
        a_t = tf.nn.softmax(score, axis=1)

        context_vector = tf.reduce_sum(a_t * enc_h_s, axis=1)

        return context_vector

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dims, dec_units, method, batch_size, dropout):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dims)

        self.lstm_1 = tf.keras.layers.LSTM(
            self.dec_units,
            return_sequences = True,
            return_state = True,
            dropout = dropout,
            recurrent_initializer = 'glorot_uniform'
        )

        self.lstm_2 = tf.keras.layers.LSTM(
            self.dec_units,
            return_sequences = True,
            return_state = True,
            dropout = dropout,
            recurrent_initializer = 'glorot_uniform'
        )
    
        self.attention = Attention(dec_units, method)

        self.W_c = tf.keras.layers.Dense(embed_dims, activation='tanh')

        self.W_s = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, pre_state, enc_output, pre_h_t):
        x = self.embedding(x)
        x = tf.concat([x, pre_h_t], axis=-1)
        x, h_state_1, c_state_1 = self.lstm_1(x, initial_state=pre_state[0])
        dec_output, h_state_2, c_state_2 = self.lstm_2(x, initial_state=pre_state[1])
        state = [[h_state_1, c_state_1], [h_state_2, c_state_2]]
        context_vector = self.attention(dec_output, enc_output)
        h_t = self.W_c(tf.concat([tf.expand_dims(context_vector, 1), dec_output], axis=-1))

        y_t = tf.squeeze(self.W_s(h_t), axis = 1)

        return y_t, state, h_t
        