import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):
    """
    The code for this model is inspired from the paper given below -
    Attention-Based Bidirectional Long Short-Term Memory Networks for
    Relation Classification - https://www.aclweb.org/anthology/P16-2034.pdf
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()
        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.training = training
        self.gru_layer = layers.Bidirectional(layers.GRU(hidden_size, return_sequences=True,
                                                         recurrent_activation='sigmoid'))
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        M = tf.tanh(rnn_outputs)
        alpha = tf.tensordot(M, self.omegas, axes=1)
        alpha = tf.nn.softmax(alpha)
        output = tf.reduce_sum(rnn_outputs * alpha, 1)
        output = tf.tanh(output)
        ### TODO(Students) END
        return output

    def call(self, inputs, pos_inputs, training):
        tokens_mask = tf.cast(inputs != 0, tf.float32)
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        # TODO(Students) START
        # This is the code for basic implementation of the bi-directional neural network
        # the 3 other experiments mentioned in the assignment PDF were performed by modifying
        # this code
        # Normal mode - data = tf.concat([word_embed, pos_embed], axis=2)
        # Only word embedding features ---- data = word_embed (with dep removed in data.py)
        # Dropping word + pos features ---- data = tf.concat([word_embed, pos_embed], axis=2)
        # (with dep removed in data.py)
        # word embedding + dep features ---- data = word_embed (with dep not removed in data.py)
        data = tf.concat([word_embed, pos_embed], axis=2)
        H = self.gru_layer(data, mask=tokens_mask)
        attention = self.attn(H)
        logits = self.decoder(attention)
        # TODO(Students) END

        return {'logits': logits}


class MyAdvancedModel(models.Model):
    """
    A simple implementation of a CNN with only 1 CNN layer and Max Pooling.
    """
    def __init__(self, vocab_size: int, embed_dim: int, training: bool = False):
        super(MyAdvancedModel, self).__init__()

        # TODO(Students) START
        self.num_classes = len(ID_TO_CLASS)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.conv_layer1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', data_format='channels_last')
        self.max_pool1 = tf.keras.layers.GlobalMaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout_layer1 = tf.keras.layers.Dropout(0.1)
        self.softmax_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        # TODO(Students END

    def call(self, inputs, pos_inputs, training):
        # TODO(Students) START
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        # stack both of them together
        data = tf.concat([word_embed, pos_embed], axis=2)
        # expand the last dimension as we are applying a convolution for dimension adjustment.
        data = tf.expand_dims(data, 3)
        df = self.conv_layer1(data)
        df = self.max_pool1(df)
        df = self.flatten(df)
        df = tf.reshape(df, [data.shape[0], -1])
        df = self.dropout_layer1(df)
        logits = self.softmax_layer(df)
        # TODO(Students END
        return {'logits': logits}
