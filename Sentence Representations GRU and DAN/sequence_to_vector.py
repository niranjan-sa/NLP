# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    print("Debug here!")
    print("I made changes here")
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.dense = []
        for i in range(num_layers):
            temp = tf.keras.layers.Dense(input_dim, activation='relu')
            self.dense.append(temp)

        # Soft-max is declared but not used
        self.softmax = tf.keras.layers.Dense(input_dim, activation='softmax')
        self.dropout = dropout
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        sequence_mask = tf.expand_dims(sequence_mask, 2)
        vector_sequence = vector_sequence*sequence_mask
        # Drop out implementation
        if training:
            mask = tf.random.uniform((vector_sequence.shape[0], vector_sequence.shape[1], 1)) >= self.dropout
            vector_sequence = tf.keras.backend.cast(mask, 'float32')*vector_sequence
        vector_sequence = tf.reduce_mean(vector_sequence, axis=1)
        lyr = []
        for dense_layer in self.dense:
            vector_sequence = dense_layer(vector_sequence)
            lyr.append(vector_sequence)
        lyr = tf.stack(lyr, axis=1)
        # TODO(students): end
        # combined_vector = self.softmax(vector_sequence)
        layer_representations = lyr
        return {"combined_vector": vector_sequence,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.grus = []

        for i in range(num_layers):
            self.grus.append(tf.keras.layers.GRU(input_dim, return_sequences=True, return_state=True))
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # sequence_mask = tf.expand_dims(sequence_mask, 2)
        # vector_sequence = vector_sequence * sequence_mask
        layer = []
        vector_sequence, layer_state = self.grus[0](vector_sequence, mask=sequence_mask)
        layer.append(layer_state)
        i = 1
        while i < len(self.grus):
            vector_sequence, layer_state = self.grus[i](vector_sequence)
            layer.append(layer_state)
            i += 1

        layer = tf.stack(layer, axis=1)
        combined_vector = layer_state
        return {"combined_vector": combined_vector,
                "layer_representations": layer}
