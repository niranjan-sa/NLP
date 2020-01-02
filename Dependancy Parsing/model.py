# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start
        return tf.pow(vector, 3)
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda
        #print ("Booo Yaa")
        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_transitions = num_transitions
        self.trainable_embeddings = trainable_embeddings
        # Weight matrices initialization
        self.embeddings = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim], stddev=0.35), name="embeddings", trainable=trainable_embeddings)

        self.hidden_weights = tf.Variable(tf.random.truncated_normal([num_tokens*embedding_dim, hidden_dim], stddev=0.02), dtype=tf.float32, trainable=True)
        # Bias initialization
        self.hidden_bias = tf.Variable(tf.zeros(hidden_dim), dtype=tf.float32, trainable=True)
        self.output_weights = tf.Variable(tf.random.truncated_normal([num_transitions, hidden_dim], stddev=0.02), dtype=tf.float32, trainable=True)

        # TODO(Students) End

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        inp_embed = tf.reshape(tf.gather(self.embeddings, inputs), [inputs.shape[0], self.num_tokens*self.embedding_dim])
        hidden_lyr_out = self._activation(tf.add(tf.matmul(inp_embed, self.hidden_weights), self.hidden_bias))
        output_lyr_out = tf.matmul(hidden_lyr_out, self.output_weights, transpose_a=False, transpose_b=True)
        # Logits as output layer
        logits = output_lyr_out

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict

    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        # masking
        msk = tf.greater_equal(labels, 0)
        seq_msk = tf.keras.backend.cast(msk, tf.keras.backend.floatx())

        # softmax of logits

        softmx = tf.nn.softmax(logits*seq_msk)

        msk = (labels==1)
        msk = tf.keras.backend.cast(msk, tf.keras.backend.floatx())
        prod = labels * msk

        result = softmx*prod
        # calculating loss
        # Library function
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        delta = 1e-11
        loss = -tf.reduce_mean(tf.math.log(tf.add(tf.reduce_sum(result, axis=1), delta)))

        # calculating ridge norm
        l2_norm = tf.nn.l2_loss(self.hidden_weights) + tf.nn.l2_loss(self.hidden_bias) + tf.nn.l2_loss(self.output_weights)
        regularization = self._regularization_lambda/2*l2_norm

        # TODO(Students) End
        return loss + regularization
