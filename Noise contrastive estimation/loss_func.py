import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    A = tf.diag_part(tf.log(tf.math.exp(tf.matmul(inputs, tf.transpose(true_w)))+1e-14))
    B = tf.log(tf.reduce_sum(tf.math.exp(tf.matmul(inputs, tf.transpose(true_w))), axis=1)+1e-14)
    """
    mat_prod_A = tf.log(tf.exp(tf.matmul(tf.transpose(true_w), inputs)))+1e-10
    A = mat_prod_A
    mat_prod_B = tf.matmul(tf.transpose(true_w), inputs)
    mat_prod_B_sum = tf.reduce_sum(tf.math.exp(mat_prod_B), axis=1)
    B = tf.log(mat_prod_B_sum)+1e-10"""
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    ##reshape (-1, embedding_size)
    pos_size = tf.nn.embedding_lookup(weights, labels).shape[0]
    pos_wts = tf.reshape(tf.nn.embedding_lookup(weights, labels), shape=[-1, pos_size])
    pos_bias = tf.nn.embedding_lookup(biases, labels)
    pos_prod = tf.diag_part(tf.matmul(pos_wts, tf.transpose(inputs)))
    pos_prod_shape = pos_prod.shape[0]
    pos_prod = tf.reshape(pos_prod, shape=[pos_prod_shape, 1])
    pos_add = tf.add(pos_prod, pos_bias)

    # pos_s = tf.diag_part(tf.math.add(tf.matmul(tf.transpose(inputs), tf.reshape(pos_wts, [-1, 128])), pos_bias))
    pos_uprob = tf.gather(unigram_prob, labels)
    pos_log_kuprob = tf.log(tf.multiply(float(len(sample)), pos_uprob)+ 1e-14)
    pos_diff = tf.subtract(pos_add, pos_log_kuprob)
    pos_Pr = tf.sigmoid(pos_diff)

    neg_size = tf.nn.embedding_lookup(weights, sample).shape[0]
    neg_wts = tf.nn.embedding_lookup(weights, sample)
    neg_bias = tf.nn.embedding_lookup(biases, sample)
    neg_bias = tf.reshape(neg_bias, shape=[neg_size, 1])
    neg_bias = tf.tile(neg_bias, [1, 128])
    neg_prod = tf.matmul( neg_wts, tf.transpose(inputs))

    neg_sum = tf.add(neg_prod, neg_bias)
    # neg_s = tf.math.add(tf.matmul(tf.transpose(inputs), tf.transpose(neg_wts)), neg_bias)
    neg_uprob = tf.gather(unigram_prob, sample)
    neg_uprob_size = neg_uprob.shape[0]
    neg_uprob = tf.reshape(neg_uprob, shape=[neg_uprob_size, 1])
    neg_uprob = tf.tile(neg_uprob, [1, 128])
    neg_log_kuprob = tf.log(tf.multiply(float(len(sample)), neg_uprob) + 1e-14)
    neg_diff = tf.subtract(tf.transpose(neg_sum), tf.transpose(neg_log_kuprob))
    neg_Pr = tf.sigmoid(neg_diff)
    neg_summation = tf.reduce_sum(tf.log(tf.subtract(float(1), neg_Pr)+ 1e-14) , axis=1)
    neg_summation_shape = neg_summation.shape[0]
    neg_summation = tf.reshape(neg_summation, shape=[neg_summation_shape, 1])
    total_summation = tf.add(tf.log(pos_Pr+ 1e-14) , neg_summation)

    return tf.multiply(float(-1),total_summation)