import numpy as np
import tensorflow as tf
import functions
import constants

def dot(A, B):
    C = A * B
    # C = [[1, 1, 1], [1, 1, 1]]; reduce_sum(C, axis=1) = [1, 1] + [1, 1] + [1, 1] = [3, 3]
    # C = [[1, 1, 1], [1, 1, 1]]; reduce_sum(C, axis=0) = [1, 1, 1] + [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    return tf.reduce_sum(C, axis=1)


def model(vocab_size, D):
    W = np.random.randn(vocab_size, D).astype(np.float32)
    V = np.random.randn(D, vocab_size).astype(np.float32)
    tf_input = tf.placeholder(tf.int32, shape=(None,))
    tf_negword = tf.placeholder(tf.int32, shape=(None,))
    tf_context = tf.placeholder(tf.int32, shape=(None,))  # targets (context)
    tfW = tf.Variable(W)
    tfV = tf.Variable(V.T)
    # positive loss
    emb_input = tf.nn.embedding_lookup(tfW, tf_input)  # 1 x D
    emb_output = tf.nn.embedding_lookup(tfV, tf_context)  # N x D
    correct_output = dot(emb_input, emb_output)  # N
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones(tf.shape(correct_output)), logits=correct_output)
    # negative loss
    emb_input = tf.nn.embedding_lookup(tfW, tf_negword)
    incorrect_output = dot(emb_input, emb_output)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(tf.shape(incorrect_output)),
                                                       logits=incorrect_output)
    # total loss
    loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)
    train_op = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
    return  train_op

def train_model():
    costs = []
    for epoch in range(epochs):
        shuffle_sentences(sentences_idx)

        for sentence in sentences_idx:
            sentence, randomly_ordered_positions = droping_word_with_p_negtive_sampling(sentence)
            if len(sentence) < 2:
                continue
            train_m = train_init_model(at_least_for_session_run, randomly_ordered_positions, window_size, sentence)
            train_m.compute_train_procces()
            learning_rate -= learning_rate_delta
            costs.append(train_m.cost)
        W, VT = session.run((tfW, tfV))
        V = VT.T
    return W, T

def learning_process(train_op, sentences, sentences_idx, all_word_counts_idx):
    session = tf.Session()
    init_op = tf.global_variables_initializer()
    session.run(init_op)

    # save the costs to plot them per iteration
    costs = []
    total_words = sum(len(sentence) for sentence in sentences_idx)
    print("total number of words in corpus:", total_words)
    prob_neg = functions.get_negative_sampling_distribution(sentences_idx, all_word_counts_idx)
    p_drop = 1 - np.sqrt(constants.threshold / prob_neg)
    train_model()


