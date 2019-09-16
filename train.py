import datetime
import functions
import numpy as np
import tensorflow as tf
import constants


class train_init_model:
    def __init__(self, randomly_ordered_positions, sentence, tf_inputs, prob_neg, vocab_size, session):
        self.cost = 0
        self.counter = 0
        self.inputs = []
        self.targets = []
        self.negwords = []
        self.window_size = constants.window_size
        self.sentence = sentence
        self.t0 = datetime.datetime.now()
        self.at_least_for_session_run = constants.at_least_for_session_run
        self.randomly_ordered_positions = randomly_ordered_positions
        self.vocab_size = vocab_size
        self.prob_neg = prob_neg
        self.tf_inputs = tf_inputs
        self.session = session

    def min_range_of_inputs_collecting(self, pos):
        word = self.sentence[pos]
        context_words = functions.get_context(pos, word, self.sentence, self.window_size)
        neg_word = np.random.choice(self.vocab_size, p=self.prob_neg)
        n = len(context_words)
        self.inputs += [word] * n
        self.negwords += [neg_word] * n
        self.targets += context_words
        self.counter += 1

    def session_run(self):
        _, c = self.session.run((self.tf_inputs['train_op'], self.tf_inputs['loss']),
                                feed_dict={self.tf_inputs['tf_input']: self.inputs,
                                         self.tf_inputs['tf_negword']: self.negwords,
                                         self.tf_inputs['tf_context']: self.targets} )
        self.inputs = []
        self.negwords = []
        self.targets = []
        self.cost += c

    def compute_train_procces(self, ):
        for j, pos in enumerate(self.randomly_ordered_positions):
            self.min_range_of_inputs_collecting(pos)
            if len(self.inputs) >= self.at_least_for_session_run:
                self.session_run()
            self.counter += 1
