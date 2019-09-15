import datetime
import functions
import numpy as np


class train_init_model:
    def __init__(self, at_least_for_session_run, randomly_ordered_positions, window_size, sentence, vocab_size, prob_neg):
        self.cost = 0
        self.counter = 0
        self.inputs = []
        self.targets = []
        self.negwords = []
        self.window_size = window_size
        self.sentence = sentence
        self.t0 = datetime.datetime.now()
        self.at_least_for_session_run = at_least_for_session_run
        self.randomly_ordered_positions = randomly_ordered_positions
        self.vocab_size = vocab_size
        self.prob_neg = prob_neg
        self.train_op =

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
        _, c = tf.session.run((self.train_op, loss), feed_dict={tf_input: train_m.inputs, tf_negword: train_m.negwords,
                                                        tf_context: train_m.targets})

        self.inputs = []
        self.negwords = []
        self.targets = []
        self.cost += c

    def cost_plot(self):
        if self.counter % 100 == 0:
            sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
            sys.stdout.flush()

    def compute_train_procces(self):
        for j, pos in enumerate(self.randomly_ordered_positions):
            self.min_range_of_inputs_collecting(pos)
            if len(self.inputs) >= self.at_least_for_session_run:
                self.session_run()
            train_m.counter += 1
