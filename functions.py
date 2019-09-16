import string
import numpy as np
import sys


def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3


def get_negative_sampling_distribution(sentences, word_freq):

    V = len(word_freq)
    p_neg = np.zeros(V)

    for j in range(V):
        p_neg[j] = word_freq[j] ** 0.75

    # normalize it
    p_neg = p_neg / p_neg.sum()
    return p_neg


def droping_word_with_p_negtive_sampling(sentence, p_drop):
    sentence_adj = []
    for w in sentence:
        if np.random.random() < (1 - p_drop[w]):
            sentence_adj.append(w)

    # randomly order words so we don't always see
    randomly_ordered_positions = np.random.choice(len(sentence_adj), size=len(sentence_adj), replace=False)
    return sentence_adj, randomly_ordered_positions

def shuffle_sentences(sentences_idx):
    np.random.shuffle(sentences_idx)

def get_context(pos, wordidx, sentence, window_size):
    # input:
    # a sentence of the form: x x x x c c c pos c c c x x x x
    # output:
    # the context word indices: c c c c c c

    start = max(0, pos - window_size - 1)
    end_  = min(len(sentence), pos + window_size)
    context = sentence[start:end_]
    context.remove(wordidx)
    return context

def get_idx_to_word(word_to_idx):
    return {i:w for w, i in word_to_idx.items()}