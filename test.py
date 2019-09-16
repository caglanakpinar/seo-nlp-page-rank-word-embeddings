import functions

import pandas as pd
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances

def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape

    # don't actually use pos2 in calculation, just print what's expected
    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("Sorry, %s not in word2idx" % w)
            return

    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]

    vec = p1 - n1 + n2
    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]
    # pick one that's not p1, n1, or n2
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    for i in idx:
        if i not in keep_out:
            best_idx = i
            break

    print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[idx[0]], neg2))
    print("closest 10:")
    for i in idx:
        print(idx2word[i], distances[i])

    print("dist to %s:" % pos2, cos_dist(p2, vec))

def embedding_test(W, V, test_set_path, word_to_idx):
    idx_to_word = functions.get_idx_to_word(word_to_idx)
    test_analogies = pd.read_csv(test_set_path).to_dict("results")
    counter = 0
    for We in (W, (W + V.T) / 2):
        t_vec = test_analogies[counter]
        analogy(t_vec[0], t_vec[1], t_vec[2], t_vec[3], word_to_idx, idx_to_word, We)
        counter += 1
