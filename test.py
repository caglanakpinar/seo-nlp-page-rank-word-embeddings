from functions import get_fist_word, get_idx_to_word

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances


def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W, is_printing, max_distance):
    V, D = W.shape

    # don't actually use pos2 in calculation, just print what's expected

    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("Sorry, %s not in word2idx" % w)
            return max_distance

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
    if is_printing:
        print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[idx[0]], neg2))
        print("closest 10:")
        for i in idx:
            print(idx2word[i], distances[i])

        print("dist to %s:" % pos2, cos_dist(p2, vec))
    print("testing: %s - %s = %s - %s: distance = %s " % (pos1, neg1, pos2, neg2, abs(1 - cos_dist(p2, vec))))
    return abs(1 - cos_dist(p2, vec))

def embedding_test(W, V, params, word_to_idx):
    emb_1, emb_2 = params['word_embedding_1'], params['word_embedding_2']
    idx_to_word = get_idx_to_word(word_to_idx)
    df = pd.read_csv(params['csv_path_test'])

    word_emb_2 = {}
    emb_1_l = list(df[emb_1])
    word_emb_1 = {}
    count = 0
    for a in emb_1_l:
        if list(df[df[emb_1] == a][emb_2])[0] == list(df[df[emb_1] == a][emb_2])[0]:
            word_emb_2[a] = list(df[df[emb_1] == a][emb_2])[0]
            word_emb_1[a] = count
            count += 1
    A = np.zeros((len(emb_1_l), len(emb_1_l)))

    max_dist = 10000
    for a_1 in word_emb_1:
        for a_2 in word_emb_1:
            g_1, g_2 = word_emb_2[a_1], word_emb_2[a_2]
            if a_1 != a_2 and g_1 and g_2:
                A[word_emb_1[a_1], word_emb_1[a_2]] = analogy(get_fist_word(a_1), get_fist_word(g_1),
                                                              get_fist_word(a_2), get_fist_word(g_2), word_to_idx,
                                                              idx_to_word, W, False, max_dist)
    top_closest_words = []
    for i in params['words_embeding_words']:
        try:
            index = list(filter(lambda x: x[0] == i, list(zip(emb_1, list(range(len(emb_1_l)))))))[0][1]
            top_closest_words += list(zip([i for i in range(10)], sorted(list(zip(A[index], emb_1_l)))[:10]))
        except:
            print(i)
    results = pd.DataFrame(top_closest_words).rename(columns={0: 'word', 1: 'related_words'})
    results = pd.merge(results, df.rename(columns={emb_1:'word'})[[emb_2, 'word']], on='word', how='left')
    results = pd.merge(results,
                       df.rename(columns={emb_1: 'related_words', emb_2: emb_2 + '_related'})[[emb_2, 'related_words']],
                       on='related_words', how='left')

    results.to_csv("results.csv")
    print("results in results.csv")