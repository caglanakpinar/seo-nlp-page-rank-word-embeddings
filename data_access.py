from functions import remove_punctuation
import pandas as pd

def get_music_bio(params):
    files = pd.read_csv(params['csv_path']).to_dict('resutls')
    params['D'] = params['D'] if params['D'] else len(files)
    all_word_counts = {}
    for f in files[:params['D']]:
        line = f['content']
        s = remove_punctuation(line).lower().split()
        if len(s) > 1:
            for word in s:
                if word not in all_word_counts:
                    all_word_counts[word] = 0
                else:
                    all_word_counts[word] += 1
    params['V'] = params['V'] if params['V'] else len(all_word_counts)
    V = min(params['V'], len(all_word_counts))
    all_word_counts_idx = all_word_counts
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}
    all_word_counts_idx = {ind: all_word_counts_idx[w] if w != '<UNK>' else 0 for ind, w in enumerate(word2idx)}
    print("finished counting")
    unk = word2idx['<UNK>']
    sents = []
    sentences = []
    for f in files[:params['D']]:
        content = f['content']
        for sentence in content.split("."):
            sentence = remove_punctuation(sentence).lower()
            if len(sentence.split()) > 1:
                sent = [word2idx[w] if w in word2idx and w != ' ' else unk for w in sentence.split()]
                sentences.append(sentence)
                sents.append(sent)
    return sentences, sents, word2idx, all_word_counts_idx, params