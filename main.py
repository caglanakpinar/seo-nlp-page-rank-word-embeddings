import data_access
import model
import test


parameters = {'V': None,
              'D': None,
              'csv_path': 'music_artist_genre_bio.csv',
              'csv_path_test': 'music_artist_genre_bio.csv',
              'word_embedding_1': 'label',
              'word_embedding_2': 'genre',
              'words_embeding_words': ['bob wills', 'radiohead', 'u2', 'aerosmith',
                                       'fatboy slim', 'kathy' 'the prodigy']
}

def main(params):
    sentences, sentences_idx, word_to_idx, all_word_counts_idx, params = data_access.get_music_bio(params)
    tf_inputs = model.get_model_parameters(params)
    W, V = model.learning_process(sentences_idx, all_word_counts_idx, tf_inputs)
    results = test.embedding_test(W, V, params, word_to_idx)

if __name__ == '__main__':
    main(parameters)