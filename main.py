import data_access
import model
import test


parameters = {'V': None,
              'D': None,
              'csv_path': 'music_artist_genre_bio.csv',
              'csv_path_test': None
}


def main(params):
    sentences, sentences_idx, word_to_idx, all_word_counts_idx, params = data_access.get_music_bio(params)
    tf_inputs = model.get_model_parameters(params)
    W, V = model.learning_process(sentences_idx, all_word_counts_idx, tf_inputs)
    test.embedding_test(W, V, params['test_set'], word_to_idx)


if __name__ == '__main__':
    main(parameters)