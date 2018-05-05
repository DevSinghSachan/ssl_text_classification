import argparse
import numpy as np
import io
import os
import h5py
from gensim.models import KeyedVectors
import pickle


def load_pretrained_embeddings(filename, known_vocab=None, unif_weight=None):
    uw = 0.0 if unif_weight is None else unif_weight
    word_vectors = []
    count = 0
    unknown_vocab = []
    embeddings = KeyedVectors.load_word2vec_format(filename, binary=False)
    for i in range(len(known_vocab)):
        word = known_vocab[i]
        try:
            word_vec = embeddings.vectors[embeddings.vocab[word].index]
            count += 1
        except KeyError:
            word_vec = np.random.uniform(-uw, uw, embeddings.vector_size)
            unknown_vocab.append(word)
        word_vectors.append(word_vec)

    print("Percentage of Pre-trained word vectors: {}".
          format(count / float(len(known_vocab)) * 100))
    return np.array(word_vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='w2v.py')
    parser.add_argument('--input', required=True,
                        help='Path to the input directory')
    parser.add_argument('--save_data', required=True,
                        help='Path to the data file')
    parser.add_argument('--embeddings', required=True,
                        help='Path to the word vector embeddings file')
    args = parser.parse_args()

    with io.open(os.path.join(args.input, args.save_data + '.vocab.pickle'),
                 'rb') as f:
        id2w = pickle.load(f)

    word_vectors = load_pretrained_embeddings(args.embeddings, id2w,
                                              unif_weight=0.25)
    np.save(os.path.join(args.input, args.save_data + '.word_vectors.npy'), word_vectors)

    # with h5py.File(os.path.join(args.input, args.save_data + '.word_vectors.h5'), 'w') as hf:
    #     hf.create_dataset("elec_word_vectors", data=word_vectors, compression="gzip", compression_opts=9)

