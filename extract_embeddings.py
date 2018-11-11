from __future__ import division
import torch
import argparse
import io
import os
import pickle
from gensim.models import KeyedVectors


parser = argparse.ArgumentParser(description='extract_embeddings.py')

parser.add_argument('--input', required=True, help='Path to the input directory')
parser.add_argument('--data', required=True, help='Path to the data file')
parser.add_argument('--model', required=True, help='Path to model .pt file')
parser.add_argument('--output_dir', default='.', help='Path to output the embeddings')
parser.add_argument('--gpu', type=int, default=-1, help="Device to run on")


def write_embeddings(filename, dict, embeddings):
    with open(filename, 'wb') as file:
        header = "{} {}".format(len(dict), len(embeddings[0]))
        file.write(header.encode("utf-8") + b"\n")
        for i in range(min(len(embeddings), len(dict))):
            str = dict[i].encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            file.write(str + b"\n")


def main():
    args = parser.parse_args()
    args.cuda = args.gpu > -1
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    with io.open(os.path.join(args.input, args.data + '.vocab.pickle'), 'rb') as f:
        id2w = pickle.load(f)

    # Add in default model arguments, possibly added since training.
    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    encoder_embeddings = model['state_dict_embedder']['embedder.weight'].tolist()

    print("Writing embeddings")
    write_embeddings(os.path.join(args.output_dir, "src_embeddings.txt"),
                     id2w, encoder_embeddings)
    print('... done.')

    embeddings = KeyedVectors.load_word2vec_format(os.path.join(args.output_dir, "src_embeddings.txt"),
                                                   binary=False)

    try:
        while True:
            query_word = input("Query Word: ")
            print(embeddings.similar_by_word(query_word, 10))
            print()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
