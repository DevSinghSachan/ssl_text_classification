from __future__ import unicode_literals
import collections
import io
import re
import six
import numpy as np
import progressbar
import json
import os
import pickle
from collections import namedtuple
import torch

from config import get_preprocess_args


split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')

Special_Seq = namedtuple('Special_Seq', ['PAD', 'EOS', 'UNK', 'BOS'])
Vocab_Pad = Special_Seq(PAD=0, EOS=1, UNK=2, BOS=3)


def split_sentence(s, tok=False):
    if tok:
        s = s.lower()
        s = s.replace('\u2019', "'")
        # s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        if tok:
            words.extend(split_pattern.split(word))
        else:
            words.append(word)
    words = [w for w in words if w]
    return words


def open_file(path):
    return io.open(path, encoding='utf-8', errors='ignore')


def count_lines(path):
    with open_file(path) as f:
        return sum([1 for _ in f])


def read_file(path, tok=False):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    with open_file(path) as f:
        for line in bar(f, max_value=n_lines):
            tokens = line.strip().split('\t')
            label = tokens[0]
            text = ' '.join(tokens[1:])
            words = split_sentence(text, tok)
            yield label, words


def count_words(path, max_vocab_size=40000, tok=False):
    counts = collections.Counter()
    for _, words in read_file(path, tok):
        for word in words:
            counts[word] += 1
    vocab = [word for (word, _) in counts.most_common(max_vocab_size)]
    return vocab


def get_label_vocab(path, tok=False):
    vocab_label = set()
    for label, _ in read_file(path, tok):
        vocab_label.update(label)
    return sorted(list(vocab_label))


def make_dataset(path, w2id, tok=False):
    labels = []
    dataset = []
    token_count = 0
    unknown_count = 0
    for label, words in read_file(path, tok):
        labels.append(label)
        array = make_array(w2id, words)
        dataset.append(array)
        token_count += array.size
        unknown_count += (array == Vocab_Pad.UNK).sum()
    print('# of tokens: %d' % token_count)
    print('# of unknown: %d (%.2f %%)' % (unknown_count,
                                          100. * unknown_count / token_count))
    return labels, dataset


def make_array(word_id, words):
    ids = [word_id.get(word, Vocab_Pad.UNK) for word in words]
    return np.array(ids, 'i')


if __name__ == "__main__":
    args = get_preprocess_args()

    print(json.dumps(args.__dict__, indent=4))

    # Vocab Construction
    train_path = os.path.join(args.input, args.train_filename)
    valid_path = os.path.join(args.input, args.dev_filename)
    test_path = os.path.join(args.input, args.test_filename)
    unlabel_path = os.path.join(args.input, args.unlabel_filename)

    word_cntr = count_words(unlabel_path, args.vocab_size, args.tok)

    all_words = word_cntr
    vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + all_words
    w2id = {word: index for index, word in enumerate(vocab)}

    label_list = get_label_vocab(train_path, args.tok)
    label2id = {l: index for index, l in enumerate(label_list)}

    # Unlabelled Dataset
    labels, data = make_dataset(unlabel_path, w2id, args.tok)
    unlabel_data = [(-1, s) for l, s in six.moves.zip(labels, data)
                    if 0 < len(s)]

    # Train Dataset
    labels, data = make_dataset(train_path, w2id, args.tok)
    train_data = [(label2id[l], s) for l, s in six.moves.zip(labels, data)
                  if 0 < len(s) < args.max_seq_length]

    # Display corpus statistics
    print("Vocab: {}".format(len(vocab)))
    print('Original training data size: %d' % len(data))
    print('Filtered training data size: %d' % len(train_data))

    # Valid Dataset
    labels, data = make_dataset(valid_path, w2id, args.tok)
    valid_data = [(label2id[l], s) for l, s in six.moves.zip(labels, data)
                  if 0 < len(s)]

    # Test Dataset
    labels, data = make_dataset(test_path, w2id, args.tok)
    test_data = [(label2id[l], s) for l, s in six.moves.zip(labels, data)
                 if 0 < len(s)]

    id2w = {i: w for w, i in w2id.items()}
    id2label = {i: l for l, i in label2id.items()}

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save the dataset as pytorch serialized files
    torch.save(unlabel_data,
               os.path.join(args.output, args.save_data + '.unlabel.pth'))
    torch.save(train_data,
               os.path.join(args.output, args.save_data + '.train.pth'))
    torch.save(valid_data,
               os.path.join(args.output, args.save_data + '.valid.pth'))
    torch.save(test_data,
               os.path.join(args.output, args.save_data + '.test.pth'))

    # Save the word vocab
    with open(os.path.join(args.output, args.save_data + '.vocab.pickle'), 'wb') as f:
        pickle.dump(id2w, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the label vocab
    with open(os.path.join(args.output, args.save_data + '.label.pickle'),
              'wb') as f:
        pickle.dump(id2label, f, protocol=pickle.HIGHEST_PROTOCOL)