from tqdm import tqdm
import torch
from collections import Counter
import numpy as np
import unicodedata
from collections import defaultdict
from sklearn.utils import compute_class_weight
import os
import math
import random
import string
import io


def long_0_tensor_alloc(nelements, dtype=None):
    lt = long_tensor_alloc(nelements)
    lt.zero_()
    return lt


def long_tensor_alloc(dims, dtype=None):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(*dims)


class SeqLabelReader(object):
    def __init__(self):
        pass

    def build_vocab(self, files, **kwargs):
        pass

    def load(self, filename, index, batchsz, **kwargs):
        pass


class TSVSeqLabelReader(SeqLabelReader):
    def __init__(self, mxlen=1000, mxfiltsz=0, vec_alloc=np.zeros):
        super(TSVSeqLabelReader, self).__init__()

        self.vocab = None
        self.label2index = {}
        self.mxlen = mxlen
        self.mxfiltsz = mxfiltsz
        self.vec_alloc = vec_alloc

    @staticmethod
    def splits(text):
        return text.lower().split()

    @staticmethod
    def label_and_sentence(line):
        label_text = line.strip().lower().split('\t')
        label = label_text[0]
        text = label_text[1:]
        text = ' '.join(text)
        return label, text

    def build_vocab(self, files, **kwargs):
        """Take a directory (as a string), or an array of files and build a vocabulary

        Take in a directory or an array of individual files (as a list).  If the argument is
        a string, it may be a directory, in which case, all files in the directory will be loaded
        to form a vocabulary.

        :param files: Either a directory (str), or an array of individual files
        :return:
        """
        label_idx = len(self.label2index)
        if type(files) == str:
            if os.path.isdir(files):
                base = files
                files = filter(os.path.isfile, [os.path.join(base, x) for x in os.listdir(base)])
            else:
                files = [files]

        y = list()
        vocab = Counter()
        for file in files:
            if file is None:
                continue
            with io.open(file, encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f):
                    label, text = TSVSeqLabelReader.label_and_sentence(line)
                    if label not in self.label2index:
                        self.label2index[label] = label_idx
                        label_idx += 1
                    for w in TSVSeqLabelReader.splits(text):
                        vocab[w] += 1
                    y.append(self.label2index[label])

        if kwargs.get("class_weight") == "balanced":
            class_weight = compute_class_weight("balanced", list(self.label2index.values()), y)
        else:
            class_weight = None

        return vocab, self.get_labels(), class_weight

    def get_labels(self):
        labels = [''] * len(self.label2index)
        for label, index in self.label2index.items():
            labels[index] = label
        return labels

    def load(self, filename, index, batchsz, **kwargs):
        PAD = index['<PAD>']
        shuffle = kwargs.get('shuffle', False)
        halffiltsz = self.mxfiltsz // 2
        nozplen = self.mxlen - 2 * halffiltsz

        examples = []
        with io.open(filename, encoding='utf-8', errors='ignore') as f:
            for offset, line in enumerate(tqdm(f)):
                label, text = TSVSeqLabelReader.label_and_sentence(line)
                y = self.label2index[label]
                toks = TSVSeqLabelReader.splits(text)
                mx = min(len(toks), nozplen)
                toks = toks[:mx]
                x = self.vec_alloc(self.mxlen, dtype=int)
                for j in range(len(toks)):
                    w = toks[j]
                    key = index.get(w, PAD)
                    x[j + halffiltsz] = key
                examples.append((x, y))

        return SeqLabelDataFeed(SeqLabelExamples(examples),
                                batchsz=batchsz,
                                shuffle=shuffle,
                                vec_alloc=self.vec_alloc,
                                src_vec_trans=None)


class SeqLabelExamples(object):
    """Unstructured prediction examples

    Datasets of paired `(x, y)` data, where `x` is a tensor of data over time and `y` is a single label
    """
    SEQ = 0
    LABEL = 1

    def __init__(self, example_list, do_shuffle=True):
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)

    def __getitem__(self, i):
        """Get a single example

        :param i: (``int``) simple index
        :return: an example
        """
        ex = self.example_list[i]
        return ex[SeqLabelExamples.SEQ], ex[SeqLabelExamples.LABEL]

    def __len__(self):
        """Number of examples

        :return: (``int``) length of data
        """
        return len(self.example_list)

    def width(self):
        """ Width of the temporal signal

        :return: (``int``) length
        """
        x, y = self.example_list[0]
        return len(x)

    def batch(self, start, batchsz, vec_alloc=np.empty):
        """Get a batch of data

        :param start: The step index
        :param batchsz: The batch size
        :param vec_alloc: A vector allocator, defaults to `numpy.empty`
        :return: batched x vector, batched y vector
        """
        siglen = self.width()
        xb = vec_alloc((batchsz, siglen), dtype=np.int)
        yb = vec_alloc((batchsz), dtype=np.int)
        sz = len(self.example_list)
        idx = start * batchsz
        for i in range(batchsz):
            if idx >= sz:
                # idx = 0
                batchsz = i
                break
            x, y = self.example_list[idx]
            xb[i] = x
            yb[i] = y
            idx += 1
        return xb[: batchsz], yb[: batchsz]

    @staticmethod
    def valid_split(data, splitfrac=0.15):
        """Function to produce a split of data based on a fraction

        :param data: Data to split
        :param splitfrac: (``float``) fraction of data to hold out
        :return: Two sets of label examples
        """
        numinst = len(data.examples)
        heldout = int(math.floor(numinst * (1 - splitfrac)))
        heldout_ex = data.example_list[1:heldout]
        rest_ex = data.example_list[heldout:]
        return SeqLabelExamples(heldout_ex), SeqLabelExamples(rest_ex)


class DataFeed(object):
    """Data collection that, when iterated, produces an epoch of data

    This class manages producing a dataset to the trainer, by iterating an epoch and producing
    a single step at a time.  The data can be shuffled per epoch, if requested, otherwise it is
    returned in the order of the dateset
    """

    def __init__(self):
        self.steps = 0
        self.shuffle = False

    def _batch(self, i):
        pass

    def __getitem__(self, i):
        return self._batch(i)

    def __iter__(self):
        shuffle = np.random.permutation(np.arange(self.steps)) if self.shuffle else np.arange(self.steps)

        for i in range(self.steps):
            si = shuffle[i]
            yield self._batch(si)

    def __len__(self):
        return self.steps


class ExampleDataFeed(DataFeed):
    """Abstract base class that works on a list of examples

    """

    def __init__(self, examples, batchsz, **kwargs):
        """Constructor from a list of examples

        Use the examples requested to provide data.  Options for batching and shuffling are supported,
        along with some optional processing function pointers

        :param examples: A list of examples
        :param batchsz: Batch size per step
        :param kwargs: See below

        :Keyword Arguments:
            * *shuffle* -- Shuffle the data per epoch? Defaults to `False`
            * *vec_alloc* -- Allocate a new tensor.  Defaults to ``numpy.zeros``
            * *vec_shape* -- Function to retrieve tensor shape.  Defaults to ``numpy.shape``
            * *trim* -- Trim batches to the maximum length seen in the batch (defaults to `False`)
                This can lead to batches being shorter than the maximum length provided to the system.
                Not supported in all frameworks.
            * *src_vec_trans* -- A transform function to use on the source tensor (`None`)
        """
        super(ExampleDataFeed, self).__init__()

        self.examples = examples
        self.batchsz = batchsz
        self.shuffle = bool(kwargs.get('shuffle', False))
        self.vec_alloc = kwargs.get('vec_alloc', np.zeros)
        self.vec_shape = kwargs.get('vec_shape', np.shape)
        self.src_vec_trans = kwargs.get('src_vec_trans', None)
        # self.steps = int(math.floor(len(self.examples) / float(batchsz)))
        self.steps = int(math.ceil(len(self.examples) / float(batchsz)))
        self.trim = bool(kwargs.get('trim', False))


class SeqLabelDataFeed(ExampleDataFeed):
    """Data feed for :class:`SeqLabelExamples`
    """
    def __init__(self, examples, batchsz, **kwargs):
        super(SeqLabelDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        """
        Get a batch of data at step `i`
        :param i: (``int``) step index
        :return: A batch tensor x, batch tensor y
        """
        x, y = self.examples.batch(i, self.batchsz, self.vec_alloc)
        if self.src_vec_trans is not None:
            x = self.src_vec_trans(x)
        return x, y
