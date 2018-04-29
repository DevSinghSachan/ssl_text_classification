import torch
from torch.autograd import Variable
import collections
import numpy as np
from chainer.dataset import convert

import preprocess

if torch.cuda.is_available():
    FLOAT_TYPE = torch.cuda.FloatTensor
    INT_TYPE = torch.cuda.IntTensor
    LONG_TYPE = torch.cuda.LongTensor
    BYTE_TYPE = torch.cuda.ByteTensor
else:
    FLOAT_TYPE = torch.FloatTensor
    INT_TYPE = torch.IntTensor
    LONG_TYPE = torch.LongTensor
    BYTE_TYPE = torch.ByteTensor

Batch = collections.namedtuple('Batch', ['batch_size',
                                         'labels',
                                         'word_ids',
                                         'sent_len'])


def seq_pad_concat(batch, device):
    labels, word_ids = zip(*batch)

    block_w = convert.concat_examples(word_ids,
                                      device,
                                      padding=preprocess.Vocab_Pad.PAD)

    sent_len = np.array(list(map(lambda x: len(x), word_ids)))
    # Converting from numpy format to Torch Tensor
    block_w = Variable(torch.LongTensor(block_w).type(LONG_TYPE),
                       requires_grad=False)
    labels = Variable(torch.LongTensor(labels).type(LONG_TYPE),
                      requires_grad=False)

    return Batch(batch_size=len(labels),
                 word_ids=block_w.transpose(0, 1).contiguous(),
                 labels=labels,
                 sent_len=sent_len)


def seq2seq_pad_concat(ly_batch,
                       device,
                       eos_id=preprocess.Vocab_Pad.EOS,
                       bos_id=preprocess.Vocab_Pad.BOS):
    labels, y_seqs = zip(*ly_batch)
    y_block = convert.concat_examples(y_seqs, device, padding=0)

    y_out_block = np.pad(y_block, ((0, 0), (0, 1)), 'constant',
                         constant_values=0)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id
    y_in_block = np.pad(y_block, ((0, 0), (1, 0)), 'constant',
                        constant_values=bos_id)

    # Converting from numpy format to Torch Tensor
    y_in_block = Variable(torch.LongTensor(y_in_block).type(LONG_TYPE),
                          requires_grad=False)
    y_out_block = Variable(torch.LongTensor(y_out_block).type(LONG_TYPE),
                           requires_grad=False)
    return y_in_block, y_out_block
