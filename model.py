from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F
import numpy as np
import os

import preprocess
import batch_utils
from rnn_decoder import StdRNNDecoder
from layers import embedded_dropout, LockedDropout

if torch.cuda.is_available():
    torch.cuda.set_device(0)


def _sequence_mask(sequence_length, max_len=None):
    sequence_length = Variable(
        torch.from_numpy(sequence_length).type(batch_utils.LONG_TYPE),
        requires_grad=False)
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand, requires_grad=False)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


class SequenceCriteria(nn.Module):
    """ SequenceCriteria takes the input sequence, targets and sequence lengths
        and computes CE Loss
    """

    def __init__(self, class_weight):
        super(SequenceCriteria, self).__init__()
        self.criteria = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, inputs, targets):
        # This is BxT, which is what we want!
        loss = self.criteria(inputs, targets)
        return loss


def _linear(in_sz, out_sz, unif):
    """
    This is a linear layer (MLP)
    :param in_sz: number of input dimensions
    :param out_sz: number of output dimensions
    :param unif: scalar to initialize the parameters
    :return:
    """
    l = nn.Linear(in_sz, out_sz)
    weight_init.xavier_uniform(l.weight.data)
    return l


def _append2seq(seq, modules):
    for module_ in modules:
        seq.add_module(str(module_), module_)


def binary_cross_entropy(x, y, smoothing=0., epsilon=1e-12):
    """Computes the averaged binary cross entropy.

  bce = y*log(x) + (1-y)*log(1-x)

  Args:
    x: The predicted labels.
    y: The true labels.
    smoothing: The label smoothing coefficient.

  Returns:
    The cross entropy.
  """
    y = y.float()
    if smoothing > 0:
        smoothing *= 2
        y = y * (1 - smoothing) + 0.5 * smoothing
    return -torch.mean(
        torch.log(x + epsilon) * y + torch.log(1.0 - x + epsilon) * (1 - y))


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        seq_in_size = config.d_hidden
        if config.brnn:
            seq_in_size *= 2
        # if config.down_projection:
        #     seq_in_size = config.d_down_proj
        layers = [nn.Dropout(0.3),
                  nn.Linear(seq_in_size, 1024),
                  nn.LeakyReLU()]
        for _ in range(config.num_discriminator_layers - 1):
            layers.append(nn.Dropout(0.3))
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(1024, 1))
        self.model = nn.Sequential(*layers)

    # def forward(self, x, sequence_lengths, ids_, smoothing=0.0):
    #     B, T, D = x.shape
    #     y = self.model(x)
    #     y = y.view(B, T)
    #     mask = _sequence_mask(sequence_lengths)
    #     y = F.logsigmoid(y) * mask.float()
    #     y = torch.sum(y, dim=1)
    #     y = torch.exp(y)
    #
    #     loss = binary_cross_entropy(y, ids_, smoothing=smoothing)
    #     return loss
    def forward(self, x, sequence_lengths, ids_, smoothing=0.0):
        B, T, D = x.shape
        y = self.model(torch.max(x.transpose(1, 2).contiguous(), 2)[0])
        y = F.sigmoid(y)
        loss = binary_cross_entropy(y, ids_, smoothing=smoothing)
        return loss


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        seq_in_size = config.d_hidden
        if config.brnn:
            seq_in_size *= 2
        if config.down_projection:
            self.down_projection = _linear(seq_in_size,
                                           config.d_down_proj,
                                           config.init_scalar)
            self.act = nn.ReLU()
            seq_in_size = config.d_down_proj
        self.clf = _linear(seq_in_size,
                           config.num_classes,
                           config.init_scalar)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        if self.config.pool_type == "max_pool":
            sent_output = torch.max(x, 2)[0]
        elif self.config.pool_type == "avg_pool":
            normalize = 1. / np.sqrt(self.max_sent_len)
            sent_output = torch.sum(x, 2).mul_(normalize)
        if self.config.down_projection:
            sent_output = self.act(self.down_projection(sent_output))
        logits = self.clf(sent_output)
        return logits


class LstmPadding(object):
    def __init__(self, sent, sent_len, config):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)

        # Sort by length (keep idx)
        self.batch_size = len(sent_len)
        self.max_sent_len = max(sent_len)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        self.idx_unsort = np.argsort(idx_sort)
        self.config = config

        idx_sort = torch.from_numpy(idx_sort).type(batch_utils.LONG_TYPE)
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        self.sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)

    def __call__(self, lstm_enc_func):
        # Un-sort by length
        idx_unsort = torch.from_numpy(self.idx_unsort). \
            type(batch_utils.LONG_TYPE)
        memory_bank, enc_final = lstm_enc_func(self.sent_packed,
                                               self.batch_size)

        enc_final = enc_final[0].index_select(1, Variable(idx_unsort)), \
                    enc_final[1].index_select(1, Variable(idx_unsort))

        memory_bank = nn.utils.rnn.pad_packed_sequence(memory_bank)[0]
        memory_bank = memory_bank.index_select(1, Variable(idx_unsort))
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        return memory_bank, enc_final


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.encoder_input_size,
                           hidden_size=config.d_hidden,
                           num_layers=config.num_layers,
                           dropout=config.lstm_dropout,
                           bidirectional=config.brnn)

    def forward(self, inputs, batch_size):
        memory_bank, encoder_final = self.rnn(inputs)
        return memory_bank, encoder_final


class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()
        self.config = config
        self.embedder = nn.Embedding(config.n_vocab,
                                     config.d_units,
                                     padding_idx=preprocess.Vocab_Pad.PAD)
                                     # max_norm=config.max_embedding_norm)
        if config.use_pretrained_embeddings:
            print("Loading pre-trained word vectors")
            embeddings = np.load(os.path.join(config.input,
                                              config.save_data +
                                              ".word_vectors.npy")).astype(np.float32)
            self.embedder.weight = torch.nn.Parameter(torch.from_numpy(embeddings),
                                                      requires_grad=config.train_embeddings)
        if config.adaptive_dropout:
            # This is a form of variational dropout, which drops the same words
            # in each minibatch
            self.word_dropout = LockedDropout(dropout=config.locked_dropout)
        else:
            self.word_dropout = nn.Dropout(p=config.word_dropout)

    def _normalize(self, emb):
        weights = self.vocab_freqs / torch.sum(self.vocab_freqs)
        weights = weights.unsqueeze(-1)
        mean = torch.sum(weights * emb, 0, keepdim=True)
        var = torch.sum(weights * torch.pow(emb - mean, 2.), 0, keepdim=True)
        stddev = torch.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    def forward(self, batch):
        # Normalize word embedding params
        if self.config.normalize_embedding:
            self.embedder.weight.data = self._normalize(self.embedder.weight.data)

        if self.config.adaptive_dropout:
            word_embedding = embedded_dropout(self.embedder,
                                          batch.word_ids,
                                          dropout=self.config.word_dropout
                                          if self.training else 0)
            dropped = self.word_dropout(word_embedding.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
        else:
            word_embedding = self.embedder(batch.word_ids)
            dropped = self.word_dropout(word_embedding)
        return dropped


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        if config.projection:
            self.projection = nn.Linear(config.d_units, config.d_proj)
            self.act1 = nn.ReLU()
        config.encoder_input_size = config.d_proj \
            if config.projection else config.d_units

        self.lstm_encoder = Encoder(config)
        seq_in_size = config.d_hidden
        if config.brnn:
            seq_in_size *= 2
        self.lstm_dropout = nn.Dropout(config.lstm_dropout)
        self.config = config

    def encode_sent(self, embedded, sent_len):
        if self.config.projection:
            embedded = self.act1(self.projection(embedded))
        memory_bank, encoder_final = LstmPadding(embedded,
                                                 sent_len,
                                                 self.config)(self.lstm_encoder)
        return memory_bank, encoder_final

    def forward(self, embedded, batch, *args, **kwargs):
        memory_bank, encoder_final = self.encode_sent(embedded,
                                                      batch.sent_len)
        memory_bank = self.lstm_dropout(memory_bank)
        return memory_bank, encoder_final


def seq_func(func, x, reconstruct_shape=True, pad_remover=None):
    """Change implicitly function's input x from ndim=3 to ndim=2

    :param func: function to be applied to input x
    :param x: Tensor of batched sentence level word features
    :param reconstruct_shape: boolean, if the output needs to be
    of the same shape as input x
    :return: Tensor of shape (batchsize, dimension, sentence_length)
    or (batchsize x sentence_length, dimension)
    """
    batch, length, units = x.shape
    e = x.view(batch * length, units)
    if pad_remover:
        e = pad_remover.remove(e)
    e = func(e)
    if pad_remover:
        e = pad_remover.restore(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = e.view(batch, length, out_units)
    assert (e.shape == (batch, length, out_units))
    return e


class AEModel(nn.Module):
    """Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
    """

    def __init__(self, config):
        super(AEModel, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_vocab,
                                  config.d_units,
                                  padding_idx=preprocess.Vocab_Pad.PAD)
        self.decoder = StdRNNDecoder(rnn_type='LSTM',
                                     bidirectional_encoder=True,
                                     num_layers=1,
                                     hidden_size=512,
                                     dropout=0.2,
                                     embeddings=self.embed,
                                     attn_type="general")
        self.affine = nn.Linear(512,
                                config.n_vocab,
                                bias=True)
        weight = torch.ones(config.n_vocab)
        weight[preprocess.Vocab_Pad.PAD] = 0
        self.criterion = nn.NLLLoss(weight,
                                    size_average=False)

    def output_and_loss(self, h_block, t_block):
        batch, length, units = h_block.shape
        # shape : (batch * sequence_length, num_classes)
        logits_flat = seq_func(self.affine,
                               h_block,
                               reconstruct_shape=False)
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = F.log_softmax(logits_flat,
                                       dim=-1)
        rebatch, _ = logits_flat.shape
        concat_t_block = t_block.view(rebatch)
        weights = (concat_t_block >= 1).float()

        loss = self.criterion(log_probs_flat,
                              concat_t_block)
        loss = loss.sum() / (weights.sum() + 1e-13)
        return loss

    def forward(self, memory_bank, enc_final, lengths, ly_batch_raw,
                dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt_in_block, tgt_out_block = batch_utils.seq2seq_pad_concat(
            ly_batch_raw, -1)
        tgt_in_block = tgt_in_block.transpose(0, 1).contiguous()
        # tgt = tgt[:-1]  # exclude last target from inputs
        memory_bank = memory_bank.transpose(0, 1).contiguous()
        lengths = torch.from_numpy(lengths).type(batch_utils.LONG_TYPE)
        enc_state = self.decoder.init_decoder_state(enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt_in_block,
                                                         memory_bank,
                                                         enc_state if dec_state is None
                                                         else dec_state,
                                                         memory_lengths=lengths)
        decoder_outputs = decoder_outputs.transpose(0, 1).contiguous()
        loss = self.output_and_loss(decoder_outputs, tgt_out_block)
        return loss


def at_loss(embedder, encoder, clf, batch, perturb_norm_length=5.0):
    embedded = embedder(batch)
    embedded.retain_grad()
    ce = F.cross_entropy((clf(encoder(embedded, batch)[0])), batch.labels)
    ce.backward()

    d = embedded.grad.data.transpose(0, 1).contiguous()
    d = get_normalized_vector(d)
    d = d.transpose(0, 1).contiguous()

    d = embedder(batch) + (perturb_norm_length * Variable(d))
    loss = F.cross_entropy(clf(encoder(d, batch)[0]), batch.labels)
    return loss


def get_normalized_vector(d):
    B, T, D = d.shape
    d = d.view(B, -1)
    d /= (1e-12 + torch.max(torch.abs(d), dim=1, keepdim=True)[0])
    # d /= (1e-12 + torch.max(torch.abs(d), dim=1, keepdim=True)[0])

    d /= torch.sqrt(1e-6 + torch.sum(d**2, dim=1, keepdim=True))
    # d /= torch.sqrt(1e-6 + torch.sum(d**2, dim=1, keepdim=True))
    d = d.view(B, T, D)
    return d


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) -
                         F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl) # F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def vat_loss(embedder, encoder, clf, batch, perturb_norm_length=5.0,
             small_constant_for_finite_diff=1e-1, Ip=1, p_logit=None):
    embedded = embedder(batch)
    d = torch.randn(embedded.shape).type(batch_utils.FLOAT_TYPE)
    d = d.transpose(0, 1).contiguous()
    d = get_normalized_vector(d).transpose(0, 1).contiguous()
    for ip in range(Ip):
        x_d = Variable(embedded.data + (small_constant_for_finite_diff * d),
                       requires_grad=True)
        x_d.retain_grad()
        p_d_logit = clf(encoder(x_d, batch)[0])
        kl_loss = kl_categorical(Variable(p_logit.data), p_d_logit)
        kl_loss.backward()
        d = x_d.grad.data.transpose(0, 1).contiguous()
        d = get_normalized_vector(d).transpose(0, 1).contiguous()
    x_adv = embedded + (perturb_norm_length * Variable(d))
    p_adv_logit = clf(encoder(x_adv, batch)[0])
    return kl_categorical(Variable(p_logit.data), p_adv_logit)


def entropy_loss(p_logit):
    p = F.softmax(p_logit, dim=-1)
    return -1 * torch.sum(p * F.log_softmax(p_logit, dim=-1)) / p_logit.size()[0]

