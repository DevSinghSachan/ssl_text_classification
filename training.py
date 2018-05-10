from __future__ import print_function, division

import os
import logging
import collections
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD, RMSprop, lr_scheduler
from helpers.ensure_directory import ensure_directory
from torchtext import data
import itertools
from time import time

import batch_utils
import utils
from model import LSTMEncoder, Classifier, Discriminator, SequenceCriteria, \
    AEModel, at_loss, Embedder, vat_loss, entropy_loss
from tensorboardX import SummaryWriter
import pprint
from metrics import ConfusionMatrix
from exp_moving_avg import ExponentialMovingAverage

if torch.cuda.is_available():
    torch.cuda.set_device(0)

global max_src_in_batch


def add_noise_python(words, dropout=0.1, k=3):
    """Applies the noise model in input words.

  Args:
    words: A numpy vector of word ids.
    dropout: The probability to drop words.
    k: Maximum distance of the permutation.

  Returns:
    A noisy numpy vector of word ids.
  """

    def _drop_words(words, probability):
        """Drops words with the given probability."""
        length = len(words)
        keep_prob = np.random.uniform(size=length)
        keep = np.random.uniform(size=length) > probability
        if np.count_nonzero(keep) == 0:
            ind = np.random.randint(0, length)
            keep[ind] = True
        words = np.take(words, keep.nonzero())[0]
        return words

    def _rand_perm_with_constraint(words, k):
        """Randomly permutes words ensuring that words are no more than k positions
    away from their original position."""
        length = len(words)
        offset = np.random.uniform(size=length) * (k + 1)
        new_pos = np.arange(length) + offset
        return np.take(words, np.argsort(new_pos))

    words = _drop_words(words, dropout)
    if k > 0:
        words = _rand_perm_with_constraint(words, k)
    return words


def add_noise(ids, noise_dropout=0.1, random_permutation=3):
    """Wraps add_noise_python for a batch of tensors."""

    def _add_noise_single(ids):
        noisy_ids = add_noise_python(ids, noise_dropout, random_permutation)
        noisy_sequence_length = len(noisy_ids)
        return noisy_ids

    noisy_ids = []
    for l, id_ in ids:
        noisy_ids.append((l, _add_noise_single(id_)))
    return noisy_ids


def batch_size_fn(new, count, sofar):
    global max_src_in_batch
    if count == 1:
        max_src_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new[1]) + 2)
    src_elements = count * max_src_in_batch
    return src_elements


def report_func(epoch, batch, num_batches, start_time, report_stats,
                report_every, logger):
    if batch % report_every == -1 % report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time, logger)


class Training(object):
    def __init__(self, config, logger=None):
        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        self.logger = logger
        self.config = config
        self.classes = list(config.id2label.keys())
        self.num_classes = config.num_classes

        self.embedder = Embedder(self.config)
        self.encoder = LSTMEncoder(self.config)
        self.clf = Classifier(self.config)
        self.clf_loss = SequenceCriteria(class_weight=None)
        if self.config.lambda_ae > 0: self.ae = AEModel(self.config)

        self.writer = SummaryWriter(log_dir="TFBoardSummary")
        self.global_steps = 0
        self.enc_clf_opt = Adam(self._get_trainabe_modules(),
                                lr=self.config.lr,
                                betas=(config.beta1,
                                       config.beta2),
                                weight_decay=config.weight_decay,
                                eps=config.eps)

        if config.scheduler == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.enc_clf_opt,
                                                            mode='max',
                                                            factor=config.lr_decay,
                                                            patience=config.patience,
                                                            verbose=True)
        elif config.scheduler == "ExponentialLR":
            self.scheduler = lr_scheduler.ExponentialLR(self.enc_clf_opt,
                                                        gamma=config.gamma)

        self._init_or_load_model()
        if config.multi_gpu:
            self.embedder.cuda()
            self.encoder.cuda()
            self.clf.cuda()
            self.clf_loss.cuda()
            if self.config.lambda_ae > 0: self.ae.cuda()

        self.ema_embedder = ExponentialMovingAverage(decay=0.999)
        self.ema_embedder.register(self.embedder.state_dict())
        self.ema_encoder = ExponentialMovingAverage(decay=0.999)
        self.ema_encoder.register(self.encoder.state_dict())
        self.ema_clf = ExponentialMovingAverage(decay=0.999)
        self.ema_clf.register(self.clf.state_dict())

        self.time_s = time()

    def _get_trainabe_modules(self):
        param_list = list(self.encoder.parameters()) + \
                     list(self.clf.parameters())
        if self.config.train_embeddings:
            param_list += list(self.embedder.parameters())
        if self.config.lambda_ae > 0:
            param_list += list(self.ae.parameters())
        return param_list

    def _get_l2_norm_loss(self):
        total_norm = 0.
        for p in self._get_trainabe_modules():
            param_norm = p.data.norm(p=2)
            total_norm += param_norm  # ** 2
        return total_norm  # / 2.

    def _init_or_load_model(self):
        # if not self._load_model():
        ensure_directory(self.config.output_path)
        self.epoch = 0
        self.best_accuracy = -np.inf

    def _compute_vocab_freq(self, train_, dev_):
        counter = collections.Counter()
        for _, ids_ in train_:
            counter.update(ids_)
        for _, ids_ in dev_:
            counter.update(ids_)
        word_freq = np.zeros(self.config.n_vocab)
        for index_, freq_ in counter.items():
            word_freq[index_] = freq_
        return torch.from_numpy(word_freq).type(batch_utils.FLOAT_TYPE)

    def _save_model(self):
        state = {'epoch': self.epoch,
                 'state_dict_encoder': self.ema_encoder.shadow_variable_dict,
                 # self.encoder.state_dict(),
                 'state_dict_embedder': self.ema_embedder.shadow_variable_dict,
                 # self.embedder.state_dict(),
                 'state_dict_clf': self.ema_clf.shadow_variable_dict,
                 # self.clf.state_dict(),
                 'best_accuracy': self.best_accuracy}
        torch.save(state, os.path.join(self.config.output_path,
                                       self.config.model_file))

    def _load_model(self):
        checkpoint_path = os.path.join(self.config.output_path,
                                       self.config.model_file)
        if self.config.load_checkpoint and os.path.isfile(checkpoint_path):
            # Code taken from here: https://github.com/pytorch/examples/blob/master/imagenet/main.py
            dict_ = torch.load(checkpoint_path)
            self.epoch = dict_['epoch']
            self.best_accuracy = dict_['best_accuracy']
            self.embedder.load_state_dict(dict_['state_dict_embedder'])
            self.encoder.load_state_dict(dict_['state_dict_encoder'])
            self.clf.load_state_dict(dict_['state_dict_clf'])
            self.logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path,
                                                              self.epoch))
            return True

    def __call__(self, train, dev, test, unlabel):
        if self.config.normalize_embedding:
            self.embedder.vocab_freqs = self._compute_vocab_freq(train, dev)
            print("Embeddings will be normalized during training")
        self.logger.info('Start training')
        self._train(train, dev, unlabel)
        self._evaluate(test)

    def _create_iter(self, data_, wbatchsize):
        if self.config.batching_strategy == "dynamic":
            iter_ = data.iterator.pool(data_,
                                       wbatchsize,
                                       key=lambda x: len(x[1]),
                                       batch_size_fn=batch_size_fn,
                                       random_shuffler=
                                       data.iterator.RandomShuffler())
            return iter_
        if self.config.batching_strategy == "fixed":
            iter_ = data.iterator.pool(data_,
                                       self.config.batchsize,
                                       key=lambda x: len(x[1]),
                                       random_shuffler=
                                       data.iterator.RandomShuffler())
            return iter_

    def _run_epoch(self, train_data, dev_data, unlabel_data):
        report_stats = utils.Statistics()
        cm = ConfusionMatrix(self.classes)
        _, seq_data = list(zip(*train_data))
        total_seq_words = len(list(itertools.chain.from_iterable(seq_data)))
        iter_per_epoch = (1.5 * total_seq_words) // self.config.wbatchsize

        self.encoder.train()
        self.clf.train()
        self.embedder.train()
        train_iter = self._create_iter(train_data, self.config.wbatchsize)
        unlabel_iter = self._create_iter(dev_data,
                                         self.config.wbatchsize_unlabel)
        for batch_index, train_batch_raw in enumerate(train_iter):
            seq_iter = list(zip(*train_batch_raw))[1]
            seq_words = len(list(itertools.chain.from_iterable(seq_iter)))
            report_stats.n_words += seq_words
            self.global_steps += 1

            # self.enc_clf_opt.zero_grad()
            if self.config.add_noise:
                train_batch_raw = add_noise(train_batch_raw,
                                            self.config.noise_dropout,
                                            self.config.random_permutation)
            train_batch = batch_utils.seq_pad_concat(train_batch_raw, -1)

            train_embedded = self.embedder(train_batch)
            memory_bank_train, enc_final_train = self.encoder(train_embedded,
                                                              train_batch)

            if self.config.lambda_vat > 0 or self.config.lambda_ae > 0 or self.config.lambda_entropy:
                try:
                    unlabel_batch_raw = next(unlabel_iter)
                except StopIteration:
                    unlabel_iter = self._create_iter(unlabel_data,
                                                     self.config.wbatchsize_unlabel)
                    unlabel_batch_raw = next(unlabel_iter)

                if self.config.add_noise:
                    unlabel_batch_raw = add_noise(unlabel_batch_raw,
                                                  self.config.noise_dropout,
                                                  self.config.random_permutation)
                unlabel_batch = batch_utils.seq_pad_concat(unlabel_batch_raw,
                                                           -1)
                unlabel_embedded = self.embedder(unlabel_batch)
                memory_bank_unlabel, enc_final_unlabel = self.encoder(
                    unlabel_embedded,
                    unlabel_batch)

            pred = self.clf(memory_bank_train)
            accuracy = self.get_accuracy(cm, pred.data, train_batch.labels.data)
            lclf = self.clf_loss(pred, train_batch.labels)

            lat = Variable(
                torch.FloatTensor([-1.]).type(batch_utils.FLOAT_TYPE))
            lvat = Variable(
                torch.FloatTensor([-1.]).type(batch_utils.FLOAT_TYPE))
            if self.config.lambda_at > 0:
                lat = at_loss(self.embedder,
                              self.encoder,
                              self.clf,
                              train_batch,
                              perturb_norm_length=self.config.perturb_norm_length)

            if self.config.lambda_vat > 0:
                lvat_train = vat_loss(self.embedder,
                                      self.encoder,
                                      self.clf,
                                      train_batch,
                                      p_logit=pred,
                                      perturb_norm_length=self.config.perturb_norm_length)
                if self.config.inc_unlabeled_loss:
                    lvat_unlabel = vat_loss(self.embedder,
                                            self.encoder,
                                            self.clf,
                                            unlabel_batch,
                                            p_logit=self.clf(memory_bank_unlabel),
                                            perturb_norm_length=self.config.perturb_norm_length)
                    if self.config.unlabeled_loss_type == "AvgTrainUnlabel":
                        lvat = 0.5 * (lvat_train + lvat_unlabel)
                    elif self.config.unlabeled_loss_type == "Unlabel":
                        lvat = lvat_unlabel
                else:
                    lvat = lvat_train

            lentropy = Variable(torch.FloatTensor([-1.]).type(batch_utils.FLOAT_TYPE))
            if self.config.lambda_entropy > 0:
                lentropy_train = entropy_loss(pred)
                if self.config.inc_unlabeled_loss:
                    lentropy_unlabel = entropy_loss(self.clf(memory_bank_unlabel))
                    if self.config.unlabeled_loss_type == "AvgTrainUnlabel":
                        lentropy = 0.5 * (lentropy_train + lentropy_unlabel)
                    elif self.config.unlabeled_loss_type == "Unlabel":
                        lentropy = lentropy_unlabel
                else:
                    lentropy = lentropy_train

            lae = Variable(torch.FloatTensor([-1.]).type(batch_utils.FLOAT_TYPE))
            if self.config.lambda_ae > 0:
                lae = self.ae(memory_bank_unlabel,
                              enc_final_unlabel,
                              unlabel_batch.sent_len,
                              unlabel_batch_raw)

            ltotal = (self.config.lambda_clf * lclf) + \
                     (self.config.lambda_ae * lae) + \
                     (self.config.lambda_at * lat) + \
                     (self.config.lambda_vat * lvat) + \
                     (self.config.lambda_entropy * lentropy)

            report_stats.clf_loss += lclf.data.cpu().numpy()
            report_stats.at_loss += lat.data.cpu().numpy()
            report_stats.vat_loss += lvat.data.cpu().numpy()
            report_stats.ae_loss += lae.data.cpu().numpy()
            report_stats.entropy_loss += lentropy.data.cpu().numpy()
            report_stats.n_sent += len(pred)
            report_stats.n_correct += accuracy
            self.enc_clf_opt.zero_grad()
            ltotal.backward()

            params_list = self._get_trainabe_modules()
            # Excluding embedder form norm constraint when AT or VAT
            if not self.config.normalize_embedding:
                params_list += list(self.embedder.parameters())

            norm = torch.nn.utils.clip_grad_norm(params_list,
                                                 self.config.max_norm)
            report_stats.grad_norm += norm
            self.enc_clf_opt.step()
            if self.config.scheduler == "ExponentialLR":
                self.scheduler.step()
            self.ema_embedder.apply(self.embedder.named_parameters())
            self.ema_encoder.apply(self.encoder.named_parameters())
            self.ema_clf.apply(self.clf.named_parameters())

            report_func(self.epoch,
                        batch_index,
                        iter_per_epoch,
                        self.time_s,
                        report_stats,
                        self.config.report_every,
                        self.logger)

            if self.global_steps % self.config.eval_steps == 0:
                cm_, accuracy, prc_dev = self._run_evaluate(dev_data)
                self.logger.info("- dev accuracy {} | best dev accuracy {} ".format(accuracy, self.best_accuracy))
                self.writer.add_scalar("Dev_Accuracy", accuracy,
                                       self.global_steps)
                pred_, lab_ = zip(*prc_dev)
                pred_ = torch.cat(pred_)
                lab_ = torch.cat(lab_)
                self.writer.add_pr_curve("Dev PR-Curve", lab_,
                                         pred_,
                                         self.global_steps)
                pprint.pprint(cm_)
                pprint.pprint(cm_.get_all_metrics())
                if accuracy > self.best_accuracy:
                    self.logger.info("- new best score!")
                    self.best_accuracy = accuracy
                    self._save_model()
                if self.config.scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(accuracy)
                self.encoder.train()
                self.embedder.train()
                self.clf.train()

                if self.config.weight_decay > 0:
                    print(">> Square Norm: %1.4f " % self._get_l2_norm_loss())

        cm, train_accuracy, _ = self._run_evaluate(train_data)
        self.logger.info("- Train accuracy  {}".format(train_accuracy))
        pprint.pprint(cm.get_all_metrics())

        cm, dev_accuracy, _ = self._run_evaluate(dev_data)
        self.logger.info("- Dev accuracy  {} | best dev accuracy {}".format(dev_accuracy, self.best_accuracy))
        pprint.pprint(cm.get_all_metrics())
        self.writer.add_scalars("Overall_Accuracy",
                                {"Train_Accuracy": train_accuracy,
                                 "Dev_Accuracy": dev_accuracy},
                                self.global_steps)
        return dev_accuracy

    @staticmethod
    def get_accuracy(cm, output, target):
        batch_size = output.size(0)
        predictions = output.max(-1)[1].type_as(target)
        correct = predictions.eq(target)
        correct = correct.float()
        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        correct = correct.sum()
        cm.add_batch(target.cpu().numpy(), predictions.cpu().numpy())
        return correct

    def _predict_batch(self, cm, batch):
        self.embedder.eval()
        self.encoder.eval()
        self.clf.eval()
        pred = self.clf(self.encoder(self.embedder(batch),
                                     batch)[0])
        accuracy = self.get_accuracy(cm, pred.data, batch.labels.data)
        return pred, accuracy

    def _run_evaluate(self, test_data):
        pr_curve_data = []
        cm = ConfusionMatrix(self.classes)
        accuracy_list = []
        test_iter = self._create_iter(test_data, self.config.wbatchsize)
        for test_batch in test_iter:
            test_batch = batch_utils.seq_pad_concat(test_batch, -1)
            pred, acc = self._predict_batch(cm, test_batch)
            accuracy_list.append(acc)
            pr_curve_data.append(
                (F.softmax(pred, -1)[:, 1].data, test_batch.labels.data))
        accuracy = 100 * (sum(accuracy_list) / len(test_data))
        return cm, accuracy, pr_curve_data

    def _train(self, train_data, dev_data, unlabel_data):
        # for early stopping
        nepoch_no_imprv = 0

        epoch_start = self.epoch + 1
        epoch_end = self.epoch + self.config.nepochs + 1
        for self.epoch in range(epoch_start, epoch_end):
            self.logger.info(
                "Epoch {:} out of {:}".format(self.epoch, self.config.nepochs))
            random.shuffle(train_data)
            random.shuffle(unlabel_data)
            accuracy = self._run_epoch(train_data, dev_data, unlabel_data)

            # early stopping and saving best parameters
            if accuracy > self.best_accuracy:
                nepoch_no_imprv = 0
                self.best_accuracy = accuracy
                self.logger.info("- new best score!")
                self._save_model()
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info(
                        "- early stopping {} epochs without improvement".format(
                            nepoch_no_imprv))
                    break
            if self.config.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(accuracy)

    def _evaluate(self, test_data):
        self.logger.info("Evaluating model over test set")
        self._load_model()
        _, accuracy, _ = self._run_evaluate(test_data)
        self.logger.info("- test accuracy {}".format(accuracy))
