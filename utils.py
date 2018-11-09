import time
import sys


def identity_fun(x):
    return x


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:
    * accuracy
    * perplexity
    * elapsed time
    Code adapted from OpenNMT-py open-source toolkit on 10/01/2018:
    URL: https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self):
        self.clf_loss = 0
        self.ae_loss = 0.
        self.at_loss = 0.
        self.vat_loss = 0.
        self.entropy_loss = 0.
        self.n_words = 0
        self.n_correct = 0
        self.n_sent = 0
        self.grad_norm = 0
        self.start_time = time.time()

    def accuracy(self):
        return 100 * (self.n_correct / self.n_sent)

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start, logger):
        """Write out statistics to stdout.
        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        logger.info(("Epoch %2d, %5d/%5d; "
                     "acc: %6.2f; "
                     "clf_loss: %1.4f; "
                     "at_loss: %1.4f; "
                     "vat_loss: %1.4f; "
                     "entropy_loss: %1.4f; "
                     "ae_loss: %1.4f; "
                     "norm: %2.4f; "
                     "%3.0f tok/s; "
                     "%6.0f s elapsed") %
                    (epoch,
                     batch,
                     n_batches,
                     self.accuracy(),
                     self.clf_loss / (batch + 1),
                     self.at_loss / (batch + 1),
                     self.vat_loss / (batch + 1),
                     self.entropy_loss / (batch + 1),
                     self.ae_loss / (batch + 1),
                     self.grad_norm / (batch + 1),
                     self.n_words / (t + 1e-5),
                     time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper", self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)
