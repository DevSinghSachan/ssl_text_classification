import json
import pickle
import os
import torch

from config import get_train_args
from training import Training
from general_utils import get_logger


args = get_train_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
logger = get_logger(args.log_path)
logger.info(json.dumps(args.__dict__, indent=4))

# Reading the int indexed text dataset
train_data = torch.load(os.path.join(args.input, args.save_data + ".train.pth"))
dev_data = torch.load(os.path.join(args.input, args.save_data + ".valid.pth"))
test_data = torch.load(os.path.join(args.input, args.save_data + ".test.pth"))
unlabel_data = torch.load(os.path.join(args.input, args.save_data + ".unlabel.pth"))

# Reading the word vocab file
with open(os.path.join(args.input, args.save_data + '.vocab.pickle'),
          'rb') as f:
    id2w = pickle.load(f)

# Reading the label vocab file
with open(os.path.join(args.input, args.save_data + '.label.pickle'),
          'rb') as f:
    id2label = pickle.load(f)

args.id2w = id2w
args.n_vocab = len(id2w)
args.id2label = id2label
args.num_classes = len(id2label)

object = Training(args, logger)

logger.info('Corpus: {}'.format(args.corpus))
logger.info('Pytorch Model')
logger.info(repr(object.embedder))
logger.info(repr(object.encoder))
logger.info(repr(object.clf))
logger.info(repr(object.clf_loss))
if args.lambda_ae:
    logger.info(repr(object.ae))
if args.lambda_dis:
    logger.info(repr(object.dis))

# Train the model
object(train_data, dev_data, test_data, unlabel_data)
