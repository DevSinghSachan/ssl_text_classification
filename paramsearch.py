import os
import sys
from multiprocessing import Process, Queue
import subprocess
import itertools

num_gpus = int(sys.argv[1])


def work(id, queue):
    gpu_id = str(id)

    while True:
        arg_val = queue.get()

        if arg_val is None:
            break

        model_file = 'model_{}'.format(gpu_id)

        arg_string = 'env CUDA_VISIBLE_DEVICES={} python main.py --corpus '\
                     'aclImdb --model LSTMEncoder --multi_gpu --input temp/aclImdb_pretrained/data '\
                      '--output_path temp/aclImdb_pretrained/model --timedistributed '\
                      '--pool_type max_pool --d_hidden 512 --nepochs 50 '\
                      '--nepoch_no_imprv 20 --optim adam --adaptive_dropout '\
                      '--lstm_dropout 0.0 --word_dropout {} --locked_dropout {} --model_file {}'

        # arg_string = 'env CUDA_VISIBLE_DEVICES={} python main.py --model {} --corpus {} --dropout {} --weight_decay {} ' \
        #              '--lr 0.005 --lr_decay 0.9998 --model_file {}'

        cmd_string = arg_string.format(gpu_id,
                                       arg_val[0],
                                       arg_val[1],
                                       model_file)

        print(cmd_string)
        my_env = os.environ.copy()
        my_env['CUDA_VISIBLE_DEVICES'] = gpu_id

        subprocess.call(cmd_string.split(), shell=False)

    queue.put(None)


word_dropout = [0, 0.15, 0.25, 0.4, 0.5, 0.75, 0.85]
locked_dropout = [0, 0.15, 0.25, 0.4, 0.5, 0.75, 0.85]


def serve(queue):
    for tuple_ in itertools.product(word_dropout,
                                    locked_dropout):
        queue.put(tuple_)


# https://stackoverflow.com/questions/914821/producer-consumer-problem-with-python-multiprocessing
class Manager:
    def __init__(self):
        self.queue = Queue()
        self.num_process = num_gpus

    def start(self):
        print("starting %d workers" % self.num_process)
        self.workers = [Process(target=work, args=(i, self.queue,))
                        for i in range(self.num_process)]
        for w in self.workers:
            w.start()

        serve(self.queue)

    def stop(self):
        self.queue.put(None)
        for i in range(self.num_process):
            self.workers[i].join()
        self.queue.close()


if __name__ == '__main__':
    m = Manager()
    m.start()
    m.stop()