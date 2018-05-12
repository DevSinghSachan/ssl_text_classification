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

        arg_string = \
        'env CUDA_VISIBLE_DEVICES={} PYTHONIOENCODING=utf-8 python main.py --corpus elec --model LSTMEncoder '\
        '--debug --save_data demo_fastText '\
        '--multi_gpu --input temp/elec_vat/data --output_path temp/elec_vat/model '\
        '--exp_name elec_clf_at_{}_vat_{}_ent '\
        '--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 40 --optim adam '\
        '--wbatchsize 2000 --wbatchsize_unlabel 2000 --eval_steps 700 --lstm_dropout 0.5 --word_dropout 0.5 '\
        '--beta1 0.0 --num_layers 1 --beta2 0.98 --scheduler ExponentialLR --gamma 0.99998 '\
        '--perturb_norm_length {} --lambda_entropy 1.0 --lambda_vat 1.0 --lambda_at 1.0 '\
        '--inc_unlabeled_loss --unlabeled_loss_type Unlabel'

        cmd_string = arg_string.format(gpu_id,
                                       arg_val[0],
                                       arg_val[0],
                                       arg_val[0])

        print(cmd_string)
        my_env = os.environ.copy()
        my_env['CUDA_VISIBLE_DEVICES'] = gpu_id
        subprocess.call(cmd_string.split(), shell=False)

    queue.put(None)


perturb_norm_length_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def serve(queue):
    for tuple_ in itertools.product(perturb_norm_length_list):
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