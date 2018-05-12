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
        'env CUDA_VISIBLE_DEVICES={} PYTHONIOENCODING=utf-8 python main.py --corpus dbpedia --model LSTMEncoder '\
        '--debug --save_data demo_fastText '\
        '--multi_gpu --input temp/dbpedia_pretrained/data --output_path temp/dbpedia_pretrained/model '\
        '--exp_name dbpedia_clf_at_{} '\
        '--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed --d_hidden 512 --nepochs 30 --optim adam '\
        '--wbatchsize 2000 --wbatchsize_unlabel 2000 --eval_steps 1000 --lstm_dropout 0.5 --word_dropout 0.5 '\
        '--beta1 0.0 --num_layers 1 --beta2 0.98 --scheduler ExponentialLR --gamma 0.999998 '\
        '--perturb_norm_length {} --lambda_entropy 0.0 --lambda_vat 0.0 --lambda_at 1.0 '

        cmd_string = arg_string.format(gpu_id,
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