from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from mimic3models import common_utils, metrics
import threading
import random
import os
import itertools
import pandas as pd



class BatchGen(object):

    def __init__(self, reader, discretizer, normalizer, batch_size,
                 small_part, target_repl, shuffle, return_names=False):
        self.batch_size = batch_size
        self.target_repl = target_repl
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_data(reader, discretizer, normalizer, small_part)

        self.steps = (len(self.data[0]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_data(self, reader, discretizer, normalizer, small_part=False):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        ys = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if (normalizer is not None):
            data = [normalizer.transform(X) for X in data]
        ys = np.array(ys, dtype=np.int32)
        self.data = (data, ys)
        self.ts = ts
        self.names = names
        self.N = N  # new add

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                # N = len(self.data[1])
                # order = list(range(N))
                # random.shuffle(order)
                # tmp_data = [[None] * N, [None] * N]
                # tmp_names = [None] * N
                # tmp_ts = [None] * N
                # for i in range(N):
                #     tmp_data[0][i] = self.data[0][order[i]]
                #     tmp_data[1][i] = self.data[1][order[i]]
                #     tmp_names[i] = self.names[order[i]]
                #     tmp_ts[i] = self.ts[order[i]]
                # self.data = tmp_data
                # self.names = tmp_names
                # self.ts = tmp_ts
                X, y, names, ts = common_utils.shuffle_tuple_list([self.data[0], self.data[1], self.names, self.ts])
                data = [X,y]
            else:
                # # sort entirely
                # (X, y, names, ts) = common_utils.sort_and_shuffle([self.data[0], self.data[1], self.names, self.ts], B)
                # data = [X, y]
                # No shuffle, No sort, just original data, sort in the batch
                (X, y, names, ts) = [self.data[0], self.data[1], self.names, self.ts]
                data = [X, y]

            # data[1] = np.array(data[1])  # this is important for Keras
            for i in range(0, len(data[0]), B):
                x_bs = data[0][i:i+B]
                y_bs = data[1][i:i+B]
                names_bs = names[i:i + B]
                ts_bs = ts[i:i + B]
                x_bs, y_bs, names_bs, ts_bs = common_utils.sort_tuple_list([x_bs, y_bs, names_bs, ts_bs]) # Newly added by Chengxi Zang, 2021/1/4
                x_length = [len(sq) for sq in x_bs]
                x_bs = common_utils.pad_zeros(x_bs)
                y_bs = np.array(y_bs)  # (B, 25)

                if self.target_repl:
                    y_rep = np.expand_dims(y_bs, axis=1).repeat(x_bs.shape[1], axis=1)  # (B, T, 25)
                    batch_data = (x_bs, [y_bs, y_rep], x_length)
                else:
                    batch_data = (x_bs, y_bs, x_length)

                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names_bs, "ts": ts_bs}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.N


def save_results(names, ts, predictions, labels, path):
    n_tasks = 25
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        header = ["stay", "period_length"]
        header += ["pred_{}".format(x) for x in range(1, n_tasks + 1)]
        header += ["label_{}".format(x) for x in range(1, n_tasks + 1)]
        header = ",".join(header)
        f.write(header + '\n')
        for name, t, pred, y in zip(names, ts, predictions, labels):
            line = [name]
            line += ["{:.6f}".format(t)]
            line += ["{:.6f}".format(a) for a in pred]
            line += [str(a) for a in y]
            line = ",".join(line)
            f.write(line + '\n')


# def generate_grid_search_BCE_cmd():
#     v_a = [0, 0.001,  0.002,
#            0.003,  0.004,
#            0.005, 0.006, 0.007,
#            0.008, 0.009, 0.01]
#     v_bs = [256, 512, 1024] #64, 128,
#     v_decay = [0, ]  #, 1e-5, 1e-4, 1e-3]  try to fix the effect of weight decay
#     with open('BCE.cmd', 'w') as f:
#         for a, bs, decay in itertools.product(v_a, v_bs, v_decay):
#             cmd = "python main_BCE.py --network lstm  --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 " \
#                   "--mode train --cuda --save_every 1 --epochs 50  " \
#                   "--coef_contra_loss {} --batch_size {} --weight_decay {} " \
#                   "2>&1 | tee log_BCE/BCE+SCL_cmd_a{}.bs{}.weDcy{}.log\n".format(a, bs, decay, a, bs, decay)
#             f.write(cmd)


def generate_grid_search_BCE_cmd():
    v_bs = [256, 512, 1024]  # 64, 128,
    v_a = [0, 0.001,  0.002,
           0.003,  0.004,
           0.005, 0.006, 0.007,
           0.008, 0.009, 0.01]
    v_decay = [0, ]  #, 1e-5, 1e-4, 1e-3]  try to fix the effect of weight decay
    with open('BCE.cmd', 'w') as f:
        for bs, a, decay in itertools.product(v_bs, v_a, v_decay):
            if bs <= 256:
                epoc = 60
            elif bs <= 512:
                epoc = 70
            else:
                epoc = 80
            cmd = "python main_BCE.py --network lstm  --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 " \
                  "--mode train --cuda --save_every 1 --epochs {}  " \
                  "--batch_size {} --coef_contra_loss {} --weight_decay {} " \
                  "2>&1 | tee log_BCE/BCE+SCL_cmd_bs{}.a{}.weDcy{}.log\n".format(epoc, bs, a, decay, bs, a, decay)
            f.write(cmd)

if __name__ == "__main__":

    generate_grid_search_BCE_cmd()

