from __future__ import absolute_import
from __future__ import print_function

from mimic3models import common_utils
import numpy as np
import os
import itertools

def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)] # should debug into transform to check
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))


def random_add_positive_samples(x, y, number_to_add):
    # x, y are np.array
    index_of_pos = np.nonzero(y)[0]
    index_to_add = np.random.choice(index_of_pos, number_to_add)
    xx = np.append(x, x[index_to_add], axis=0)
    yy = np.append(y, np.ones(number_to_add), axis=0)
    return xx, yy


def generate_grid_search_CBCE_cmd():
    v_a = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
    v_bs = [64, 128, 256, 512, 1024]
    v_decay = [0, 1e-5, 1e-4, 1e-3]
    with open('CBCE.cmd', 'w') as f:
        for a, bs, decay in itertools.product(v_a, v_bs, v_decay):
            # 2>&1 | tee log/MCE+SCL_hasstatic_a0_bs256_new.log"
            cmd = "echo $LINENO && python main_CBCE.py --network lstm  " \
                  "--dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 " \
                  "--coef_contra_loss {}  --batch_size {} --weight_decay {}\n".format(a, bs, decay)
            f.write(cmd)


if __name__ == "__main__":
    # execute only if run as a script
    generate_grid_search_CBCE_cmd()