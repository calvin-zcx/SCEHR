from __future__ import absolute_import
from __future__ import print_function

from mimic3models import common_utils, metrics
import numpy as np
import os
import itertools
import os
import pandas as pd
import re


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
    v_a = [0, 0.0005, 0.001, 0.0015, 0.002,
           0.0025, 0.003, 0.0035, 0.004, 0.0045,
           0.005, 0.0055, 0.006, 0.0065, 0.007,
           0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
    v_bs = [64, 128, 256, 512, 1024]
    v_decay = [0, ]  #, 1e-5, 1e-4, 1e-3]  try to fix the effect of weight decay
    with open('CBCE.cmd', 'w') as f:
        for a, bs, decay in itertools.product(v_a, v_bs, v_decay):
            cmd = "echo $LINENO && python main_CBCE.py --network lstm  " \
                  "--dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 " \
                  "--coef_contra_loss {}  --batch_size {} --weight_decay {} " \
                  "2>&1 | tee log/CBCE+SCL_bach_cmd_a{}.bs{}.weightDecay{}.log\n".format(a, bs, decay, a, bs, decay)
            f.write(cmd)


def generate_grid_search_CBCE_nonorm_cmd():
    v_a = [0, 0.0005, 0.001, 0.0015, 0.002,
           0.0025, 0.003, 0.0035, 0.004, 0.0045,
           0.005, 0.0055, 0.006, 0.0065, 0.007,
           0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
    v_bs = [64, 128, 256, 512, 1024]
    v_decay = [0, ]  #, 1e-5, 1e-4, 1e-3]  try to fix the effect of weight decay
    with open('CBCE_nonorm.cmd', 'w') as f:
        for a, bs, decay in itertools.product(v_a, v_bs, v_decay):
            cmd = "python main_CBCE.py --output_dir pytorch_states/CBCE_nonorm/ --network lstm  " \
                  "--dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 " \
                  "--coef_contra_loss {}  --batch_size {} --weight_decay {} " \
                  "2>&1 | tee log_CBCE_nonorm/CBCE+SCL_bach_cmd_a{}.bs{}.weightDecay{}.log\n".format(a, bs, decay, a, bs, decay)
            f.write(cmd)


def generate_grid_search_BCE_cmd():
    v_a = [0, 0.0005, 0.001, 0.0015, 0.002,
           0.0025, 0.003, 0.0035, 0.004, 0.0045,
           0.005, 0.0055, 0.006, 0.0065, 0.007,
           0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
    v_bs = [64, 128, 256, 512, 1024]
    v_decay = [0, ]  #, 1e-5, 1e-4, 1e-3]  try to fix the effect of weight decay
    with open('BCE.cmd', 'w') as f:
        for a, bs, decay in itertools.product(v_a, v_bs, v_decay):
            # 2>&1 | tee log/MCE+SCL_hasstatic_a0_bs256_new.log"
            cmd = "python main_BCE.py --network lstm  " \
                  "--dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 " \
                  "--coef_contra_loss {}  --batch_size {} --weight_decay {} " \
                  "2>&1 | tee log_BCE/BCE+SCL_bach_cmd_a{}.bs{}.weDcy{}.log\n".format(a, bs, decay, a, bs, decay)
            f.write(cmd)


def generate_grid_search_MCE_cmd():
    v_a = [0, 0.0005, 0.001, 0.0015, 0.002,
           0.0025, 0.003, 0.0035, 0.004, 0.0045,
           0.005, 0.0055, 0.006, 0.0065, 0.007,
           0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
    v_bs = [64, 128, 256, 512, 1024]
    v_decay = [0, ]  #, 1e-5, 1e-4, 1e-3]  try to fix the effect of weight decay
    with open('MCE.cmd', 'w') as f:
        for a, bs, decay in itertools.product(v_a, v_bs, v_decay):
            cmd = "python main_MCE.py --network lstm  " \
                  "--dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --cuda --save_every 0 --epochs 100 " \
                  "--coef_contra_loss {}  --batch_size {} --weight_decay {} " \
                  "2>&1 | tee log_MCE/MCE+SCL_bach_cmd_a{}.bs{}.weDcy{}.log\n".format(a, bs, decay, a, bs, decay)
            f.write(cmd)


def generate_grid_search_downsampling_cmd():
    v_model = ['BCE', 'CBCE', 'MCE']
    v_ds= [0.05, 0.01, 0.001]
    v_a = [0, 0.0005, 0.001, 0.0015, 0.002,
           0.0025, 0.003, 0.0035, 0.004, 0.0045,
           0.005, 0.0055, 0.006, 0.0065, 0.007,
           0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
    v_bs = [128, 256, 512, 1024]

    with open('downsample_all.cmd', 'w') as f:
        for model, downsample, a, bs in itertools.product(v_model, v_ds, v_a, v_bs):
            if bs <= 256:
                epoc = 60  # 60
            else:
                epoc = 100
            cmd = "python main_ds_{}.py --network lstm  --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 " \
                  "--mode train --cuda --save_every 0  " \
                  "--targeted_positive_ratio {} --coef_contra_loss {}  --batch_size {} --epochs {} " \
                  "2>&1 | tee log_{}_ds/{}+SCL_cmd_prate{}.a{}.bs{}.log\n".format(
                model,
                downsample, a, bs,  epoc,
                model, model, downsample, a, bs)
            f.write(cmd)

def summarize_results_from_csv_files(dir, f_cond=None, out_file=None):
    # pytorch_states/CBCE/
    if f_cond is None:
        f_list = list(filter(lambda x: '.csv' in x, os.listdir(dir)))
    else:
        f_list = list(filter(f_cond, os.listdir(dir)))
    print('len(f_list)', len(f_list))
    val = []
    test = []
    for f_name in f_list:
        df = pd.read_csv(os.path.join(dir, f_name), index_col=0)
        id_val = df['auroc-val'].idxmax()
        id_test = df['auroc-test'].idxmax()
        r_val = df.loc[id_val,:].copy()
        r_val['file-name'] = f_name
        r_test = df.loc[id_test, :].copy()
        r_test['file-name'] = f_name
        val.append(r_val)
        test.append(r_test)

    df_val = pd.DataFrame(val)
    df_val = df_val.sort_values(by='auroc-val', ascending=False)
    df_test = pd.DataFrame(test)
    df_test = df_test.sort_values(by='auroc-test', ascending=False)
    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)
    if out_file is None:
        last = list(filter(lambda a: a != '', re.split("\/", dir)))[-1]
        fo = r'pytorch_states/'+last+'_results.xlsx'
    else:
        fo = out_file
    with pd.ExcelWriter(fo) as writer:
        df_test.to_excel(writer, sheet_name='test')
        df_val.to_excel(writer, sheet_name='validation')
    print('Dump ', fo, 'done!')


def boostrap_interval_and_std(y_pre, y_true,  n_bootstraps=100, path=None):
    assert  len(y_pre) == len(y_true)
    n = len(y_true)
    results = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        idx = np.random.choice(n, n)
        r = metrics.print_metrics_binary(y_true[idx], y_pre[idx], verbose=0)
        results.append(r)

    results_dict = {
        'auroc': [x['auroc'] for x in results],
        'auprc': [x['auprc'] for x in results],
        'acc': [x['acc'] for x in results],
        'minpse': [x['minpse'] for x in results],
        'prec0': [x['prec0'] for x in results],
        'prec1': [x['prec1'] for x in results],
        'rec0': [x['rec0'] for x in results],
        'rec1': [x['rec1'] for x in results],
    }
    pd_r = pd.DataFrame(data=results_dict)
    # pd_r.to_csv(path + '.csv')
    return pd_r


def label_targed_downsample(reader, targeted_ratio, label=1):
    idx = []
    N = len(reader._data)
    for i in range(N):
        if reader._data[i][1] == label:
            idx.append(i)
    n = len(idx)
    ratio = n * 1.0 / N
    print('original ratio: ', ratio)
    print('targeted ratio: ', targeted_ratio)
    assert targeted_ratio <= ratio
    x = (n - targeted_ratio * N) / (1 - targeted_ratio)
    drop = x / n
    data = []
    n_new = 0
    for d in reader._data:
        if d[1] == label:
            if np.random.random_sample() < 1 - drop:
                data.append(d)
                n_new += 1
        else:
            data.append(d)
    print('actual ration: ', 1.0*n_new / len(data))
    reader._data = data
    return data



if __name__ == "__main__":
    generate_grid_search_CBCE_nonorm_cmd()

    # generate_grid_search_downsampling_cmd()


    # execute only if run as a script
    # generate_grid_search_CBCE_cmd()
    # generate_grid_search_BCE_cmd()
    # generate_grid_search_MCE_cmd()
    # summarize_results_from_csv_files(r'pytorch_states/CBCE_Linux/')
    # summarize_results_from_csv_files(r'pytorch_states/BCE_Linux/')
    # summarize_results_from_csv_files(r'pytorch_states/BCE_Linux/',
    #                                  lambda x: ('.csv' in x) and ('BCE+SCL.a0.0.bs' in x),
    #                                  r'pytorch_states/BCE_Linux_OnlyBCENoSCL_results.xlsx')
    # summarize_results_from_csv_files(r'pytorch_states/CBCE_Linux/',
    #                                  lambda x: ('.csv' in x) and ('CBCE+SCL.a0.0.bs' in x),
    # #                                  r'pytorch_states/CBCE_Linux_OnlyCBCENoSCL_results.xlsx')
    # summarize_results_from_csv_files(r'pytorch_states/MCE_Linux/')
    # summarize_results_from_csv_files(r'pytorch_states/MCE_Linux/',
    #                                  lambda x: ('.csv' in x) and ('MCE+SCL.a0.0.bs' in x),
    #                                  r'pytorch_states/MCE_Linux_OnlyBCENoSCL_results.xlsx')

    # # change prate for different positive ratio attack results.
    # # bare BCE
    # prate = 0.001
    # summarize_results_from_csv_files(r'pytorch_states/BCE_ds_Linux/',
    #                                  lambda x: ('.csv' in x) and ('.BCE+SCL.prate{}.a0.0.'.format(prate) in x),
    #                                  r'pytorch_states/BCE_ds_Linux.prate{}.a0.0.results.xlsx'.format(prate))
    # # BCE + SCL
    # summarize_results_from_csv_files(r'pytorch_states/BCE_ds_Linux/',
    #                                  lambda x: ('.csv' in x) and ('.BCE+SCL.prate{}.a'.format(prate) in x),
    #                                  r'pytorch_states/BCE_ds_Linux.prate{}.results.xlsx'.format(prate))
    # # CBCE + SCL
    # summarize_results_from_csv_files(r'pytorch_states/CBCE_ds_Linux/',
    #                                  lambda x: ('.csv' in x) and ('.CBCE+SCL.prate{}.a'.format(prate) in x),
    #                                  r'pytorch_states/CBCE_ds_Linux.prate{}.results.xlsx'.format(prate))
    # # MCE + SCL
    # summarize_results_from_csv_files(r'pytorch_states/MCE_ds_Linux/',
    #                                  lambda x: ('.csv' in x) and ('.MCE+SCL.prate{}.a'.format(prate) in x),
    #                                  r'pytorch_states/MCE_ds_Linux.prate{}.results.xlsx'.format(prate))
