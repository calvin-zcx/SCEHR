from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import sys
# for linux env.
sys.path.insert(0,'../..')
import imp
import re
import time
from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
# from mimic3models import keras_utils
from mimic3models import common_utils

# from keras.callbacks import ModelCheckpoint, CSVLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torch.utils.data.dataset import random_split
# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from mimic3models.pytorch_models.lstm import LSTM_PT, predict_labels
from mimic3models.pytorch_models.losses import SupConLoss
from tqdm import tqdm
from mimic3models.time_report import TimeReport
from mimic3models.pytorch_models.torch_utils import model_summary, TimeDistributed, shuffle_within_labels,shuffle_time_dim
from mimic3models.in_hospital_mortality.utils import random_add_positive_samples
import functools
print = functools.partial(print, flush=True)
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.utils.rnn as rnn_utils

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
# New added
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed manually for reproducibility.')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--coef_contra_loss', type=float, default=0, help='CE + coef * contrastive loss')

args = parser.parse_args()
print(args)

# if args.small_part:
#     args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'))

val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ph'
args_dict['num_classes'] = 25
args_dict['target_repl'] = target_repl

#
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Build the model
if args.network == "lstm":
    model = LSTM_PT(input_dim=76, hidden_dim=args.dim, num_layers=args.depth, num_classes=25,
                dropout=args.dropout, target_repl=False, deep_supervision=False, task='ph')
else:
    raise NotImplementedError

# NOTE for keras code: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    # loss = ['binary_crossentropy'] * 2
    # loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
    raise NotImplementedError
else:
    criterion = nn.BCELoss()  # nn.CrossEntropyLoss()
    criterion_SCL = SupConLoss() #temperature=0.01)  # temperature=opt.temp

    def criterion_SCL_multilabel(SCL, features, labels, mask=None):
        assert len(labels.shape) == 2
        losses = []
        for j in range(labels.shape[1]):
            y = labels[:, j]
            if y.sum().item() > 1:
                losses.append(SCL(features, y))
        mean = torch.mean(torch.stack(losses))
        return mean


# set lr and weight_decay later # or use other optimization say adamW later?
optimizer = torch.optim.Adam(model.parameters(),  lr=1e-3, weight_decay=0) # 1e-4)
training_losses = []
validation_losses = []
validation_results = []

# Load model weights # n_trained_chunks = 0
start_from_epoch = 0
if args.load_state != "":
    print('Load model state from: ', args.load_state)
    checkpoint = torch.load(args.load_state, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_from_epoch = checkpoint['epoch']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    validation_results = checkpoint['validation_results']
    print("Load epoch: ", start_from_epoch, 'Load model: ', )
    print(model)
    print('Load model done!')
    # n_trained_chunks = int(re.match(".*Epoch([0-9]+).*", args.load_state).group(1))

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: Got {} CUDA devices! Probably run with --cuda".format(torch.cuda.device_count()))
device = torch.device("cuda" if args.cuda else "cpu")
print('Using device: ', device)
model.to(device)
try:
    criterion_SCL.to(device)
except NameError:
    print("No criterion_SCL")
print(model)
model_summary(model)

if args.mode == 'train':
    print('Beginning training & validation data loading...')
    # Read data
    start_time = time.time()
    train_data_gen = utils.BatchGen(train_reader, discretizer,
                                    normalizer, args.batch_size,
                                    args.small_part, target_repl, shuffle=True)
    val_data_gen = utils.BatchGen(val_reader, discretizer,
                                  normalizer, args.batch_size,
                                  args.small_part, target_repl, shuffle=False)

    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Load training & validation data done. Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))
    print('Data summary:')
    print(' len(train_raw_torch):', len(train_data_gen), 'len(val_raw_torch):', len(val_data_gen))
    print(' batch size:', args.batch_size, 'epoch:', args.epochs, 'iters/epoch:', train_data_gen.steps)

    print("Beginning model training...")
    model.train()
    tr = TimeReport(total_iter=args.epochs * train_data_gen.steps)
    for epoch in (range(1 + start_from_epoch, 1 + args.epochs)):  # tqdm
        train_losses_batch = []
        for i in (range(train_data_gen.steps)):  # tqdm
            # print("predicting {} / {}".format(i, train_data_gen.steps), end='\r')
            X_batch_train, labels_batch_train, x_length = next(train_data_gen)

            X_batch_train = torch.tensor(X_batch_train, dtype=torch.float32)
            labels_batch_train = torch.tensor(labels_batch_train, dtype=torch.float32)
            X_batch_train = rnn_utils.pack_padded_sequence(X_batch_train, x_length, batch_first=True)

            optimizer.zero_grad()

            X_batch_train = X_batch_train.to(device)
            labels_batch_train = labels_batch_train.to(device)
            bsz = labels_batch_train.shape[0]
            # Data augmentation
            # # X_batch_aug = shuffle_time_dim(X_batch_train)
            # # X_batch_aug = torch.flip(X_batch_train, dims=[1])
            # X_batch_aug = X_batch_train + torch.randn(X_batch_train.shape).to(device)
            # X_batch_train = torch.cat([X_batch_train, X_batch_aug], dim=0)
            #
            y_hat_train, y_representation = model(X_batch_train)
            loss_CE = criterion(y_hat_train, labels_batch_train)
            if args.coef_contra_loss > 0:
                # y_representation = torch.cat([y_representation.unsqueeze(1), y_representation.unsqueeze(1)], dim=1)
                # y_representation = torch.cat([y_representation.unsqueeze(1),
                #                               shuffle_within_labels(y_representation, labels_batch_train).unsqueeze(1)]
                #                              , dim=1)
                # f1, f2 = torch.split(y_representation, [bsz, bsz], dim=0)  # (bs, 128), (bs, 128)
                # y_representation = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # (bs, 2, 128)
                y_representation = y_representation.unsqueeze(1)
                # loss_SCL = criterion_SCL(y_representation, labels_batch_train)
                loss_SCL = criterion_SCL_multilabel (criterion_SCL, y_representation, labels_batch_train)
                # loss = criterion(y_hat_train.chunk(2, dim=0)[0], labels_batch_train) + args.coef_contra_loss * loss_SCL
                loss = loss_CE + args.coef_contra_loss * loss_SCL
            else:
                loss = loss_CE  # criterion(y_hat_train, labels_batch_train) #
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)  # args.clip) seems litte effect
            optimizer.step()

            train_loss_batch = loss.item()
            train_losses_batch.append(train_loss_batch)
            tr.update()

        training_loss = np.mean(train_losses_batch)
        training_losses.append(training_loss)

        with torch.no_grad():
            val_losses_batch = []
            model.eval()
            predicted_prob_val = []
            true_labels_val = []
            for i in range(val_data_gen.steps):
                X_batch_val, labels_batch_val, x_length = next(val_data_gen)

                X_batch_val = torch.tensor(X_batch_val, dtype=torch.float32)
                labels_batch_val = torch.tensor(labels_batch_val, dtype=torch.float32)
                X_batch_val = rnn_utils.pack_padded_sequence(X_batch_val, x_length, batch_first=True)

                X_batch_val = X_batch_val.to(device)
                labels_batch_val = labels_batch_val.to(device)
                bsz = labels_batch_val.shape[0]

                y_hat_val, _ = model(X_batch_val)
                val_loss_batch = criterion(y_hat_val, labels_batch_val).item()
                val_losses_batch.append(val_loss_batch)
                # predicted labels
                p_batch_val, _ = predict_labels(y_hat_val)
                predicted_prob_val.append(p_batch_val)
                true_labels_val.append(labels_batch_val)

            validation_loss = np.mean(val_losses_batch)
            validation_losses.append(validation_loss)

            predicted_prob_val = torch.cat(predicted_prob_val, dim=0)
            true_labels_val = torch.cat(true_labels_val, dim=0)
            predicted_prob_val_pos = (predicted_prob_val.cpu().detach().numpy())
            true_labels_val = true_labels_val.cpu().detach().numpy()
            val_result = metrics.print_metrics_multilabel(true_labels_val, predicted_prob_val_pos, verbose=0)
            validation_results.append(val_result)
            model.train()

        # if (epoch+1) % args.log_interval == 0: move into interation
        print('Epoch [{}/{}], {} Iters/Epoch, training_loss: {:.5f}, validation_loss: {:.5f}, '
              '{:.2f} sec/iter, {:.2f} iters/sec: '.
              format(epoch, args.epochs, train_data_gen.steps,
                     training_loss, validation_loss,
                     tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
        tr.print_summary()
        print(val_result)

        if args.save_every and epoch % args.save_every == 0:
            # Set model checkpoint/saving path
            model_final_name = model.say_name()
            path = os.path.join('pytorch_states/' + model_final_name
                                + '.bs{}.epoch{}.trainLos{:.3f}'
                                  '.Val-CoefCL{}.Los{:.3f}.AUCmic{:.3f}.AUCmac{:.4f}.AUCwei{:.4f}.pt'.
                                format(args.batch_size, epoch, training_loss,
                                       "{:.3f}".format(args.coef_contra_loss) if args.coef_contra_loss != 0 else "0",
                                       validation_loss,
                                       val_result['ave_auc_micro'], val_result['ave_auc_macro'], val_result['ave_auc_weighted']))
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'training_loss': training_loss,
                'validation_loss': validation_loss,
                'model_str': model.__str__(),
                'args:': args,
                'validation_result': val_result,
                'validation_results': validation_results
            }, path)
            # print('Save ', path, ' done.')

    print('Training complete...')
    try:
        r = {
            'ave_auc_micro': [x['ave_auc_micro'] for x in validation_results],
            # 'acc': [x['acc'] for x in validation_results],
            # 'auprc': [x['auprc'] for x in validation_results]
        }
        pdr = pd.DataFrame(data=r, index=range(1, len(validation_results) + 1))
        ax = pdr.plot.line()
        plt.grid()
        fig = ax.get_figure()
        plt.ylim((0.7, 0.8))
        plt.show()
        fig.savefig(path + '.png')
        fig.savefig(path + '.pdf')
        r_all = {
            'ave_auc_micro': [x['ave_auc_micro'] for x in validation_results],
            'ave_auc_macro': [x['ave_auc_macro'] for x in validation_results],
            'ave_auc_weighted': [x['ave_auc_weighted'] for x in validation_results]
        }
        pd_r_all = pd.DataFrame(data=r_all, index=range(1, len(validation_results) + 1))
        pd_r_all.to_csv(path + '.csv')
    except ValueError:
        print("Error in Plotting validation auroc values over epochs")

elif args.mode == 'test':
    print('Beginning testing...')
    start_time = time.time()
    # ensure that the code uses test_reader
    model.eval()
    model.to(torch.device('cpu'))

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    # del train_data_gen
    # del val_data_gen

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

    test_data_gen = utils.BatchGen(test_reader, discretizer,
                                   normalizer, args.batch_size,
                                   args.small_part, target_repl,
                                   shuffle=False, return_names=True)

    names = []
    ts = []
    labels = []
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(test_data_gen.steps)):
            print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
            ret = next(test_data_gen)
            x = ret["data"][0]
            y = ret["data"][1]
            x_length = ret["data"][2]
            cur_names = ret["names"]
            cur_ts = ret["ts"]
            # x = np.array(x)

            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            x_pack = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)

            pred, _ = model(x_pack)
            predictions.append(pred)
            labels.append(y)
            names += list(cur_names)
            ts += list(cur_ts)

        predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
        labels = torch.cat(labels, dim=0).cpu().detach().numpy()

    results = metrics.print_metrics_multilabel(labels, predictions)
    path = os.path.join("test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, ts, predictions, labels, path)
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Testing elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))

else:
    raise ValueError("Wrong value for args.mode")
