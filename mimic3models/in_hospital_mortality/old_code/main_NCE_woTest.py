from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import time
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import common_utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torch.utils.data.dataset import random_split
# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from mimic3models.pytorch_models.lstm import LSTM_PT, predict_labels
from mimic3models.pytorch_models.losses import SupConLoss, SupNCELoss
from tqdm import tqdm
from mimic3models.time_report import TimeReport
from mimic3models.pytorch_models.torch_utils import model_summary, TimeDistributed, shuffle_within_labels,shuffle_time_dim
from mimic3models.in_hospital_mortality.utils import random_add_positive_samples
import functools
print = functools.partial(print, flush=True)

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../../data/in-hospital-mortality/'))
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

# parser.add_argument('--add_positive', action='store_true',
#                     help='add_positive_to_data')
args = parser.parse_args()
print(args)
# if args.small_part:
#     args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl
#
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Build the model
if args.network == "lstm":
    model = LSTM_PT(input_dim=76, hidden_dim=args.dim, num_layers=args.depth, num_classes=2,
                dropout=args.dropout, target_repl=False, deep_supervision=False, task='ihm')
else:
    raise NotImplementedError

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    # loss = ['binary_crossentropy'] * 2
    # loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
    raise NotImplementedError
else:
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    criterion_SCL = SupConLoss(temperature=0.1)   # temperature=0.01)  # temperature=opt.temp
    criterion_SupNCE = SupNCELoss(temperature=1)

    def NEC_loss(y_pre, y):
        # y_pre: (bs, 2), after sigmoid
        # y: (bs,)
        i1 = (y==1)
        i0 = (y==0)
        pos_anchor = y_pre[i1, 1].log() + (1.-y_pre[i1, 0]).log()
        neg_anchor = y_pre[i0, 0].log() + (1.-y_pre[i0, 1]).log()
        ret = torch.cat([pos_anchor, neg_anchor]).mean()
        return -ret

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
    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)  # (14681,48,76), (14681,)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)  # (3222,48,76), (3222,)

    # print('add positive data! train_raw[0].shape:', train_raw[0].shape)
    # y = np.array(train_raw[1])
    # xx, yy = random_add_positive_samples(train_raw[0], y, y.shape[0] - 2*y.sum())
    # train_raw = (xx, yy)
    # print('add positive data done! train_raw[0].shape:', train_raw[0].shape)

    train_raw_torch = TensorDataset(torch.tensor(train_raw[0], dtype=torch.float32),
                                    torch.tensor(train_raw[1], dtype=torch.float32))
    val_raw_torch = TensorDataset(torch.tensor(val_raw[0], dtype=torch.float32),
                                  torch.tensor(val_raw[1], dtype=torch.float32))

    train_loader = DataLoader(dataset=train_raw_torch, batch_size=args.batch_size, drop_last=False, shuffle=True)
    val_loader = DataLoader(dataset=val_raw_torch, batch_size=args.batch_size, drop_last=False, shuffle=False)
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Load training & validation data done. Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))
    print('Data summary:')
    print(' len(train_raw_torch): ', len(train_raw_torch), 'len(val_raw_torch): ', len(val_raw_torch))
    print(' batch size: ', args.batch_size)
    print(' len(train_loader): ', len(train_loader), 'len(val_loader): ', len(val_loader))

    print("Beginning model training...")
    model.train()
    iter_per_epoch = (len(train_loader) + args.batch_size - 1) // args.batch_size
    tr = TimeReport(total_iter=args.epochs * iter_per_epoch)
    for epoch in (range(1+start_from_epoch, 1+args.epochs)): #tqdm
        train_losses_batch = []
        i = 0
        for X_batch_train, labels_batch_train in train_loader:
            i += 1
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
            # loss_CE = criterion(y_hat_train, labels_batch_train)
            loss_CE = NEC_loss(y_hat_train, labels_batch_train)
            if args.coef_contra_loss > 0:
                # y_representation = torch.cat([y_representation.unsqueeze(1), y_representation.unsqueeze(1)], dim=1)
                # y_representation = torch.cat([y_representation.unsqueeze(1),
                #                               shuffle_within_labels(y_representation, labels_batch_train).unsqueeze(1)]
                #                              , dim=1)
                # f1, f2 = torch.split(y_representation, [bsz, bsz], dim=0)  # (bs, 128), (bs, 128)
                # y_representation = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # (bs, 2, 128)
                y_representation = y_representation.unsqueeze(1)
                loss_SCL = criterion_SCL(y_representation, labels_batch_train)
                # loss_SCL = criterion_SupNCE(y_representation, labels_batch_train)
                # loss = criterion(y_hat_train.chunk(2, dim=0)[0], labels_batch_train) + args.coef_contra_loss * loss_SCL
                loss = (1. - args.coef_contra_loss) * loss_CE + args.coef_contra_loss * loss_SCL
            else:
                loss = loss_CE  #  criterion(y_hat_train, labels_batch_train) #
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)  # args.clip) seems little effect
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
            for X_batch_val, labels_batch_val in val_loader:
                X_batch_val = X_batch_val.to(device)
                labels_batch_val = labels_batch_val.to(device)
                y_hat_val, _ = model(X_batch_val)
                # val_loss_batch = criterion(y_hat_val, labels_batch_val).item()
                val_loss_batch = NEC_loss(y_hat_val, labels_batch_val).item()
                val_losses_batch.append(val_loss_batch)
                # predicted labels
                # p_batch_val, _ = predict_labels(y_hat_val)
                # predicted_prob_val.append(p_batch_val)
                predicted_prob_val.append(y_hat_val)
                true_labels_val.append(labels_batch_val)

            validation_loss = np.mean(val_losses_batch)
            validation_losses.append(validation_loss)

            predicted_prob_val = torch.cat(predicted_prob_val, dim=0)
            true_labels_val = torch.cat(true_labels_val, dim=0)
            predicted_prob_val_pos = (predicted_prob_val.cpu().detach().numpy())[:, 1]
            # predicted_prob_val_pos = predicted_prob_val_pos[:, 1] * (1 - predicted_prob_val_pos[:, 0])

            true_labels_val = true_labels_val.cpu().detach().numpy()
            val_result = metrics.print_metrics_binary(true_labels_val, predicted_prob_val_pos, verbose=0)
            validation_results.append(val_result)
            model.train()

        # if (epoch+1) % args.log_interval == 0: move into interation
        print('Epoch [{}/{}], {} Iters/Epoch, training_loss: {:.5f}, validation_loss: {:.5f}, '
              '{:.2f} sec/iter, {:.2f} iters/sec: '.
              format(epoch, args.epochs, iter_per_epoch,
                     training_loss, validation_loss,
                     tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
        tr.print_summary()
        print(val_result)

        if args.save_every and epoch % args.save_every == 0:
            # Set model checkpoint/saving path
            model_final_name = model.say_name()
            path = os.path.join('pytorch_states/' + model_final_name
                                + 'gclip1.5.bs{}.epoch{}.trainLoss{:.3f}'
                                  '.Val-NCE+SCL-CoefCL{}.Loss{:.3f}.ACC{:.3f}.AUROC{:.4f}.AUPRC{:.4f}.pt'.
                                format(args.batch_size, epoch, training_loss,
                                       "{:.3f}".format(args.coef_contra_loss) if args.coef_contra_loss != 0 else "0",
                                       validation_loss,
                                       val_result['acc'], val_result['auroc'], val_result['auprc']))
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
            'auroc': [x['auroc'] for x in validation_results],
            # 'acc': [x['acc'] for x in validation_results],
            # 'auprc': [x['auprc'] for x in validation_results]
        }
        pdr = pd.DataFrame(data=r, index=range(1, len(validation_results)+1))
        ax = pdr.plot.line()
        plt.grid()
        fig = ax.get_figure()
        plt.ylim((0.8, 0.9))
        plt.show()
        fig.savefig(path + '.png')
        fig.savefig(path + '.pdf')
        r_all = {
            'auroc': [x['auroc'] for x in validation_results],
            'acc': [x['acc'] for x in validation_results],
            'auprc': [x['auprc'] for x in validation_results]
        }
        pd_r_all = pd.DataFrame(data=r_all, index=range(1, len(validation_results) + 1))
        pd_r_all.to_csv(path+'.csv')
    except ValueError:
        print("Error in Plotting validation auroc values over epochs")


elif args.mode == 'test':
    print('Beginning testing...')
    start_time = time.time()
    # ensure that the code uses test_reader
    model.eval()
    model.to(torch.device('cpu'))

    del train_reader
    del val_reader
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)
    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]
    test_raw_torch = TensorDataset(torch.tensor(data, dtype=torch.float32),
                                   torch.tensor(labels, dtype=torch.long))
    test_loader = DataLoader(dataset=test_raw_torch, batch_size=args.batch_size, drop_last=False, shuffle=False)

    predicted_prob = []
    true_labels = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            y_hat_batch, _ = model(X_batch)
            # loss = criterion(y_hat, y_batch)
            # p_batch, _ = predict_labels(y_hat_batch)
            # predicted_prob.append(p_batch)
            predicted_prob.append(y_hat_batch)
            true_labels.append(y_batch)
        predicted_prob = torch.cat(predicted_prob, dim=0)
        true_labels = torch.cat(true_labels, dim=0) # with threshold 0.5, not used here

    predictions = (predicted_prob.cpu().detach().numpy())[:, 1]
    # predictions = predictions[:, 1] * (1-predictions[:, 0])
    true_labels = true_labels.cpu().detach().numpy()
    test_results = metrics.print_metrics_binary(true_labels, predictions)
    print(test_results)
    # path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
    path = os.path.join("../test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, true_labels, path)
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Testing elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))

else:
    raise ValueError("Wrong value for args.mode")


