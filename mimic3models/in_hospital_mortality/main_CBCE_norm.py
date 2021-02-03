from __future__ import absolute_import
from __future__ import print_function
import sys
# for linux env.
sys.path.insert(0, '../..')

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
from torch.utils.data import DataLoader

from mimic3models.pytorch_models.lstm import LSTM_PT, LSTM_PT_v2
from mimic3models.pytorch_models.losses import SupConLoss, SupNCELoss, CBCE_loss, CBCE_WithLogitsLoss
from tqdm import tqdm
from mimic3models.time_report import TimeReport
from mimic3models.pytorch_models.torch_utils import Dataset, optimizer_to, model_summary
import matplotlib.pyplot as plt
import pandas as pd
import functools
import random
print = functools.partial(print, flush=True)

# %%
# Arguments:
parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='pytorch_states/CBCE_normlinear/')
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
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay of adam')
args = parser.parse_args()
print(args)

# %%
# Set the random seed manually for reproducibility.
# https://pytorch.org/docs/stable/notes/randomness.html
# https://pytorch.org/docs/stable/generated/torch.set_deterministic.html#torch.set_deterministic
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
np.random.seed(args.seed)
random.seed(args.seed)

torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

# %%
# Load data readers
target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')
# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
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

# %%
# GPU, Model, Optimizer, Criterion Setup/Loading
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: Got {} CUDA devices! Probably run with --cuda".format(torch.cuda.device_count()))
device = torch.device("cuda" if args.cuda else "cpu")
print('Using device: ', device)

# Build the model
if args.network == "lstm":
    model = LSTM_PT_v2(input_dim=76, hidden_dim=args.dim, num_layers=args.depth, num_classes=2,
                    dropout=args.dropout, target_repl=False, deep_supervision=False, task='ihm',
                    final_act=nn.Identity())
else:
    raise NotImplementedError


if target_repl:
    raise NotImplementedError
else:
    # criterion_BCE = nn.BCELoss()
    criterion_SCL = SupConLoss(temperature=0.1)   # temperature=0.01)  # temperature=opt.temp
    # criterion_SupNCE = SupNCELoss(temperature=1)
    # criterion_MCE = nn.CrossEntropyLoss()

    def get_loss(y_pre, labels, representation, alpha=0):
        # CBCE_WithLogitsLoss is more numerically stable than CBCE_Loss when model is complex/overfitting
        loss = CBCE_WithLogitsLoss(y_pre, labels)
        if alpha > 0:
            if labels.sum().item() < 2:
                print('Warning: # positives < 2, NOT USING Supervised Contrastive Regularizer')
            else:
                if len(representation.shape) == 2:
                    representation = representation.unsqueeze(1)
                scl_loss = criterion_SCL(representation, labels)
                loss = loss + alpha * scl_loss
        return loss

# set lr and weight_decay later # or use other optimization say adamW later?
optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)
training_losses = []
validation_losses = []
validation_results = []
test_results = []
model_names = []
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
    test_results = checkpoint['test_results']
    optimizer_to(optimizer, device)
    print("Load epoch: ", start_from_epoch, 'Load model: ', )
    print(model)
    print('Load model done!')
    # n_trained_chunks = int(re.match(".*Epoch([0-9]+).*", args.load_state).group(1))

model.to(device)
try:
    criterion_SCL.to(device)
except NameError:
    print("No criterion_SCL")
print(model)
model_summary(model)

# %%
# Training & Testing parts:
if args.mode == 'train':
    print('Training part: beginning loading training & validation datasets...')
    start_time = time.time()
    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part, return_names=True)  # (14681,48,76), (14681,)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part, return_names=True)  # (3222,48,76), (3222,)
    test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part, return_names=True)

    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False)

    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Load training & validation data done. Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))
    print('Data summary:')
    print(' batch size: ', args.batch_size)
    print(' len(train_dataset): ', len(train_dataset), 'len(train_loader): ', len(train_loader))
    print(' len(valid_dataset): ', len(valid_dataset), 'len(val_loader): ', len(val_loader))
    print(' len(test_dataset): ', len(test_dataset), 'len(test_loader): ', len(test_loader))
    print("Beginning model training...")

    iter_per_epoch = len(train_loader)
    tr = TimeReport(total_iter=args.epochs * iter_per_epoch)
    for epoch in (range(1+start_from_epoch, 1+args.epochs)): #tqdm
        model.train()
        train_losses_batch = []
        for i, (X_batch_train, labels_batch_train, name_batch_train) in enumerate(train_loader):
            optimizer.zero_grad()
            X_batch_train = X_batch_train.float().to(device)
            labels_batch_train = labels_batch_train.float().to(device)
            # bsz = labels_batch_train.shape[0]
            y_hat_train, y_representation = model(X_batch_train)
            loss = get_loss(y_hat_train, labels_batch_train, y_representation, args.coef_contra_loss)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)  # args.clip) seems little effect
            optimizer.step()
            train_losses_batch.append(loss.item())
            tr.update()

        training_loss = np.mean(train_losses_batch)
        training_losses.append(training_loss)

        # Validation Part
        print('Validation results:')
        with torch.no_grad():
            model.eval()
            val_losses_batch = []
            predicted_prob_val = []
            true_labels_val = []
            for X_batch_val, labels_batch_val, name_batch_val in val_loader:
                X_batch_val = X_batch_val.float().to(device)
                labels_batch_val = labels_batch_val.float().to(device)
                y_hat_val, y_representation_val = model(X_batch_val)
                val_loss_batch = get_loss(y_hat_val, labels_batch_val, y_representation_val, args.coef_contra_loss)
                val_losses_batch.append(val_loss_batch.item())
                # predicted labels
                y_hat_val = torch.sigmoid(y_hat_val)
                y_hat_val = y_hat_val / y_hat_val.sum(dim=1, keepdim=True)
                predicted_prob_val.append(y_hat_val[:, 1])
                true_labels_val.append(labels_batch_val)

            validation_loss = np.mean(val_losses_batch)
            validation_losses.append(validation_loss)

            predicted_prob_val = torch.cat(predicted_prob_val, dim=0).cpu().detach().numpy()
            true_labels_val = torch.cat(true_labels_val, dim=0).cpu().detach().numpy()

            val_result = metrics.print_metrics_binary(true_labels_val, predicted_prob_val, verbose=1)
            print(val_result)
            validation_results.append(val_result)

        # Additional test part. God View. should not used for model selection
        print('Test results:')
        predicted_prob_test = []
        true_labels_test = []
        name_test = []
        with torch.no_grad():
            model.eval()
            for X_batch_test, y_batch_test, name_batch_test in test_loader:
                X_batch_test = X_batch_test.float().to(device)
                y_batch_test = y_batch_test.float().to(device)
                y_hat_batch_test, _ = model(X_batch_test)

                y_hat_batch_test = torch.sigmoid(y_hat_batch_test)
                y_hat_batch_test = y_hat_batch_test / y_hat_batch_test.sum(dim=1, keepdim=True)
                predicted_prob_test.append(y_hat_batch_test[:,1])
                true_labels_test.append(y_batch_test)
                name_test.append(name_batch_test)

            predicted_prob_test = torch.cat(predicted_prob_test, dim=0)
            true_labels_test = torch.cat(true_labels_test, dim=0)  # with threshold 0.5, not used here

            predictions_test = (predicted_prob_test.cpu().detach().numpy())
            true_labels_test = true_labels_test.cpu().detach().numpy()
            name_test = np.concatenate(name_test)

            test_result = metrics.print_metrics_binary(true_labels_test, predictions_test, verbose=1)
            print(test_result)
            test_results.append(test_result)

        print('Epoch [{}/{}], {} Iters/Epoch, training_loss: {:.5f}, validation_loss: {:.5f}, '
              '{:.2f} sec/iter, {:.2f} iters/sec: '.
              format(epoch, args.epochs, iter_per_epoch,
                     training_loss, validation_loss,
                     tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
        tr.print_summary()
        print("=" * 50)

        model_final_name = model.say_name()
        path = os.path.join(args.output_dir + model_final_name
                            + '.CBCE+SCL.a{}.bs{}.wdcy{}.epo{}.TrLos{:.2f}.'
                              'VaLos{:.2f}.ACC{:.3f}.ROC{:.4f}.PRC{:.4f}.'
                              'TstACC{:.3f}.ROC{:.4f}.PRC{:.4f}'.
                            format(args.coef_contra_loss, args.batch_size, args.weight_decay,
                                   epoch, training_loss, validation_loss,
                                   val_result['acc'], val_result['auroc'], val_result['auprc'],
                                   test_result['acc'], test_result['auroc'], test_result['auprc']))
        model_names.append(path + '.pt')
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if args.save_every and epoch % args.save_every == 0:
            # Set model checkpoint/saving path
            test_details = {
                'name': name_test,
                'prediction': predictions_test,
                'true_labels': true_labels_test
            }
            torch.save({
                'model_str': model.__str__(),
                'args:': args,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'validation_results': validation_results,
                'test_results': test_results,
                'test_details': test_details,
            }, path+'.pt')
            # pd_test = pd.DataFrame(data=test_details)  # , index=range(1, len(validation_results) + 1))
            # pd_test.to_csv(path + '_[TEST].csv')

    print('Training complete...')
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Total Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))

    r = {
        'auroc-val': [x['auroc'] for x in validation_results],
        'auroc-test': [x['auroc'] for x in test_results]
        # 'acc': [x['acc'] for x in validation_results],
        # 'auprc': [x['auprc'] for x in validation_results]
    }
    pdr = pd.DataFrame(data=r, index=range(1, len(validation_results)+1))
    ax = pdr.plot.line()
    plt.grid()
    fig = ax.get_figure()
    plt.ylim((0.82, 0.87))
    plt.show()
    fig.savefig(path + '.png')
    # fig.savefig(path + '.pdf')
    r_all = {
        'model-name': model_names,
        'auroc-val': [x['auroc'] for x in validation_results],
        'acc-val': [x['acc'] for x in validation_results],
        'auprc-val': [x['auprc'] for x in validation_results],
        'auroc-test': [x['auroc'] for x in test_results],
        'acc-test': [x['acc'] for x in test_results],
        'auprc-test': [x['auprc'] for x in test_results]
    }
    pd_r_all = pd.DataFrame(data=r_all, index=range(1, len(validation_results) + 1))
    pd_r_all.to_csv(path+'.csv')

elif args.mode == 'test':
    print('Beginning testing...')
    start_time = time.time()
    boostrap = True
    # ensure that the code uses test_reader
    model.to(torch.device('cpu'))

    del train_reader
    del val_reader
    # del test_reader
    # test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
    #                                         listfile=os.path.join(args.data, 'test_listfile.csv'),
    #                                         period_length=48.0)

    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part, return_names=True)
    test_dataset = Dataset(ret['data'][0], ret['data'][1], ret['names'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False)

    predicted_prob = []
    true_labels = []
    names = []
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch, name_batch in tqdm(test_loader):
            X_batch = X_batch.float()
            y_batch = y_batch.float()
            y_hat_batch, _ = model(X_batch)

            y_hat_batch = torch.sigmoid(y_hat_batch)
            y_hat_batch = y_hat_batch / y_hat_batch.sum(dim=1, keepdim=True)
            predicted_prob.append(y_hat_batch[:, 1])
            true_labels.append(y_batch)
            names.append(name_batch)

        predicted_prob = torch.cat(predicted_prob, dim=0)
        true_labels = torch.cat(true_labels, dim=0)  # with threshold 0.5, not used here
        names = np.concatenate(names)

    predictions = predicted_prob.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()
    test_results = metrics.print_metrics_binary(true_labels, predictions)
    print(test_results)
    print('Format print :.4f for results:')

    def format_print(dict):
        print("AUC of ROC = {:.4f}".format(dict['auroc']))
        print("AUC of PRC = {:.4f}".format(dict['auprc']))
        print("accuracy = {:.4f}".format(dict['acc']))
        print("min(+P, Se) = {:.4f}".format(dict['minpse']))
        print("precision class 0 = {:.4f}".format(dict['prec0']))
        print("precision class 1 = {:.4f}".format(dict['prec1']))
        print("recall class 0 = {:.4f}".format(dict['rec0']))
        print("recall class 1 = {:.4f}".format(dict['rec1']))
    format_print(test_results)

    if boostrap:
        from utils import boostrap_interval_and_std

        pd_bst = boostrap_interval_and_std(predictions, true_labels, 100)
        pd.set_option('display.max_columns', None)
        pd.set_option("precision", 4)
        print(pd_bst.describe())

    path = os.path.join("test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, true_labels, path)
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Testing elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))
else:
    raise ValueError("Wrong value for args.mode")


