from __future__ import absolute_import
from __future__ import print_function
import sys
# for linux env.
sys.path.insert(0, '../..')

import numpy as np
import argparse
import os
import time
from mimic3models.phenotyping import utils
from mimic3benchmark.readers import PhenotypingReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import common_utils
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
# from torch.utils.data import DataLoader, Dataset, TensorDataset
from mimic3models.pytorch_models.lstm import LSTM_PT
from mimic3models.pytorch_models.losses import SupConLoss_MultiLabel, CrossEntropy_multilabel
# SupNCELoss, CBCE_loss, CBCE_WithLogitsLoss
from tqdm import tqdm
from mimic3models.time_report import TimeReport
from mimic3models.pytorch_models.torch_utils import Dataset, optimizer_to, model_summary, TimeDistributed, shuffle_within_labels,shuffle_time_dim
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
parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/phenotyping/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='./pytorch_states/MCE/')
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
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay of adam')
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
train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                 listfile=os.path.join(args.data, 'train_listfile.csv'))

val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                               listfile=os.path.join(args.data, 'val_listfile.csv'))

test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                listfile=os.path.join(args.data, 'test_listfile.csv'))

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
args_dict['task'] = 'ihm'
args_dict['num_classes'] = 25
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
    model = LSTM_PT(input_dim=76, hidden_dim=args.dim, num_layers=args.depth, num_classes=25 * 2,
                    dropout=args.dropout, target_repl=False, deep_supervision=False, task='ph',
                    final_act=nn.Identity())
else:
    raise NotImplementedError


if target_repl:
    raise NotImplementedError
else:
    # criterion_BCE = nn.BCELoss()
    criterion_SCL_MultiLabel = SupConLoss_MultiLabel(temperature=0.1)   # temperature=0.01)  # temperature=opt.temp

    def get_loss(y_pre, labels, representation, alpha=0):
        # CBCE_WithLogitsLoss is more numerically stable than CBCE_Loss when model is complex/overfitting
        loss = CrossEntropy_multilabel(y_pre, labels)
        if alpha > 0:
            if len(representation.shape) == 2:
                representation = representation.unsqueeze(1)
            scl_loss = criterion_SCL_MultiLabel(representation, labels)
            loss = loss + alpha * scl_loss
        return loss

    def get_probability_from_logits(wx):
        # wx: (bs, c*2)
        wx_pos, wx_neg = torch.chunk(wx, 2, dim=1)
        y = torch.softmax(
            torch.cat((wx_pos.unsqueeze(1), wx_neg.unsqueeze(1)), dim=1),
            dim=1)
        y_pos, y_neg = torch.chunk(y, 2, dim=1)
        y_pos, y_neg = y_pos.squeeze(), y_neg.squeeze()
        return y_pos


# set lr and weight_decay later # or use other optimization say adamW later?
optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)
training_losses = []
validation_losses = []
test_losses = []
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
    # print(model)
    print('Load model done!')
    # n_trained_chunks = int(re.match(".*Epoch([0-9]+).*", args.load_state).group(1))


model.to(device)
try:
    criterion_SCL_MultiLabel.to(device)
except NameError:
    print("No criterion_SCL_MultiLabel")
print(model)
model_summary(model)

# %%
# Training & Testing parts:
if args.mode == 'train':
    print('Training part: beginning loading training & validation datasets...')
    start_time = time.time()

    train_data_gen = utils.BatchGen(train_reader, discretizer, normalizer, args.batch_size,
                                    args.small_part, target_repl, shuffle=True, return_names=True)
    val_data_gen = utils.BatchGen(val_reader, discretizer, normalizer, args.batch_size,
                                  args.small_part, target_repl, shuffle=False, return_names=True)
    test_data_gen = utils.BatchGen(test_reader, discretizer, normalizer, args.batch_size,
                                   args.small_part, target_repl, shuffle=False, return_names=True)

    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Load data done. Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))
    print('Data summary:')
    print('len(train_data_gen):', len(train_data_gen),
          'len(val_data_gen):', len(val_data_gen),
          'len(test_data_gen):', len(test_data_gen))
    print('batch size:', args.batch_size, 'epoch:', args.epochs, 'iters/epoch:', train_data_gen.steps)

    print("Beginning model training...")
    iter_per_epoch = train_data_gen.steps
    tr = TimeReport(total_iter=args.epochs * train_data_gen.steps)
    for epoch in (range(1+start_from_epoch, 1+args.epochs)): #tqdm
        model.train()
        train_losses_batch = []
        for i in (range(train_data_gen.steps)):  # tqdm
            # print("predicting {} / {}".format(i, train_data_gen.steps), end='\r')
            ret = next(train_data_gen)
            X_batch_train, labels_batch_train, x_length = ret["data"]
            name_batch_train = ret["names"]

            X_batch_train = torch.tensor(X_batch_train, dtype=torch.float32)
            labels_batch_train = torch.tensor(labels_batch_train, dtype=torch.long)
            X_batch_train = rnn_utils.pack_padded_sequence(X_batch_train, x_length, batch_first=True)

            optimizer.zero_grad()

            X_batch_train = X_batch_train.to(device)
            labels_batch_train = labels_batch_train.to(device)
            bsz = labels_batch_train.shape[0]

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
            for i in range(val_data_gen.steps):
                ret = next(val_data_gen)
                X_batch_val, labels_batch_val, x_length = ret["data"]
                name_batch_val = ret["names"]

                X_batch_val = torch.tensor(X_batch_val, dtype=torch.float32)
                labels_batch_val = torch.tensor(labels_batch_val, dtype=torch.long)
                X_batch_val = rnn_utils.pack_padded_sequence(X_batch_val, x_length, batch_first=True)

                X_batch_val = X_batch_val.to(device)
                labels_batch_val = labels_batch_val.to(device)
                bsz = labels_batch_val.shape[0]

                y_hat_val, y_representation_val = model(X_batch_val)
                val_loss_batch = get_loss(y_hat_val, labels_batch_val, y_representation_val, args.coef_contra_loss)
                val_losses_batch.append(val_loss_batch.item())
                # get predicted probability
                y_hat_val = get_probability_from_logits(y_hat_val)
                predicted_prob_val.append(y_hat_val)
                true_labels_val.append(labels_batch_val)

            validation_loss = np.mean(val_losses_batch)
            validation_losses.append(validation_loss)

            predicted_prob_val = torch.cat(predicted_prob_val, dim=0).cpu().detach().numpy()
            true_labels_val = torch.cat(true_labels_val, dim=0).cpu().detach().numpy()
            val_result = metrics.print_metrics_multilabel(true_labels_val, predicted_prob_val, verbose=0)
            print(val_result)
            validation_results.append(val_result)

        # Additional test part. God View. should not used for model selection
        print('Test results:')
        with torch.no_grad():
            model.eval()
            predicted_prob_test = []
            true_labels_test = []
            name_test = []
            test_losses_batch = []
            for i in range(test_data_gen.steps):
                ret = next(test_data_gen)
                X_batch_test, y_batch_test, x_length = ret["data"]
                name_batch_test = ret["names"]

                X_batch_test = torch.tensor(X_batch_test, dtype=torch.float32)
                y_batch_test = torch.tensor(y_batch_test, dtype=torch.long)
                X_batch_test = rnn_utils.pack_padded_sequence(X_batch_test, x_length, batch_first=True)

                X_batch_test = X_batch_test.to(device)
                y_batch_test = y_batch_test.to(device)
                bsz = y_batch_test.shape[0]

                y_hat_batch_test, y_representation_test = model(X_batch_test)
                test_loss_batch = get_loss(y_hat_batch_test, y_batch_test, y_representation_test, args.coef_contra_loss)
                test_losses_batch.append(test_loss_batch.item())

                # get predicted probability
                y_hat_batch_test = get_probability_from_logits(y_hat_batch_test)
                predicted_prob_test.append(y_hat_batch_test)
                true_labels_test.append(y_batch_test)
                name_test.append(name_batch_test)

            test_loss = np.mean(test_losses_batch)
            test_losses.append(test_loss)

            predicted_prob_test = torch.cat(predicted_prob_test, dim=0)
            true_labels_test = torch.cat(true_labels_test, dim=0)  # with threshold 0.5, not used here
            predictions_test = (predicted_prob_test.cpu().detach().numpy())
            true_labels_test = true_labels_test.cpu().detach().numpy()
            name_test = np.concatenate(name_test)
            test_result = metrics.print_metrics_multilabel(true_labels_test, predictions_test, verbose=1)

            print(test_result)
            test_results.append(test_result)

        print('Epoch [{}/{}], {} Iters/Epoch, training_loss: {:.3f}, validation_loss: {:.3f}, test_loss: {:.3f},'
              '{:.2f} sec/iter, {:.2f} iters/sec: '.
              format(epoch, args.epochs, iter_per_epoch,
                     training_loss, validation_loss, test_loss,
                     tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
        tr.print_summary()
        print("=" * 50)

        model_final_name = model.say_name()
        path = os.path.join(args.output_dir + model_final_name
                            + '.MCE+SCL.a{}.bs{}.wdcy{}.epo{}.'
                              'Val-AucMac{:.4f}.AucMic{:.4f}.'
                              'Tst-AucMac{:.4f}.AucMic{:.4f}'.
                            format(args.coef_contra_loss, args.batch_size, args.weight_decay, epoch,
                                   val_result['ave_auc_macro'], val_result['ave_auc_micro'],
                                   test_result['ave_auc_macro'], test_result['ave_auc_micro']))

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
            print('\n-----Save model: \n{}\n'.format(path+'.pt'))

            # pd_test = pd.DataFrame(data=test_details)  # , index=range(1, len(validation_results) + 1))
            # pd_test.to_csv(path + '_[TEST].csv')

    print('Training complete...')
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Total Elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))

    r = {
        'ave_auc_macro-val': [x['ave_auc_macro'] for x in validation_results],
        'ave_auc_macro-test': [x['ave_auc_macro'] for x in test_results],
        'ave_auc_micro-val': [x['ave_auc_micro'] for x in validation_results],
        'ave_auc_micro-test': [x['ave_auc_micro'] for x in test_results],
    }
    pdr = pd.DataFrame(data=r, index=range(1, len(validation_results)+1))
    ax = pdr.plot.line()
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':') #, linewidth='0.5', color='black')
    plt.grid()
    fig = ax.get_figure()
    plt.ylim((0.74, 0.85))
    plt.show()
    fig.savefig(path + '.png')
    # fig.savefig(path + '.pdf')
    r_all = {
        'model-name': model_names,
        'ave_auc_macro-val': [x['ave_auc_macro'] for x in validation_results],
        'ave_auc_micro-val': [x['ave_auc_micro'] for x in validation_results],
        'ave_auc_weighted-val': [x['ave_auc_weighted'] for x in validation_results],
        'ave_auc_macro-test': [x['ave_auc_macro'] for x in test_results],
        'ave_auc_micro-test': [x['ave_auc_micro'] for x in test_results],
        'ave_auc_weighted-test': [x['ave_auc_weighted'] for x in test_results],
    }
    pd_r_all = pd.DataFrame(data=r_all, index=range(1, len(validation_results) + 1))
    pd_r_all.to_csv(path+'.csv')
    print('Dump', path + '[.png/.csv] done!')

elif args.mode == 'test':
    print('Beginning testing...')
    start_time = time.time()
    # ensure that the code uses test_reader
    model.to(torch.device('cpu'))

    del train_reader
    del val_reader

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
            y = torch.tensor(y, dtype=torch.long)
            x_pack = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)

            pred, _ = model(x_pack)
            pred = get_probability_from_logits(pred)
            predictions.append(pred)
            labels.append(y)
            names += list(cur_names)
            ts += list(cur_ts)

        predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
        labels = torch.cat(labels, dim=0).cpu().detach().numpy()

    results = metrics.print_metrics_multilabel(labels, predictions)
    print(results)
    print('Format print :.4f for results:')

    # def format_print(dict):
    #     print("AUC of ROC = {:.4f}".format(dict['auroc']))
    #     print("AUC of PRC = {:.4f}".format(dict['auprc']))
    #     print("accuracy = {:.4f}".format(dict['acc']))
    # format_print(test_results)
    # if boostrap:
    #     from utils import boostrap_interval_and_std
    #     pd_bst = boostrap_interval_and_std(predictions, true_labels, 100)
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option("precision", 4)
    #     print(pd_bst.describe())

    path = os.path.join("test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, ts, predictions, labels, path)
    h_, m_, s_ = TimeReport._hms(time.time() - start_time)
    print('Testing elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))

else:
    raise ValueError("Wrong value for args.mode")


