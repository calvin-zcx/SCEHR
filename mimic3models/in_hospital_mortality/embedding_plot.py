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
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
from mimic3models.pytorch_models.lstm import LSTM_PT, predict_labels
from mimic3models.pytorch_models.losses import SupConLoss, SupNCELoss, CBCE_loss, CBCE_WithLogitsLoss
from tqdm import tqdm
from mimic3models.time_report import TimeReport
from mimic3models.pytorch_models.torch_utils import Dataset, optimizer_to, model_summary, TimeDistributed, shuffle_within_labels,shuffle_time_dim
import matplotlib.pyplot as plt
import pandas as pd
import functools
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
print = functools.partial(print, flush=True)

# %%
# Arguments:
parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='./pytorch_states/embedding/')
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

if '.BCE+SCL.' in args.load_state:
    type = 'BCE'
elif '.CBCE+SCL.' in args.load_state:
    type = 'CBCE'
elif '.MCE+SCL.' in args.load_state:
    type = 'MCE'
else:
    type = None
    print('wrong load model type')
    # exit(0)

print('Model type: ', type)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: Got {} CUDA devices! Probably run with --cuda".format(torch.cuda.device_count()))
device = torch.device("cuda" if args.cuda else "cpu")
print('Using device: ', device)

# Build the model
if args.network == "lstm":
    if type == 'BCE':
        model = LSTM_PT(input_dim=76, hidden_dim=args.dim, num_layers=args.depth, num_classes=1,
                    dropout=args.dropout, target_repl=False, deep_supervision=False, task='ihm')
    else:
        model = LSTM_PT(input_dim=76, hidden_dim=args.dim, num_layers=args.depth, num_classes=2,
                        dropout=args.dropout, target_repl=False, deep_supervision=False, task='ihm',
                        final_act=nn.Identity())
else:
    raise NotImplementedError


if target_repl:
    raise NotImplementedError
else:
    criterion_BCE = nn.BCELoss()
    criterion_SCL = SupConLoss(temperature=0.1)   # temperature=0.01)  # temperature=opt.temp
    # criterion_SupNCE = SupNCELoss(temperature=1)
    # criterion_MCE = nn.CrossEntropyLoss()

    def get_loss(y_pre, labels, representation, alpha=0):
        # CBCE_WithLogitsLoss is more numerically stable than CBCE_Loss when model is complex/overfitting
        loss = criterion_BCE(y_pre, labels)
        if alpha > 0:
            if labels.sum().item() < 2:
                print('Warning: # positives < 2, NOT USING Supervised Contrastive Regularizer')
            else:
                if len(representation.shape) == 2:
                    representation = representation.unsqueeze(1)
                scl_loss = criterion_SCL(representation, labels)
                loss = loss + alpha * scl_loss
        return loss

    def get_probability(wx, type):
        if type == 'BCE':
            print('Type BCE')
            return wx
        elif type == 'CBCE':
            print('Type CBCE')
            y = torch.sigmoid(wx)
            y = y / y.sum(dim=1, keepdim=True)
            return y[:, 1]
        elif type == 'MCE':
            print('Type MCE')
            y = F.softmax(wx, dim=1)
            return y[:, 1]

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

print('Beginning testing...')
start_time = time.time()

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
embeddings = []
with torch.no_grad():
    model.eval()
    for X_batch, y_batch, name_batch in tqdm(test_loader):
        X_batch = X_batch.float()
        y_batch = y_batch.float()
        y_hat_batch, embedding = model(X_batch)

        predicted_prob.append(y_hat_batch)
        true_labels.append(y_batch)
        names.append(name_batch)
        embeddings.append(embedding)

    predicted_prob = torch.cat(predicted_prob, dim=0)
    true_labels = torch.cat(true_labels, dim=0) # with threshold 0.5, not used here
    names = np.concatenate(names)
    embeddings = torch.cat(embeddings, dim=0)

predicted_prob = get_probability(predicted_prob, type)
predictions = predicted_prob.cpu().detach().numpy()

true_labels = true_labels.cpu().detach().numpy().flatten()
test_results = metrics.print_metrics_binary(true_labels, predictions)
embeddings = embeddings.cpu().detach().numpy()
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



# dim = embeddings.shape[1]
emethod = 'pca' # "tsne"
# tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=300)
# tsne_results = tsne.fit_transform(embeddings)

pca = PCA(n_components=2)
tsne_results = pca.fit_transform(embeddings)

df = pd.DataFrame(data={'y': true_labels,
                        '{}-2d-one'.format(emethod): tsne_results[:, 0],
                        '{}-2d-two'.format(emethod): tsne_results[:, 1]})

# plt.figure(figsize=(16, 10))
ax = sns.scatterplot(
    x="{}-2d-one".format(emethod),
    y="{}-2d-two".format(emethod),
    hue="y",
    # palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3
)

fig = ax.get_figure()
plt.show()
path = os.path.join(emethod, os.path.basename(args.load_state))
dirname = os.path.dirname(path)
if not os.path.exists(dirname):
    os.makedirs(dirname)
fig.savefig(path + '.png')
fig.savefig(path + '.pdf')
# utils.save_results(names, predictions, true_labels, path)
h_, m_, s_ = TimeReport._hms(time.time() - start_time)
print('Testing elapsed time: {:02d}h-{:02d}m-{:02d}s'.format(h_, m_, s_))



