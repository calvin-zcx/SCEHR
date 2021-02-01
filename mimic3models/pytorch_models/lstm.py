import numpy as np
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
# from pytorch_model_summary import summary
import torch.nn.functional as F
# from wrappers import TimeDistributed
from mimic3models.pytorch_models.torch_utils import TimeDistributed
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


def predict_labels(input):
    """
    Args:
        input: N_samples * C_classes. The input is expected to contain raw, unnormalized scores for each class.
        The input is the same as the input of  torch.nn.CrossEntropyLoss
    Outputs:
        p: N_samples * C_classes, the normalized probability
        labels: N_samples * 1, the class labels
    """
    p = F.softmax(input, dim=1)
    labels = torch.argmax(p, dim=1)
    return p, labels


def squash_packed_iid(x, fn):
    """
    Applying fn to each element of x i.i.i
    x is torch.nn.utils.rnn.PackedSequence
    """
    return PackedSequence(fn(x.data), x.batch_sizes,
                          x.sorted_indices, x.unsorted_indices)


class LSTM_PT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_classes=2,
                 dropout=0.3, target_repl=False, deep_supervision=False, task='ihm', final_act=None, **kwargs):
        super(LSTM_PT, self).__init__()

        print("==> not used params in network class:", kwargs.keys())
        self.input_dim = input_dim  # 76
        self.hidden_dim = hidden_dim  # 16
        self.num_layers = num_layers  # using 2 here
        self.num_classes = num_classes  # 2, for binary classification using (softmax + ) cross entropy
        self.dropout = dropout  # 0.3

        if final_act is None:
            # Set default activation
            if task in ['decomp', 'ihm', 'ph']:
                self.final_activation = nn.Sigmoid()
            elif task in ['los']:
                if num_classes == 1:
                    self.final_activation = nn.ReLU()
                else:
                    self.final_activation = nn.Softmax()
            else:
                raise ValueError("Wrong value for task")
        else:
            self.final_activation = final_act

        # Input layers and masking
        # X = Input(shape=(None, input_dim), name='X')
        # inputs = [X]
        # mX = Masking()(X)
        #
        # if deep_supervision:
        #     M = Input(shape=(None,), name='M')
        #     inputs.append(M)

        # Configurations
        self.bidirectional = True  # default bidirectional for the LSTM layer except output LSTM layer
        if deep_supervision:
            self.bidirectional = False

        # Main part of the network - pytorch
        self.input_dropout_layer = nn.Dropout(p=self.dropout)

        num_hidden_dim = self.hidden_dim
        if self.bidirectional:
            num_hidden_dim = num_hidden_dim // 2

        if self.num_layers > 1:
            self.main_lstm_layer = nn.LSTM(
                input_size=self.input_dim,  # 76
                hidden_size=num_hidden_dim,  # 8
                batch_first=True,  # X should be:  (batch, seq, feature)
                dropout=self.dropout,
                # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
                num_layers=self.num_layers - 1,
                bidirectional=self.bidirectional)
            num_input_dim = self.hidden_dim  # for output lstm layer
        else:
            num_input_dim = self.input_dim  # for output lstm layer

        self.inner_dropout_layer = nn.Dropout(p=self.dropout)
        # Output module of the network - Pytorch
        # output lstm is not bidirectional. So, what if num_layers = 1, then, is_bidirectional is useless.
        # return_sequences = (target_repl or deep_supervision) # always return sequence in pytorch
        # Should I add input dropout layer here?
        self.output_lstm_layer = nn.LSTM(
            input_size=num_input_dim,  # output 1 layer
            hidden_size=self.hidden_dim,
            batch_first=True,  # X should be:  (batch, seq, feature)
            # dropout=self.dropout, # only one layer, no need for inner dropout
            num_layers=1,  # output 1 layer
            bidirectional=False)  # output one direction for the output layer

        self.output_dropout_layer = nn.Dropout(p=self.dropout)

        if target_repl:
            # y = TimeDistributed(Dense(num_classes, activation=final_activation),
            #                     name='seq')(L)
            # y_last = LastTimestep(name='single')(y)
            # outputs = [y_last, y]
            raise NotImplementedError
        elif deep_supervision:
            # y = TimeDistributed(Dense(num_classes, activation=final_activation))(L)
            # y = ExtendMask()([y, M])  # this way we extend mask of y to M
            # outputs = [y]
            raise NotImplementedError
        else:
            # Only use last output.
            # y = Dense(num_classes, activation=final_activation)(L)
            # outputs = [y]
            self.output_linear = nn.Linear(self.hidden_dim, self.num_classes)  # , bias=False
            self.head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )

        self.reset_parameters()
        # taking care of initialization problem later
        # self.hidden = (
        #     torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dims[0]).to(
        #         self.device),
        #     torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dims[0]).to(
        #         self.device)
        # )

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, x):
        # add masking, padded sequence?
        if isinstance(x, PackedSequence):
            x = squash_packed_iid(x, self.input_dropout_layer)
        else:
            x = self.input_dropout_layer(x)  # (bs, seq=48, dim=76)

        if self.num_layers > 1:
            x, (hn, cn) = self.main_lstm_layer(
                x)  # (bs, seq=48, dim=16), hn, cn (num_layers * num_directions=2, batch, hidden_size = 8) add h0, c0 in the initilization
            if isinstance(x, PackedSequence):
                x = squash_packed_iid(x, self.inner_dropout_layer)
            else:
                x = self.inner_dropout_layer(x)  # (bs, seq=48, dim=16)
            x, (hn, cn) = self.output_lstm_layer(x)  # (bs, seq=48, hidden_dim=16), hn,cn (1, bs, hidden_dim=16)
        else:
            x, (hn, cn) = self.output_lstm_layer(x)  # initilization

        #
        if isinstance(x, PackedSequence):
            x, lens_unpacked = pad_packed_sequence(x, batch_first=True)
            indices = lens_unpacked - 1
            last_time_step = torch.gather(x, 1,
                                          indices.view(-1, 1).unsqueeze(2).repeat(1, 1, x.shape[2]).to(device=x.device))
            last_time_step = last_time_step.squeeze()
        else:
            last_time_step = x[:, -1, :]  # (bs, hidden_dim=16) lstm_out[:,-1,:] for batch first or h_n[-1,:,:]

        last_time_step = self.output_dropout_layer(last_time_step)
        # representation = F.normalize(last_time_step, dim=1)
        representation = F.normalize(self.head(last_time_step), dim=1)
        # representation = last_time_step
        # out = self.output_linear(representation)
        out = self.output_linear(last_time_step)  # (bs, 2)
        # out = self.head(last_time_step)
        # representation = F.normalize(out, dim=1)
        # No softmax activation if for pytorch crossentropy loss
        # original used in keras. should use sigmoid activation before keras binary cross entropy loss
        out = self.final_activation(out).squeeze()
        return out, representation

    def say_name(self):
        return "{}.i{}.h{}.L{}.c{}{}".format('LSTM',
                                             self.input_dim,
                                             self.hidden_dim,
                                             self.num_layers,
                                             self.num_classes,
                                             ".D{}".format(self.dropout) if self.dropout > 0 else "-"
                                             )


#
# class LSTM_PT(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=2, num_classes=2,
#                  dropout=0.3, target_repl=False, deep_supervision=False, **kwargs):
#         # __init__(self, hidden_dim, batch_norm, dropout, rec_dropout, task,
#         #          target_repl=False, deep_supervision=False, num_classes=2,
#         #          depth=1, input_dim=76, **kwargs):
#         super(LSTM_PT, self).__init__()
#
#         print("==> not used params in network class:", kwargs.keys())
#         self.input_dim = input_dim  # 76
#         self.hidden_dim = hidden_dim  # 16
#         self.num_layers = num_layers  # using 2 here
#         self.num_classes = num_classes  # 2, for binary classification using (softmax + ) cross entropy
#         self.dropout = dropout  # 0.3
#         # self.rec_dropout = rec_dropout # rec_dropout=0.0, No recurrent dropout in Pytorch
#
#
#         # if task in ['decomp', 'ihm', 'ph']:
#         #     final_activation = 'sigmoid'
#         #     self.final_activation = nn.Sigmoid
#         # elif task in ['los']:
#         #     if num_classes == 1:
#         #         final_activation = 'relu'
#         #         self.final_activation = nn.ReLU
#         #     else:
#         #         final_activation = 'softmax'
#         #         self.final_activation = nn.Softmax
#         # else:
#         #     raise ValueError("Wrong value for task")
#
#
#         # Input layers and masking
#         # X = Input(shape=(None, input_dim), name='X')
#         # inputs = [X]
#         # mX = Masking()(X)
#         #
#         # if deep_supervision:
#         #     M = Input(shape=(None,), name='M')
#         #     inputs.append(M)
#
#         # Configurations
#         self.bidirectional = True  # default bidirectional for the LSTM layer except output LSTM layer
#         if deep_supervision:
#             self.bidirectional = False
#
#         # Main part of the network - pytorch
#         self.input_dropout_layer = nn.Dropout(p=self.dropout)
#
#         num_hidden_dim = self.hidden_dim
#         if self.bidirectional:
#             num_hidden_dim = num_hidden_dim // 2
#
#         if self.num_layers > 1:
#             self.main_lstm_layer = nn.LSTM(
#                 input_size=self.input_dim,  # 76
#                 hidden_size=num_hidden_dim,  # 8
#                 batch_first=True,  # X should be:  (batch, seq, feature)
#                 dropout=self.dropout,  # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
#                 num_layers=self.num_layers - 1,
#                 bidirectional=self.bidirectional)
#             num_input_dim = self.hidden_dim  # for output lstm layer
#         else:
#             num_input_dim = self.input_dim  # for output lstm layer
#
#         self.inner_dropout_layer = nn.Dropout(p=self.dropout)
#         # Output module of the network - Pytorch
#         # output lstm is not bidirectional. So, what if num_layers = 1, then, is_bidirectional is useless.
#         # return_sequences = (target_repl or deep_supervision) # always return sequence in pytorch
#         # Should I add input dropout layer here?
#         self.output_lstm_layer = nn.LSTM(
#             input_size=num_input_dim,  # output 1 layer
#             hidden_size=self.hidden_dim,
#             batch_first=True,  # X should be:  (batch, seq, feature)
#             # dropout=self.dropout, # only one layer, no need for inner dropout
#             num_layers=1,  # output 1 layer
#             bidirectional=False)  # output one direction for the output layer
#
#         self.output_dropout_layer = nn.Dropout(p=self.dropout)
#
#         if target_repl:
#             # y = TimeDistributed(Dense(num_classes, activation=final_activation),
#             #                     name='seq')(L)
#             # y_last = LastTimestep(name='single')(y)
#             # outputs = [y_last, y]
#             raise NotImplementedError
#         elif deep_supervision:
#             # y = TimeDistributed(Dense(num_classes, activation=final_activation))(L)
#             # y = ExtendMask()([y, M])  # this way we extend mask of y to M
#             # outputs = [y]
#             raise NotImplementedError
#         else:
#             # Only use last output.
#             # y = Dense(num_classes, activation=final_activation)(L)
#             # outputs = [y]
#             self.output_linear = nn.Linear(self.hidden_dim, self.num_classes)
#             self.head = nn.Sequential(
#                 nn.Linear(self.hidden_dim, self.hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.hidden_dim, self.hidden_dim)
#             )
#
#         self.reset_parameters()
#         # taking care of initialization problem later
#         # self.hidden = (
#         #     torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dims[0]).to(
#         #         self.device),
#         #     torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dims[0]).to(
#         #         self.device)
#         # )
#
#     def reset_parameters(self):
#         """
#         Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
#         """
#         ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
#         hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
#         b = (param.data for name, param in self.named_parameters() if 'bias' in name)
#         for t in ih:
#             nn.init.xavier_uniform_(t)
#         for t in hh:
#             nn.init.orthogonal_(t)
#         for t in b:
#             nn.init.constant_(t, 0)
#
#     def encoder(self, x):
#         # add masking, padded sequence?
#         x = self.input_dropout_layer(x)  # (bs, seq=48, dim=76)
#         if self.num_layers > 1:
#             x, (hn, cn) = self.main_lstm_layer(
#                 x)  # (bs, seq=48, dim=16), hn, cn (num_layers * num_directions=2, batch, hidden_size = 8) add h0, c0 in the initilization
#             x = self.inner_dropout_layer(x)  # (bs, seq=48, dim=16)
#             x, (hn, cn) = self.output_lstm_layer(x)  # (bs, seq=48, hidden_dim=16), hn,cn (1, bs, hidden_dim=16)
#         else:
#             x, (hn, cn) = self.output_lstm_layer(x)  # initilization
#
#         x = self.output_dropout_layer(x)
#
#         last_time_step = x[:, -1, :]  # (bs, hidden_dim=16) lstm_out[:,-1,:] for batch first or h_n[-1,:,:]
#         return last_time_step
#
#     def forward_contrastive_pretrain(self, x):
#         feat = self.encoder(x)
#         feat = self.head(feat)
#         feat = F.normalize(feat, dim=1)
#         return feat
#
#     def forward_fine_tune(self, x, isFineTune=True):
#         if isFineTune:
#             with torch.no_grad():
#                 self.eval()
#                 feat = self.encoder(x)
#                 self.train()
#             output = self.output_linear(feat)
#         else:
#             feat = self.encoder(x)
#             output = self.output_linear(feat)
#         return output
#
#     def forward(self, x, is_return_representation=False):
#         last_time_step = self.encoder(x)  # x[:, -1, :]  # (bs, hidden_dim=16) lstm_out[:,-1,:] for batch first or h_n[-1,:,:]
#         representation = F.normalize(last_time_step, dim=1)
#         # out = self.output_linear(representation)
#         out = self.output_linear(last_time_step)  # (bs, 2)
#         # out = self.head(last_time_step)
#         # representation = F.normalize(out, dim=1)
#         # No softmax activation if for pytorch crossentropy loss
#         # original used in keras. should use sigmoid activation before keras binary cross entropy loss
#         # y_pred = self.final_activation(y_pred)
#         return (out, representation) if is_return_representation else out
#
#
#     # def forward(self, x, is_return_representation=False):
#     #     # add masking, padded sequence?
#     #     x = self.input_dropout_layer(x)  # (bs, seq=48, dim=76)
#     #     if self.num_layers > 1:
#     #         x, (hn, cn) = self.main_lstm_layer(x)  # (bs, seq=48, dim=16), hn, cn (num_layers * num_directions=2, batch, hidden_size = 8) add h0, c0 in the initilization
#     #         x = self.inner_dropout_layer(x)  # (bs, seq=48, dim=16)
#     #         x, (hn, cn) = self.output_lstm_layer(x)  # (bs, seq=48, hidden_dim=16), hn,cn (1, bs, hidden_dim=16)
#     #     else:
#     #         x, (hn, cn) = self.output_lstm_layer(x)  # initilization
#     #
#     #     x = self.output_dropout_layer(x)
#     #
#     #     last_time_step = x[:, -1, :]  # (bs, hidden_dim=16) lstm_out[:,-1,:] for batch first or h_n[-1,:,:]
#     #     representation = F.normalize(last_time_step, dim=1)
#     #     # out = self.output_linear(representation)
#     #     out = self.output_linear(last_time_step)  # (bs, 2)
#     #     # out = self.head(last_time_step)
#     #     # representation = F.normalize(out, dim=1)
#     #     # No softmax activation if for pytorch crossentropy loss
#     #     # original used in keras. should use sigmoid activation before keras binary cross entropy loss
#     #     # y_pred = self.final_activation(y_pred)
#     #     return (out, representation) if is_return_representation else out
#
#     def say_name(self):
#         return "{}.i{}.h{}.L{}.c{}{}".format('LSTM_PT',
#                                                self.input_dim,
#                                                self.hidden_dim,
#                                                self.num_layers,
#                                                self.num_classes,
#                                                ".D{}".format(self.dropout) if self.dropout > 0 else "-"
#                                                )
#

class LSTM_PT_Demo(nn.Module):
    def __init__(self, n_features=25, hidden_dims=[80, 80], seq_length=250, batch_size=64, n_predictions=1,
                 device=torch.device("cpu"), dropout=0.3, bidirectional=False):  # cuda:0
        super(LSTM_PT_Demo, self).__init__()

        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.seq_length = seq_length
        self.num_layers = len(self.hidden_dims)
        self.batch_size = batch_size
        self.device = device
        self.bidirectional = bidirectional

        print(f'number of layers :{self.num_layers}')

        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dims[0],
            batch_first=True,
            dropout=dropout,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)

        self.linear = nn.Linear(self.hidden_dims[0], n_predictions)

        self.hidden = (
            torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dims[0]).to(
                self.device),
            torch.randn(self.num_layers * (2 ** int(self.bidirectional)), self.batch_size, self.hidden_dims[0]).to(
                self.device)
        )

    # def init_hidden_state(self):
    #     # initialize hidden states (h_n, c_n)
    #
    #     self.hidden = (
    #         torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0]).to(self.device),
    #         torch.randn(self.num_layers, self.batch_size, self.hidden_dims[0]).to(self.device)
    #     )

    def forward(self, sequences):
        batch_size, seq_len, n_features = sequences.size()  # batch_first

        # LSTM inputs: (input, (h_0, c_0))
        # input of shape (seq_len, batch, input_size)....   input_size = num_features
        # or (batch, seq_len, input_size) if batch_first = True

        lstm1_out, (h1_n, c1_n) = self.lstm1(sequences,
                                             (self.hidden[0], self.hidden[1]))  # hidden[0] = h_n, hidden[1] = c_n

        # Output: output, (h_n, c_n)
        # output is of shape (batch_size, seq_len, hidden_size) with batch_first = True

        last_time_step = lstm1_out[:, -1, :]  # lstm_out[:,-1,:] or h_n[-1,:,:]
        y_pred = self.linear(last_time_step)
        # output is shape (N, *, H_out)....this is (batch_size, out_features)

        return y_pred


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight.data)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_uniform_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)


def predict_model(model, data, batch_size, device):
    print('Starting predictions...')
    data_loader = DataLoader(dataset=data, batch_size=batch_size, drop_last=True)
    y_hat = torch.empty(data_loader.batch_size, 1).to(device)

    with torch.no_grad():
        for X_batch in data_loader:
            y_hat_batch = model(X_batch)
            y_hat = torch.cat([y_hat, y_hat_batch])

    y_hat = torch.flatten(y_hat[batch_size:, :]).cpu().numpy()  # y_hat[batchsize:] is to remove first empty 'section'
    print('Predictions complete...')
    return y_hat


def train_model(model, train_data, train_labels, test_data, test_labels, batch_size, num_epochs, device):
    model.apply(initialize_weights)

    training_losses = []
    validation_losses = []

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    train_hist = np.zeros(num_epochs)

    X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_labels, train_size=0.8)

    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(X_validation, y_validation)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model.train()

    print("Beginning model training...")

    for t in range(num_epochs):
        train_losses_batch = []
        for X_batch_train, y_batch_train in train_loader:
            y_hat_train = model(X_batch_train)
            loss = loss_function(y_hat_train.float(), y_batch_train)
            train_loss_batch = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses_batch.append(train_loss_batch)

        training_loss = np.mean(train_losses_batch)
        training_losses.append(training_loss)

        with torch.no_grad():
            val_losses_batch = []
            for X_val_batch, y_val_batch in val_loader:
                model.eval()
                y_hat_val = model(X_val_batch)
                val_loss_batch = loss_function(y_hat_val.float(), y_val_batch).item()
                val_losses_batch.append(val_loss_batch)
            validation_loss = np.mean(val_losses_batch)
            validation_losses.append(validation_loss)

        print(f"[{t + 1}] Training loss: {training_loss} \t Validation loss: {validation_loss} ")

    print('Training complete...')
    return model.eval()


if __name__ == "__main__":
    torch.manual_seed(0)

    bs = 3
    seq = 48
    input_dim = 76
    num_layers = 2
    hidden_dim = 16

    X = torch.randn(bs, seq, input_dim)  # batch, seq_len,  input_size
    h0 = torch.randn(2 * num_layers, bs, hidden_dim)  # num_layers * num_directions, batch, hidden_size
    c0 = torch.randn(2 * num_layers, bs, hidden_dim)  # num_layers * num_directions, batch, hidden_size

    mylstm = LSTM_PT(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=2,
                     dropout=0.3, target_repl=False, deep_supervision=False)
    # output: (seq_len, batch, num_directions * hidden_size)
    Y = mylstm(X)  # , (h0, c0)
    labels = predict_labels(Y)

    # d = 4
    # rnn = nn.LSTM(10, d, 2)  # input_size, hidden_size, num_layers
    # input = torch.randn(5, 3, 10)  # seq_len, batch, input_size
    # h0 = torch.randn(2, 3, d)  # num_layers * num_directions, batch, hidden_size
    # c0 = torch.randn(2, 3, d) # num_layers * num_directions, batch, hidden_size
    # # output: (seq_len, batch, num_directions * hidden_size)
    # output, (hn, cn) = rnn(input, (h0, c0))  # (5,3,d) ((2, 3, d), (2, 3, d))
    #
    # rnn2 = nn.LSTM(10, d, 2, bidirectional=True)  # input_size, hidden_size, num_layers
    # input2 = torch.randn(5, 3, 10)  # seq_len, batch, input_size
    # h02 = torch.randn(2*2, 3, d)  # num_layers * num_directions, batch, hidden_size
    # c02 = torch.randn(2*2, 3, d)  # num_layers * num_directions, batch, hidden_size
    # # output: (seq_len, batch, num_directions * hidden_size)
    # output2, (hn2, cn2) = rnn2(input2, (h02, c02))  # (5,3,d*2) ((2*2, 3, d), (2*2, 3, d))

    # n = 19
    # f = LSTM_PT(n_features=25, hidden_dims=[80, 80], seq_length=250, batch_size=19, n_predictions=1,
    #             device=torch.device("cpu"), dropout=0.3, bidirectional=True)
    #
    # lstm1 = nn.LSTM(
    #     input_size=25,
    #     hidden_size=80,
    #     batch_first=True,
    #     dropout=.5,
    #     num_layers=2,
    #     bidirectional=False) #True)
    #
    # # summary(lstm1.to(torch.device("cuda:0")), (250,25)) #, device="cpu")
    # print(summary(lstm1, torch.zeros((1, 250, 25)), show_hierarchical=True))
    #
    # x = torch.randn(n, 250, 25)  # 454 samples of 250 time steps with 25 features.
    # y = torch.randn(n)
    # y_hat = f(x)
    # print(y_hat)

    # nn.NLLLoss()
    #
    # sm = nn.Softmax(dim=1)
    # m = nn.LogSoftmax(dim=1)
    # loss = nn.NLLLoss()
    # inpt = torch.randn(3, 5) #, requires_grad=True)
    # target = torch.tensor([1, 0, 4])
    # output = loss(m(inpt), target)
