import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.name[index]

    def __len__(self):
        return len(self.x)


def model_summary(model):
    print("=" * 50)
    print('model_summary')
    table = []
    total_params = 0
    i = 0
    print("Index\tModules\tParameters\tCumsum")
    for name, parameter in model.named_parameters():
        i += 1
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.append([name, param])
        total_params+=param
        print("{}\t{}\t{}\t{}".format(i, name, param,total_params))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: ', pytorch_total_params)
    print('Total trainable parameters: ', pytorch_total_trainable_params)
    print("=" * 50)
    return table, total_params


class TimeDistributed(nn.Module):
    """
    This wrapper applies a layer to every temporal slice of an input.
    The input should be at least 3D, and the dimension of index one
    will be considered to be the temporal dimension when batch_first = True:
    (batch_samples, timesteps, output_size)
    Else when batch_first = False, the dimension of index 0 is the temporal one:
    (timesteps, batch_samples, output_size)
    """
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            print('x.size()', x.size())
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


def shuffle_within_labels(x, y):
    unique_class = y.unique()
    id_list = []
    shuffle_list = []
    sfx = x.clone()  # .detach().clone()
    for v in unique_class:
        vid = torch.nonzero(y==v, as_tuple=True)[0]
        id_list.append(vid)
        vs = torch.randperm(vid.size()[0])
        shuffle_list.append(vid[vs])
    id_list = torch.cat(id_list)
    shuffle_list = torch.cat(shuffle_list)
    sfx[id_list] = sfx[shuffle_list]
    return sfx


def shuffle_time_dim(x):
    # x has shape (bs, T, dim)
    # shuffle T dimension for each data in this batch
    T = x.shape[1]
    ridx = torch.randperm(T)
    return x[:,ridx, :]


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
