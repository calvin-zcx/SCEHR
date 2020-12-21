import torch.nn as nn


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