'''
    Compute receptive field of neural network neurons.
'''

import torch
import torch.nn as nn

def receptive_field(f, n=32, ch=3):
    """
    Compute the receptive field of `f`, usually the network function up to a certain layer.
    It does so by setting all the weights to a constant, replacing max-pooling by avg-pooling and deactivating batch-norm.

    :param torch.nn.Module f: network layer function
    :param int n: input size is n x n x ch
    :param int ch: input channels
    :return torch.Tensor: receptive field (as an n x n image) for each neuron in the activation map of layer f. Values of receptive_field are in [0, 1].
        Shape: [activation_map_size, activation_map_size, n, n]

    """

    constant_weights(f)
    replace_pooling(f)
    x = one_pixel_inputs(n=n)

    out = f(x[:, None].expand(-1, ch, -1, -1))[:, 0].detach()
    m = out.shape[-1]
    out = out.reshape(n, n, -1).permute(2, 0, 1)
    out /= out.max()

    return out.reshape(m, m, n, n)


def receptive_field_center(rf):
    """
    :param torch.Tensor rf: receptive field for each neuron
        Shape: [number_neurons, n, n]
    :returns torch.Tensor: receptive fields centers
        Shape: [number_neurons, 2]
    """
    n = rf.shape[-1]
    X = torch.meshgrid(torch.arange(n, device=rf.device), torch.arange(n, device=rf.device))
    return torch.stack([torch.nanmean((rf * X[i]).sum(dim=2) / rf.sum(dim=2), dim=1) for i in [1, 0]]).t()


def constant_weights(model):
    """
    Set all weights to a constant and biases to zero in `model`.
    Deactivate batch norm.
    :param torch.nn.Module model: neural network model written in PyTorch.
    :return None: in-place function.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module = module.eval()
        try:
            nn.init.constant_(module.weight, 0.01)
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
        except:
            pass
        
        
def replace_pooling(m, name=None):
    """
    Replace MaxPool2d by AvgPool2d in `m`.
    :param torch.nn.Module m: neural network model written in PyTorch.
    """
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if target_attr == torch.nn.MaxPool2d:
            print('Replaced MaxPool with AvgPool at layer:', name)
            setattr(m, attr_str, torch.nn.AvgPool2d)
            m.count_include_pad = True
            m.divisor_override = None
    for n, ch in m.named_children():
        replace_pooling(ch, n)
        
        
def one_pixel_inputs(n=32):
    """
    :param int n: input image side size (image is n x n)
    :return torch.Tensor: a batch of n^2 images such that `imgs[i]` has the i-th pixel on (=1) and the rest is zero.
    """
    x = torch.zeros(n, n, n, n)
    for i in range(n):
        for j in range(n):
            x[i, j, i, j] += 1
    return x.reshape(-1, n, n)
