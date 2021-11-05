'''
    Compute receptive field of neural network neurons.
'''

import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

def receptive_field(f, layers, n=32, ch=3):
    """
    Compute the receptive field of `f`, usually the network function up to a certain layer.
    It does so by setting all the weights to a constant, replacing max-pooling by avg-pooling and deactivating batch-norm.

    :param torch.nn.Module f: network layer function
    :param list of str layers: compute the receptive field of layers in the list
    :param int n: input size is n x n x ch
    :param int ch: input channels
    :return dict of torch.Tensor: dict of receptive fields (as an n x n image) for each layers in list
                                  for each neuron in the activation map. Values of `receptive_field` are in [0, 1].
        Values shape: [activation_map_size, activation_map_size, n, n]

    """

    constant_weights(f)
    replace_pooling(f)
    x = one_pixel_inputs(n=n)

    features = create_feature_extractor(
        f, return_nodes=layers)

    out = features(x[:, None].expand(-1, ch, -1, -1))
    for k in layers:
        out[k] = out[k][:, 0].detach()
        m = out[k].shape[-1]
        out[k] = out[k].reshape(n, n, -1).permute(2, 0, 1)
        out[k] /= out[k].max()
        out[k] = out[k].reshape(m, m, n, n)

    return out


def receptive_field_center(rf):
    """
    :param torch.Tensor rf: receptive field for each neuron
        Shape: [number_neurons, n, n]
    :returns torch.Tensor: receptive fields centers
        Shape: [number_neurons, 2]
    """
    n = rf.shape[-1]
    if len(rf.shape) >= 3:
        rf = rf.reshape(-1, n, n)
    X = torch.meshgrid(torch.arange(n, device=rf.device), torch.arange(n, device=rf.device))
    return torch.stack([torch.nanmean((rf * X[i]).sum(dim=2) / rf.sum(dim=2), dim=1) for i in [1, 0]]).t()


def receptive_field_size(rf):
    """
    :param torch.Tensor rf: receptive field for each neuron
        Shape: [number_neurons, n, n]
    :returns torch.Tensor: receptive fields size (as an estimate of the side if it was a square)
        Shape: [number_neurons]
    """
    return rf.sum(dim=[-2, -1]).sqrt()


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
        
        
def replace_pooling(m):
    """
    Replace MaxPool2d by AvgPool2d in `m`.
    :param torch.nn.Module m: neural network model written in PyTorch.
    """
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if target_attr == torch.nn.MaxPool2d:
            setattr(m, attr_str, torch.nn.AvgPool2d)
            m.count_include_pad = True
            m.divisor_override = None
    for ch in m.children():
        replace_pooling(ch)
        
        
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
