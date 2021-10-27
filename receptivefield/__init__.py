'''
    Compute receptive field of neural network neurons.
'''
import torch
import torch.nn as nn

def constant_weights(model):
    '''
    Set all weights to a constant and biases to zero in `model`.
    Deactivate batch norm.
    '''
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
    '''
    Replace MaxPool2d by AvgPool2d in `m`.
    '''
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
    '''
    :param integer n: input image side size (image is n x n)
    Returns a batch of n^2 images such that imgs[i] has the i-th pixel on (=1) and the rest is zero. 
    '''
    x = torch.zeros(n, n, n, n)
    for i in range(n):
        for j in range(n):
            x[i, j, i, j] += 1
    return x.reshape(-1, n, n)

def receptive_field(f, n=32, ch=3):
    '''
    Compute the receptive field of `f`, usually the network function up to a certain layer.
    It does so by setting all the weights to a constant, replacing max-pooling by avg-pooling and deactivating batch norm.
            Parameters:
                    f (torch.module): network function 
                    n (int): input size is n x n x ch
                    ch (int)

            Returns:
                    receptive_field (torch.tensor) [number_of_neurons, n, n]: receptive field (as an n x n image) for each neuron in layer f.
                    Values of receptive_field are in [0, 1].

    '''
    constant_weights(f)
    replace_pooling(f)
    x = one_pixel_inputs(n=n)
    out = f(x[:, None].expand(-1, ch, -1, -1))[:, 0].detach()
    out = out.reshape(n, n, -1).permute(2, 0, 1)
    out /= out.max()
    return out