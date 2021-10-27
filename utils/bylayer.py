'''
    Return a network function up to layer `i` for architectures in "models".
'''

import torch
import torch.nn as nn

def vgg_bylayer(net, i=1, return_names=False):
    ls = []
    lsn = []
    for n, c in list(net.children())[0].named_children():
        ls.append(c)
        lsn.append(n)
    ls.append(nn.Flatten())
    lsn.append('fl')
    ls.append(list(net.children())[1])
    lsn.append('classifier')
    if return_names:
        return lsn
    else:
        return nn.Sequential(*ls[:(i+1)])
    
def resnet_bylayer(net, i=-1, return_names=False):
    ls = []
    lsn = []
    for n, c in net.named_children():
        ls.append(c)
        lsn.append(n)
        if n == 'bn1':
            ls.append(nn.ReLU())
            lsn.append('relu')
        if n == 'layer4':
            ls.append(nn.AvgPool2d(4))
            ls.append(nn.Flatten())
            lsn.append('avgp')
            lsn.append('fl')
        if n == 'linear':
            ls.append(nn.Flatten(0, -1))
            lsn.append('fl')
    ii = [0, 1, 3, 4, 5, 6, 9, 11]
    return nn.Sequential(*ls[:ii[i]]) #, lsn[:ii[i+1]]


### VGG: operation of each layer ###
# cfg = {
# 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
# 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
# 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
# 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# i = 0
# layer_names = {}
# for k in cfg:
#     for bn in ['', 'bn']:
#         kbn = k + bn
#         layer_names[kbn] = []
#         for l in cfg[k]:
#             if l == 'M':
#                 layer_names[kbn].append('M')
#                 i += 1
#             else:
#                 layer_names[kbn].append('Conv')
#                 if bn == 'bn':
#                     layer_names[kbn].append('BN')
#                 layer_names[kbn].append('Relu')
#                 i += 3 if bn == 'bn' else 2
#         layer_names[kbn].append('A')
#         layer_names[kbn].append('fl')
#         layer_names[kbn].append('classifier')