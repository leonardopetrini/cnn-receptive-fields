# CNN Receptive Fields
Receptive field computation for any CNN in PyTorch.

The *receptive field* of a neuron is defined as the region of the input that can change the output of such a neuron. 

The contribution to the value of neuron $n$ of pixel $(i, j)$ in an input image $x$ is computed by 
- setting all the weights in the network to constant values
- replacing Max-Pooling by Average Pooling
- switching off batch norm
- switching on pixels in the input one at a time and see to which neurons they contribute. 
- contributions are then rescaled to be in [0, 1]

The resulting receptive fields are a set of images $\mathcal R^l = [R_n^l |  n \text{ is a neuron in layer } l]$ that have the same size of the input. 

The value of $R^l_n[i, j] \in [0, 1]$ corresponds to how much pixel (i, j), in an input image - i.e. $x[:, i, j]$ - would contribute to the output of neuron $n$, which lives in layer $l$, if the network had constant weights. 

### Install
```
python -m pip install git+https://github.com/leonardopetrini/cnn-receptive-fields.git
```

### Usage
Examples of how to use the code are reported `examples.ipynb`.

### Dependencies
- torch
- itertools

For the example
- torchvision
- matplotlib

