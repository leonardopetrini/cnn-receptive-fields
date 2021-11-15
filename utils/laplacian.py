import torch

def laplacian_eigenvectors(N=3):
    """Compute the eigenvectors of the Laplacian on the NxN grid.
    :param int n: grid size
    :returns torch.Tensor: the N^2 eigenvectors of the grid Laplacian. Each eigenvector is a NxN matrix."""
    laplacian = torch.zeros(N**2, N**2)
    for i in range(N**2):
        x = i % N
        y = i // N
        n = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            xx = x + dx
            yy = y + dy
            if 0 <= xx < N and 0 <= yy < N:
                laplacian[i, N * yy + xx] = -1
                n += 1
        laplacian[i, i] = n

    lambd, psi = torch.symeig(laplacian, eigenvectors=True)
    return psi.T.reshape(N**2, N, N)

### Modify network weights ###

def new_weights_same_laplacian_spectrum(f, psi, fs=3):
    """
    [work in progress...]
    :param f: network function
    :param psi: Laplacian eigenvectors
    :param fs: filter size
    :return: random conv. weights with same Laplacian components as f's weights.
    """
    pnew = []

    for p in f.modules():
        if 'conv' in str(type(p)) and p.weight.shape[-1] == fs:
            cout, cin = p.weight.shape[:2]

            w = p.weight.detach()
            c = torch.einsum('iyx,jyx->ij', w.reshape(-1, fs, fs), psi)
            mean = c.mean(dim=[0])
            std = c.std(dim=[0])

            coeffs = torch.randn(cout, cin, fs ** 2).mul(std).add(mean)

            pnew.append(torch.einsum('ijc,cnm->ijnm', coeffs, psi))

    return pnew

def init_with_weights(f, w, fs=3):
    for p in f.modules():
        if 'conv' in str(type(p)) and p.weight.shape[-1] == fs:
            p.weight.data = w.pop(0)