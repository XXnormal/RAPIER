from typing import List, Optional
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU

# This implementation of MADE is copied from: https://github.com/e-hulten/made.

class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, cuda_device: Optional[int] = None) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(n_in, n_out, bias)
        self.mask = None
        self.cuda_device = cuda_device

    def initialise_mask(self, mask: Tensor):
        """Internal method to initialise mask."""
        if self.cuda_device == None:
            self.mask = mask
        else:
            torch.cuda.set_device(self.cuda_device)
            self.mask = mask.cuda()
            
    def set_device(self, device):
        torch.cuda.set_device(device)
        self.cuda_device = device
        self.mask = self.mask.cpu().cuda()
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = None,
        cuda_device: Optional[int] = None,
    ) -> None:
        """Initalise MADE model.
    
        Args:
            n_in: Size of input.
            hidden_dims: List with sizes of the hidden layers.
            gaussian: Whether to use Gaussian MADE. Default: False.
            random_order: Whether to use random order. Default: False.
            seed: Random seed for numpy. Default: None.
        """
        super().__init__()
        # Set random seed.
        np.random.seed(seed)
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        self.layers = []
        self.cuda_device = cuda_device

        if self.cuda_device != None:
            torch.cuda.set_device(self.cuda_device)

        # List of layers sizes.
        dim_list = [self.n_in, *hidden_dims, self.n_out]

        # Make layers and activation functions.
        for i in range(len(dim_list) - 2):
            if self.cuda_device == None:
                self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))
                self.layers.append(ReLU())
            else:
                self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1], 
                    cuda_device=self.cuda_device).cuda())
                self.layers.append(ReLU().cuda())

        # Hidden layer to output layer.
        if self.cuda_device == None:
            self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        else:
            self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1], 
                cuda_device=self.cuda_device).cuda())

        # Create model.
        self.model = nn.Sequential(*self.layers)

        # Get masks for the masked activations.
        self._create_masks()

    def set_device(self, device):
        torch.cuda.set_device(device)
        self.cuda_device = device
        for model in self.model:
            if isinstance(model, MaskedLinear):
                model.set_device(device)
            model = model.cpu().cuda()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.gaussian:
            # If the output is Gaussian, return raw mus and sigmas.
            res = self.model(x)
            return res
        else:
            # If the output is Bernoulli, run it trough sigmoid to squash p into (0,1).
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """Create masks for the hidden layers."""
        # Define some constants for brevity.
        L = len(self.hidden_dims)
        D = self.n_in

        # Whether to use random or natural ordering of the inputs.
        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        # Set the connectivity number m for the hidden layers.
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)

        # Add m for output layer. Output order same as input order.
        self.masks[L + 1] = self.masks[0]

        # Create mask matrix for input -> hidden_1 -> ... -> hidden_L.
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            # Initialise mask matrix.
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                # Use broadcasting to compare m_next[j] to each element in m.
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            # Append to mask matrix list.
            self.mask_matrix.append(M)

        # If the output is Gaussian, double the number of output units (mu,sigma).
        # Pairwise identical masks.
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        # Initalise the MaskedLinear layers with weights.
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))
