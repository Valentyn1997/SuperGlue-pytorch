import torch
import torch.nn as nn
from torch.nn import init


class MaskedBatchNorm1d(nn.SyncBatchNorm):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.

        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``

        Shape:
            - Input: :math:`(N, C, L)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, input_mask=None):
        # Calculate the masked mean and variance
        B, C, L = input.shape
        if input_mask is not None and (input_mask.shape != (B, 1, L)):
            raise ValueError('Mask should have shape (B, 1, L).')
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        if input_mask is not None:
            masked = input * input_mask
            n = input_mask.sum()
        else:
            masked = input
            n = B * L
        # Sum
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        # Update running stats

        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                if len(self.running_mean.shape) == 1:  # Problems with multi-GPU sharing
                    self.running_mean = self.running_mean.unsqueeze(0).unsqueeze(-1)
                    self.running_var = self.running_var.unsqueeze(0).unsqueeze(-1)

                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var

                # print(self.running_mean.shape)
                self.running_mean = self.running_mean.squeeze()
                self.running_var = self.running_var.squeeze()

            self.num_batches_tracked += 1
        # Norm the inputPY
        if self.track_running_stats and not self.training:
            # print(self.running_mean.shape)
            if len(self.running_mean.shape) == 1:  # Problems with multi-GPU sharing
                self.running_mean = self.running_mean.unsqueeze(0).unsqueeze(-1)
                self.running_var = self.running_var.unsqueeze(0).unsqueeze(-1)
            normed = (masked - self.running_mean) / (torch.sqrt(self.running_var + self.eps))

            self.running_mean = self.running_mean.squeeze()
            self.running_var = self.running_var.squeeze()
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
        return normed