import logging
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        self.logger.info(f'Nbr of trainable parameters: {total_trainable_params}')

    def __str__(self):
        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {total_trainable_params}'
