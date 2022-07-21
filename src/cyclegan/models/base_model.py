import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def set_requires_grad(self, requires_grad: bool = False):
        for param in self.parameters():
            param.requires_grad = False
