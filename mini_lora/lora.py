import torch
import torch.nn as nn


class LoRAModule(nn.Module):
    def __init__(self, d: int, r: int, k: int, alpha: float = 1.0):
        super().__init__()
        assert d > 0 and r > 0 and k > 0, "d, r and k must be positive"
        self.B = nn.Parameter(torch.zeros(d, r, dtype=torch.float32))
        self.A = nn.Parameter(torch.normal(0, 1, (r, k), dtype=torch.float32))
        self.scaling = alpha / r

class VeRAModule(nn.Module):
    def __init__(self, d: int, r: int, k: int, alpha: float = 1.0):
        super().__init__()
        assert d > 0 and r > 0 and k > 0, "d, r and k must be positive"
        self.A = nn.Parameter(torch.normal(0, 1, (r, k), dtype=torch.float32), requires_grad=False)
        self.va = nn.Parameter(torch.normal(0, 1, (r, 1), dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(d, r, dtype=torch.float32), requires_grad=False)
        self.vb = nn.Parameter(torch.zeros((d, 1), dtype=torch.float32))
        self.scaling = alpha / r


class VeRALinear(VeRAModule):
    def __init__(self, module: nn.Linear, r: int, alpha: float = 1.0):
        d, k = module.weight.shape
        super().__init__(d, r, k, alpha)
        self.W0 = module.weight
        self.bias = module.bias

    def forward(self, x: torch.Tensor):
        W = self.W0 + (self.B * self.vb) @ (self.A * self.va) * self.scaling
        return nn.functional.linear(x, W, self.bias)

class LoRALinear(LoRAModule):
    def __init__(self, module: nn.Linear, r: int, alpha: float = 1.0):
        d, k = module.weight.shape
        super().__init__(d, r, k, alpha)
        self.W0 = module.weight
        self.bias = module.bias

    def forward(self, x: torch.Tensor):
        W = self.W0 + self.B @ self.A * self.scaling
        return nn.functional.linear(x, W, self.bias)


def _lorafy(model: nn.Module, r: int = 2, alpha: float = 1.0, should_lorafy=lambda mod, name: True, should_freeze=lambda x, name: False):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and should_lorafy(module, name):
            setattr(model, name, LoRALinear(module, r, alpha))
        else:
            _lorafy(module, r, alpha)


def lorafy(model: nn.Module, r: int = 2, alpha: float = 1.0, should_apply=lambda x: True, should_freeze=lambda x, name: False):
    for name, module in model.named_parameters():
        if should_freeze(module, name):
            module.requires_grad = False
    _lorafy(model, r, alpha, should_apply)
    return model

def _verafy(model: nn.Module, r: int = 2, alpha: float = 1.0, should_verafy=lambda mod, name: True, should_freeze=lambda x, name: False):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and should_verafy(module, name):
            setattr(model, name, VeRALinear(module, r, alpha))
        else:
            _verafy(module, r, alpha)

def verafy(model: nn.Module, r: int = 2, alpha: float = 1.0, should_apply=lambda x: True, should_freeze=lambda x, name: False):
    for name, module in model.named_parameters():
        if should_freeze(module, name):
            module.requires_grad = False
    _verafy(model, r, alpha, should_apply)
    return model