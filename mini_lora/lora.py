import torch
import torch.nn as nn


class LoRAModule(nn.Module):
    def __init__(self, d: int, r: int, k: int, alpha: float = 1.0):
        super().__init__()
        assert d > 0 and r > 0 and k > 0, "d, r and k must be positive"
        self.B = nn.Parameter(torch.zeros(d, r, dtype=torch.float32))
        self.A = nn.Parameter(torch.normal(0, 1, (r, k), dtype=torch.float32))
        self.scaling = alpha / r


class LoRALinear(LoRAModule):
    def __init__(self, module: nn.Linear, r: int, alpha: float = 1.0):
        d, k = module.weight.shape
        super().__init__(d, r, k, alpha)
        self.W0 = module.weight
        self.W0.requires_grad_(False)  # freeze the original weights
        self.bias = module.bias

    def forward(self, x: torch.Tensor):
        W = self.W0 + self.B @ self.A * self.scaling
        return nn.functional.linear(x, W, self.bias)


def _lorafy(model: nn.Module, r: int = 2, alpha: float = 1.0):
    total_params_lorafied = 0
    total_modules_lorafied = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r, alpha))
            total_params_lorafied += module.weight.numel()
            total_modules_lorafied += 1
        else:
            _total_params_lorafied, _total_modules_lorafied = _lorafy(module, r, alpha)
            total_params_lorafied += _total_params_lorafied
            total_modules_lorafied += _total_modules_lorafied

    return total_params_lorafied, total_modules_lorafied


def lorafy(model: nn.Module, r: int = 2, alpha: float = 1.0):
    total_params_lorafied, total_modules_lorafied = _lorafy(model, r, alpha)
    print(
        f"LoRAfied {total_modules_lorafied} modules with a total of {total_params_lorafied} weight parameters"
    )
    return model
