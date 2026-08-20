import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (self.num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        pred_log = torch.log_softmax(pred, dim=1)
        return self.criterion(pred_log, smooth_target)


def centralize_gradient(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight.grad is not None:
                grad = module.weight.grad
                if len(grad.shape) > 1:
                    grad.data = grad.data - grad.data.mean(
                        dim=tuple(range(1, len(grad.shape))), keepdim=True
                    )


class PatchWhitening(nn.Module):
    def __init__(self, patch_size=4):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h // self.patch_size, self.patch_size,
                   w // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, -1, c * self.patch_size * self.patch_size)

        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x = (x - mean) / std

        x = x.view(b, h // self.patch_size, w // self.patch_size,
                   c, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, h, w)
        return x
