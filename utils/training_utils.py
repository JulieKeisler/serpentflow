"""
Training utilities for SerpentFlow.

Includes:
- EMA (Exponential Moving Average) wrapper for models
- Gradient scaling and norm utilities
- Training loops for flow-matching and classifiers
"""

import os
import math
import torch
from torch.nn import Module, Parameter, ParameterList
from torch import Tensor
from typing import Iterable, List
from torchmetrics.aggregation import MeanMetric
from torch.nn.parallel import DistributedDataParallel
from flow_matching.path import CondOTProbPath


class EMA(Module):
    """
    Exponential Moving Average wrapper for PyTorch models.

    Keeps a shadow copy of model parameters and swaps them during evaluation.
    """
    def __init__(self, model: Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.register_buffer("num_updates", torch.tensor(0))

        # Shadow copy of parameters
        self.shadow_params: ParameterList = ParameterList([
            Parameter(p.clone().detach(), requires_grad=False)
            for p in model.parameters() if p.requires_grad
        ])
        self.backup_params: List[torch.Tensor] = []

    def train(self, mode: bool) -> None:
        """
        Overrides .train() to swap EMA parameters when switching between train/eval modes.
        """
        if self.training == mode:
            super().train(mode)
            return

        if not mode:
            print("EMA: Switching to eval mode, backing up parameters and copying EMA params")
            self.backup()
            self.copy_to_model()
        else:
            print("EMA: Switching to train mode, restoring original parameters")
            self.restore_to_model()

        super().train(mode)

    def update_ema(self) -> None:
        """Update shadow parameters using EMA formula."""
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        """Copy EMA parameters to the model."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self) -> None:
        """Backup current model parameters for later restoration."""
        assert self.training, "Backup only allowed in train mode."
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) -> None:
        """Restore model parameters from backup."""
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> Tensor:
    """
    Compute total gradient norm of parameters.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return Tensor(0.0)

    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach(), norm_type) for p in parameters
        ]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    """
    Wrapper for mixed-precision training with gradient scaling.
    Handles loss scaling, gradient clipping, and optimizer stepping.
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None,
                 create_graph=False, update_grad=True):
        """
        Scale loss, backward, optionally clip gradients, and step optimizer.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)

        norm = None
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Sample timesteps with a skewed distribution for flow matching.
    """
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    return torch.clip(time, min=0.0001, max=1.0)

def dbg(x, name):
    print(name, x.shape, x.numel()*4/1e9, "GB")
    
def train_one_epoch(model, data_loader, optimizer, lr_schedule, device,
                    epoch, loss_scaler, path, skewed_timesteps=True, accum_iter=1, test_run=False, debug=False, mask=None):
    """
    Training loop for one epoch for flow-matching.
    """
    model.train(True)
    epoch_loss = MeanMetric().to(device)

    for data_iter_step, data in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        samples = data['data'].to(device)
        noise = data.get('noisy', torch.randn_like(samples)).to(device)

        t = skewed_timestep_sample(samples.shape[0], device=device) if skewed_timesteps \
            else torch.rand(samples.shape[0], device=device)

        path_sample = path.sample(t=t, x_0=torch.nan_to_num_(noise), x_1=torch.nan_to_num_(samples))
        x_t, u_t = path_sample.x_t, path_sample.dx_t

        with torch.amp.autocast("cuda"):
            pred_vel = model(x_t, t)
            if mask is None:
                loss = (pred_vel - u_t)
            else:
                loss = (pred_vel - u_t)*mask
            loss = loss.pow(2).mean()
        

        loss /= accum_iter
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=apply_update)

        if apply_update:
            if isinstance(model, EMA):
                model.update_ema()
            elif isinstance(model, DistributedDataParallel) and isinstance(model.module, EMA):
                model.module.update_ema()
            if lr_schedule is not None:
                lr_schedule.step()

        epoch_loss.update(loss.item())

        if data_iter_step % 100 == 0:
            print(f"[epoch {epoch}] step {data_iter_step}/{len(data_loader)}, loss = {loss.item():.4f}")

        if test_run and data_iter_step > 0:
            break

    return float(epoch_loss.compute().detach().cpu())


def train_classifier(model, dataloader, epochs=5, lr=1e-3, device="cpu", mask=None):
    """
    Train a simple binary image classifier.
    """
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in dataloader:
            x, y = x.to(device), y.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            if mask is not None:
                outputs = model(torch.nan_to_num_(x)*mask)
            else:
                outputs = model(torch.nan_to_num_(x))
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            preds = (outputs > 0.5).long()
            correct += (preds.squeeze() == y.squeeze().long()).sum().item()
            total += y.size(0)
        epoch_loss = running_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {acc:.4f}, {(torch.nan_to_num_(x)*mask).min()} | {(torch.nan_to_num_(x)*mask).max()}")
    return model


def classifier_prediction(model, dataloader, device="cpu", mask=None):
    """
    Compute accuracy of a binary classifier.
    
    Handles logits or probability outputs, and works with y of shape (N,) or (N,1).
    
    Args:
        model (torch.nn.Module): binary classifier
        dataloader (torch.utils.data.DataLoader): data loader
        device (str): device to run on ("cpu" or "cuda")
        
    Returns:
        float: classification accuracy in [0, 1]
    """
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if mask is not None:
                outputs = model(torch.nan_to_num_(x)*mask)
            else:
                outputs = model(torch.nan_to_num_(x))
            
            # Ensure outputs are probabilities
            if outputs.dtype.is_floating_point and outputs.max() > 1.0:
                outputs = torch.sigmoid(outputs)
            
            preds = (outputs > 0.5).long().view(-1)
            y_true = y.view(-1).long()
            
            correct += (preds == y_true).sum().item()
            total += y_true.size(0)
    
    return correct / total

