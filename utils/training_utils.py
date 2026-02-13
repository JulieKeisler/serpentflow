"""
Training utilities for SerpentFlow.

Includes:
- EMA (Exponential Moving Average) wrapper for models
- Gradient scaling and norm utilities
- Training loops for flow-matching and classifiers
"""

from dataclasses import dataclass
import os
import math
import torch
from torch.nn import Module, Parameter, ParameterList
from torch import Tensor
from typing import Iterable, List
from torchmetrics.aggregation import MeanMetric
from torch.nn.parallel import DistributedDataParallel
from flow_matching.path import CondOTProbPath, PathSample
import time

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
            #print("EMA: Switching to eval mode, backing up parameters and copying EMA params")
            self.backup()
            self.copy_to_model()
        else:
            #print("EMA: Switching to train mode, restoring original parameters")
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

@dataclass
class PathSample:
    x_t: Tensor
    v_target: Tensor   # pour b_theta
    z_target: Tensor   # pour z_theta


class SDEPath:
    """
    Stochastic interpolant path

    x_t = I(t, x0, x1) + γ(t) z
    dX_t = (b̂ ± ε(t) ŝ) dt + sqrt(2 ε(t)) dW_t
    """

    def __init__(self, gamma_scale: float = 0.5, eps_min: float = 1e-6):
        self.gamma_scale = gamma_scale
        self.eps_min = eps_min
        print(f"======================================================\nInitialized SDEPath with gamma_scale={gamma_scale}, eps_min={eps_min}\n======================================================")

    def gamma_fn(self, t: torch.Tensor) -> torch.Tensor:
        # γ(t) = gamma_scale * sqrt(t (1 - t))
        return self.gamma_scale * torch.sqrt(t * (1 - t) + self.eps_min)

    def gamma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return (
            self.gamma_scale
            * (0.5 * (1 - 2 * t))
            / torch.sqrt(t * (1 - t) + self.eps_min)
        )

    def epsilon(self, t: torch.Tensor) -> torch.Tensor:
        # ε(t) = γ(t)^2
        g = self.gamma_fn(t)
        gg = g ** 2
        return gg

    def sample(self, x_0, x_1, t):
        t_exp = t.view(-1, *([1] * (x_0.ndim - 1)))

        I_t = t_exp * x_1 + (1 - t_exp) * x_0
        dI_t = x_1 - x_0

        z = torch.randn_like(x_0)

        g_t = self.gamma_fn(t_exp)
        dg_t = self.gamma_dot(t_exp)

        x_t = I_t + g_t * z
        v = dI_t + dg_t * z

        return PathSample(
            x_t=x_t,
            v_target=v,
            z_target=z
        )



def eta_loss(eta: Tensor, eta_hat: Tensor, mask=None) -> Tensor:
    loss = (eta_hat - eta).pow(2)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)  # broadcasting si nécessaire
    return loss.mean()

def vel_loss(z: Tensor, z_hat: Tensor, mask=None) -> Tensor:
    loss = (z_hat - z).pow(2)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)  # si mask est 1D
    return loss.mean()


def train_one_epoch(model, data_loader, optimizer, lr_schedule, device,
                    epoch, loss_scaler, path, loss_fn=vel_loss, skewed_timesteps=True, accum_iter=1, test_run=False, debug=False, mask=None):
    """
    Training loop for one epoch for flow-matching.
    """
    model.train(True)
    epoch_loss = MeanMetric().to(device)

    for data_iter_step, data in enumerate(data_loader):
        s_time = time.time()
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        samples = data['data'].to(device)
        noise = data.get('noisy', torch.randn_like(samples)).to(device)
        stats = data.get('stats', None)
        if stats is not None:
            stats = stats.to(device)
        if mask is not None:
            noise = torch.nan_to_num(noise, nan=0.0) * mask
            samples = torch.nan_to_num(samples, nan=0.0) * mask

        t = skewed_timestep_sample(samples.shape[0], device=device) if skewed_timesteps \
            else torch.rand(samples.shape[0], device=device)

        path_sample = path.sample(t=t, x_0=noise, x_1=samples)
        x_t, u_t = path_sample.x_t, path_sample.dx_t

        with torch.amp.autocast("cuda"):
            if mask is not None:
                x_t = torch.cat([x_t, mask.repeat(x_t.shape[0], 1, 1, 1)], dim=1)
                
            pred_vel = model(x_t, t, stats=stats)
            loss = loss_fn(u_t, pred_vel, mask) 

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
            print(f"[epoch {epoch}] step {data_iter_step}/{len(data_loader)}, loss = {loss.item():.4f}, time = {time.time()-s_time:.2f} sec")

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

def train_one_epoch_si_sde(
    vel_model,
    denoise_model,
    data_loader,
    optimizer,
    lr_schedule,
    device,
    epoch,
    path,
    loss_scaler,
    accum_iter=1,
    mask=None,
    skewed_timesteps=True,
    test_run=False,
    alpha=0.1
):
    vel_model.train(True)
    denoise_model.train(True)

    epoch_loss = MeanMetric().to(device)

    for step, data in enumerate(data_loader):

        if step % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        x1 = data["data"].to(device)          # target sample
        x0 = data.get("noisy", torch.randn_like(x1)).to(device)
        stats = data.get("stats", None)
        if stats is not None:
            stats = stats.to(device)

        if mask is not None:
            x0 = torch.nan_to_num(x0) * mask
            x1 = torch.nan_to_num(x1) * mask

        # Sample time
        t = (
            skewed_timestep_sample(x1.shape[0], device=device)
            if skewed_timesteps
            else torch.rand(x1.shape[0], device=device)
        )

        # --- Stochastic interpolant ---
        path_sample = path.sample(x_0=x0, x_1=x1, t=t)
        x_t = path_sample.x_t
        v_target = path_sample.v_target
        z_target = path_sample.z_target

        if mask is not None:
            x_t = torch.cat(
                [x_t, mask.repeat(x_t.shape[0], 1, 1, 1)], dim=1
            )

        with torch.amp.autocast("cuda"):
            v_pred = vel_model(x_t, t, stats=stats)
            z_pred = denoise_model(x_t, t, stats=stats)

            loss_v = vel_loss(v_target, v_pred, mask)
            loss_z = eta_loss(z_target, z_pred, mask)

            loss = loss_v + alpha * loss_z 

        loss_v = loss_v / accum_iter
        loss_z = loss_z / accum_iter
        loss = loss / accum_iter
        apply_update = (step + 1) % accum_iter == 0

        loss_scaler(
            loss,
            optimizer,
            parameters=list(vel_model.parameters()) +
                       list(denoise_model.parameters()),
            update_grad=apply_update
        )

        if apply_update:
            if lr_schedule is not None:
                lr_schedule.step()
            if isinstance(vel_model, EMA):
                vel_model.update_ema()
            if isinstance(denoise_model, EMA):
                denoise_model.update_ema()

        epoch_loss.update(loss.item())

        if step % 100 == 0:
            print(
                f"[epoch {epoch}] step {step}/{len(data_loader)}, "
                f"loss={loss.item():.4f}, "
                f"loss_v={loss_v.item():.4f}, "
                f"alpha x loss_z={(alpha*loss_z).item():.4f}, "
            )

        if test_run:
            break
        if step == 0 and (epoch % 10) == 1:
            print(f"\n🔍 DIAGNOSTIC TRAIN MODE:")
            print(f"  vel_model.training: {vel_model.training}")
            print(f"  denoise_model.training: {denoise_model.training}")
            print(f"  t shape: {t.shape}, values: {t[:3]}")
            print(f"  x_t shape: {x_t.shape}, mean: {x_t.mean():.6f}, std: {x_t.std():.6f}")
            print(f"  v_pred: mean={v_pred.mean():.6f}, std={v_pred.std():.6f}, min={v_pred.min():.6f}, max={v_pred.max():.6f}")
            print(f"  z_pred: mean={z_pred.mean():.6f}, std={z_pred.std():.6f}, min={z_pred.min():.6f}, max={z_pred.max():.6f}")
            print()


    return float(epoch_loss.compute().detach().cpu())
