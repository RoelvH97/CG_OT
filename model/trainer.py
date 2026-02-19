# import necessary libraries
import lightning.pytorch as pl
import sys

from einops import rearrange
from topolosses.losses import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchmetrics.classification import BinaryAUROC
from .losses import BCEDiceLoss
from .model import *


class SequentialScheduler(LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, switch_epoch):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.switch_epoch = switch_epoch
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.switch_epoch:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.switch_epoch:
            self.scheduler1.step(epoch)
        else:
            self.scheduler2.step(epoch)
        super().step(epoch)


class LightningFan(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.c_d, self.c_m, self.c_o = config["DATA"], config["MODEL"], config["OPTIMIZATION"]
        self.alpha_l = self.c_o["alpha_l"]

        self.loss = self.str_to_attr(self.c_m["loss"])(**self.c_m.get("loss_kwargs", {}))
        self.loss_l = nn.L1Loss()
        self.model = self.str_to_attr(self.c_m["name"])(self.c_m)
        self.metric_a = self.str_to_attr(self.c_m["metric"])().to(f"cuda:{self.c_o['gpu']}")
        self.metric_b = BettiMatchingLoss(use_base_loss=False, sigmoid=True)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            y_pred = self.model(batch[0])
        else:
            y_pred = self.model(batch)

            len_, theta = len(batch), self.c_d['polar_transform']['n_theta']
            x = rearrange(batch.x, "(B Z θ) R -> B R θ Z", B=len_, θ=theta)
            y = rearrange(batch.y, "(B Z θ) R -> B R θ Z", B=len_, θ=theta)
            y_pred = rearrange(y_pred, "(B Z θ) R -> B R 1 θ Z", B=len_, θ=theta)
            batch = (x, y, batch.locs)

        losses = []
        for i, name in enumerate(["GuidewireLoss", "CalciumLoss", "BifurcationLoss"]):
            mid = y_pred.shape[-1] // 2
            a = y_pred[:, i:i+1, 0, :, mid:mid+1] if i == 0 else y_pred[:, i:i+1, 0]
            b = batch[1][:, i:i+1, :, mid:mid+1] if i == 0 else batch[1][:, i:i+1]

            loss = self.loss(a, b)
            losses.append(loss)
            self.log(f"train/{name}", loss)

        if self.c_d["modality"] == "IVUS":
            loss = sum(losses) / len(losses)
        else:
            loss = (losses[1] + losses[2]) / 2
        self.log(f"train/{self.c_m['loss']}", loss)

        # lumen loss
        loss_l = self.loss_l(torch.sigmoid(y_pred[:, 3, 0]), batch[1][:, 3])
        self.log(f"train/LumenLoss", loss_l)

        if batch_idx % 5 == 0 and self.current_epoch % 5 == 0:
            self.log_rays(batch[0], batch[1][:, 3], y_pred[:, 3, 0], batch_idx)

        return (loss + self.alpha_l * loss_l) / 2

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            y_pred = self.model(batch[0])
        else:
            y_pred = self.model(batch)

            len_, theta = len(batch), self.c_d['polar_transform']['n_theta']
            x = rearrange(batch.x, "(B Z θ) R -> B R θ Z", B=len_, θ=theta)
            y = rearrange(batch.y, "(B Z θ) R -> B R θ Z", B=len_, θ=theta)
            y_pred = rearrange(y_pred, "(B Z θ) R -> B R 1 θ Z", B=len_, θ=theta)
            batch = (x, y, batch.locs)

        losses = []
        metrics_a = []
        metrics_b = []
        for i, name in enumerate(["GuidewireLoss", "CalciumLoss", "BifurcationLoss"]):
            a, b = y_pred[:, i:i+1, 0, :, batch[2]], batch[1][:, i:i+1, :, batch[2]]

            loss = self.loss(a[..., 0, :], b[..., 0, :])
            losses.append(loss)
            self.log(f"val/{name}", loss)

            metric_b = self.metric_b(a[..., 0, :], b[..., 0, :])
            metrics_b.append(metric_b)
            self.log(f"val/{name.replace('Loss', 'BettiError')}", metric_b)

            # get metric
            a = torch.sigmoid(a).flatten()
            b = b.flatten()
            metric_a = self.metric_a(torch.sigmoid(a).flatten(), b.flatten() > 0.5)
            metrics_a.append(metric_a)
            self.log(f"val/{name.replace('Loss', 'Metric')}", metric_a)

        if self.c_d["modality"] == "IVUS":
            loss = sum(losses) / len(losses)
            metric_a = sum(metrics_a) / len(metrics_a)
        else:
            loss = (losses[1] + losses[2]) / 2
            metric_a = (metrics_a[1] + metrics_a[2]) / 2
        metric_b = metrics_b[1] + metrics_b[2]
        self.log(f"val/{self.c_m['loss']}", loss)
        self.log(f"val/{self.c_m['metric']}", metric_a)
        self.log(f"val/TotalBettiError", metric_b)

        if batch_idx % 20 == 0:
            self.log_images(y_pred[:, :, 0], batch[1], batch_idx, mode="val")

        # lumen loss
        loss_l = self.loss_l(torch.sigmoid(y_pred[:, 3, 0]), batch[1][:, 3])
        self.log(f"val/LumenLoss", loss_l)
        return (loss + self.alpha_l * loss_l) / 2

    def warmup_lambda(self, iter_):
        return iter_ / self.c_o["n_warmup"] if iter_ < self.c_o["n_warmup"] else 1

    @staticmethod
    def str_to_attr(attr_name):
        return getattr(sys.modules[__name__], attr_name)

    def configure_optimizers(self):
        optimizer = self.str_to_attr(self.c_o["name"])(self.model.parameters(), **self.c_o["optimizer"])

        if self.c_o["n_warmup"] == 0:
            scheduler = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
        else:
            scheduler1 = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
            scheduler2 = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
            scheduler = SequentialScheduler(optimizer, scheduler1, scheduler2, self.c_o["n_warmup"])
        return [optimizer], [scheduler]

    def log_images(self, input, target, idx, mode):
        for i, name in enumerate(["Guidewire", "Calcium", "Bifurcation"]):
            inp = torch.sigmoid(input[:, i].squeeze()).cpu().detach().numpy()
            tar = target[0, i].squeeze().cpu().detach().numpy()

            self.logger.experiment.add_image(f"{mode}_pred/{name}_{idx}", inp[None], self.current_epoch)
            self.logger.experiment.add_image(f"{mode}_ref/{name}_{idx}", tar[None], self.current_epoch)

    def log_rays(self, x, y, y_hat, idx):
        pad = self.c_d['padding']

        for i in range(0, y.shape[-1], 5):
            x_ = x[0, 0, :, pad:-pad, i + pad].cpu().detach()
            y_ = y[0, :, i].cpu().detach()
            y_hat_ = F.sigmoid(y_hat[0, :, i]).cpu().detach()

            x_rgb = x_[None].repeat(3, 1, 1)
            y_ = torch.clip((1 - y_) * x_rgb.shape[1], 0, 63)
            y_hat_ = torch.clip((1 - y_hat_) * x_rgb.shape[1], 0, 63)

            x_rgb[0, y_.to(int), torch.arange(x_rgb.shape[2])] = 1
            x_rgb[1, y_hat_.to(int), torch.arange(x_rgb.shape[2])] = 1
            self.logger.experiment.add_image(f"raycasting/lumen_{idx}_{i}", x_rgb, self.current_epoch)
