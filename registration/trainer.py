# import necessary libraries
import json
import lightning.pytorch as pl
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pyvista as pv
import sys
import trimesh

from model.losses import BCEDiceLoss, NCCLoss, NMILoss, SinkhornOTLoss
from model.model import *
from model.trainer import SequentialScheduler
from os.path import join
from topolosses.losses import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from utils import numpy_to_sitk
from .model import RegNet

class LightningReg(pl.LightningModule):
    def __init__(self, data, config):
        super().__init__()
        self.c_d, self.c_m, self.c_o = config["DATA"], config["MODEL"], config["OPTIMIZATION"]

        self.losses = {
            k: self.str_to_attr(v)(**self.c_m["loss_kwargs"].get(k, {})) if v is not None else None
            for k, v in self.c_m["loss"].items()}
        self.model = self.str_to_attr(self.c_m["name"]["registration_module"])(data, config)

        # set classifiers
        self.classifiers = nn.ModuleList()
        for i, model in enumerate(self.c_m["ckpts_MPR"]):
            state_dict = torch.load(join(model, "model", "last.ckpt"), map_location=f"cuda:{self.c_o['gpu']}", weights_only=False)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}

            classifier = self.str_to_attr(self.c_m["name"]["classifier"])(self.c_m)
            classifier.load_state_dict(state_dict)
            self.classifiers.append(classifier)

        # classification channel mode
        self.cls_bif_only = self.c_m.get("cls_bif_only", False)

        # logging frequencies
        self.metrics_log_freq = self.c_o.get("metrics_log_freq", 10)
        self.image_log_freq = self.c_o.get("image_log_freq", 50)
        self.theta_init_steps = self.c_o.get("theta_init_steps", 72)

        self.has_been_called = False

        # for plotting consistency
        self.min_ = None
        self.max_ = None

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        cpu_tensors = {
            modal: {k: batch[modal].pop(k) for k in ("image", "lumen")}
            for modal in ("IVUS", "MPR")
        }
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        for modal, tensors in cpu_tensors.items():
            batch[modal].update(tensors)
        return batch

    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _calc_parameter_gradient(param):
        grad = torch.abs(param[1:] - param[:-1])
        return (torch.mean(grad ** 2) + torch.max(grad ** 2)) / 2

    def warmup_lambda(self, iter_):
        return iter_ / self.c_o["n_warmup"] if iter_ < self.c_o["n_warmup"] else 1

    @staticmethod
    def str_to_attr(attr_name):
        return getattr(sys.modules[__name__], attr_name)

    def configure_optimizers(self):
        optimizer = self.str_to_attr(self.c_o["name"])(self.model.parameters(), **self.c_o["optimizer"])

        # freeze classifier parameters
        for param in self.classifiers.parameters():
            param.requires_grad = False

        if self.c_o["n_warmup"] == 0:
            scheduler = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
        else:
            scheduler1 = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
            scheduler2 = self.str_to_attr(self.c_o["lr_policy"])(optimizer, **self.c_o["scheduler"])
            scheduler = SequentialScheduler(optimizer, scheduler1, scheduler2, self.c_o["n_warmup"])
        return [optimizer], [scheduler]

    def init_dz(self, batch):
        """Initialize z-displacement and scaling parameters."""
        self.log_metrics(batch, "pre_init_dz")

        images_dir = join(self.logger.log_dir, "images")
        self._ensure_dir(images_dir)
        pred = batch["CCTA"]["output_polar"]
        radii = pred[0, 2, 0] * batch["CCTA"]["image_polar"].shape[2]
        calc = torch.sigmoid(pred[0, 0, 0])
        bif = torch.sigmoid(pred[0, 1, 0])
        self.to_stl(
            self.model.get_current_frame()[0],
            radii.T,
            join(images_dir, "mesh_pre_init_dz"),
            calc_map=calc.T,
            bif_map=bif.T,
        )

        # set search range for sz and dz
        len_mpr, len_ivus = self.model.dim_mpr[-1], self.model.dim_ivus[-1]
        sz_values = [-len_mpr / len_ivus, len_mpr / len_ivus]

        # define search arrays for each sz value
        dz_step, bound = 0.02, 1.1
        loss_arrays = []
        dz_arrays = []

        true = torch.max(batch["IVUS"]["output_polar"], dim=3, keepdim=True)[0]

        for sz in sz_values:
            dz_min = 1.0 - bound * abs(sz)
            dz_max = bound * abs(sz) - 1.0
            num_steps = max(int((dz_max - dz_min) / dz_step), 50)

            # create search array
            dz_array = torch.linspace(dz_min, dz_max, num_steps, device=self.device)
            dz_arrays.append(dz_array)

            # search for this sz value
            self.model.sz.data.fill_(sz)
            loss_array = torch.zeros([len(dz_array), 3], device=self.device)

            for i, dz in enumerate(dz_array):
                self.model.dz.data.fill_(dz)
                batch["CCTA"]["image_polar"] = self.model(batch["CCTA"]["image"], mode="polar")
                batch["CCTA"]["output_polar"] = self.classify(batch["CCTA"]["image_polar"])

                # classification-based
                pred = torch.max(batch["CCTA"]["output_polar"], dim=3, keepdim=True)[0]

                # mask
                z = self.model.return_z_grid()
                mask = (z <= -1) & (z >= 1)
                pred[:, :2, :, :, mask] = -100

                if self.cls_bif_only:
                    l_cls_a = self.losses['cls_dz_a'](pred[:, 1], true[:, 1]).detach()
                else:
                    l_cls_a = (self.losses['cls_dz_a'](pred[:, 0], true[:, 0]).detach() +
                               self.losses['cls_dz_a'](pred[:, 1], true[:, 1]).detach()) / 2
                l_dice = l_cls_a
                if self.losses.get('cls_dz_b') is not None:
                    if self.cls_bif_only:
                        l_cls_b = self.losses['cls_dz_b'](pred[:, 1], true[:, 1]).detach()
                    else:
                        l_cls_b = (self.losses['cls_dz_b'](pred[:, 0], true[:, 0]).detach() +
                                   self.losses['cls_dz_b'](pred[:, 1], true[:, 1]).detach()) / 2
                    l_dice = (l_cls_a * self.c_o["reg"]["alpha_cls_a"] + l_cls_b * self.c_o["reg"]["alpha_cls_b"]) / 2

                # lumen-based
                pred_l = batch["CCTA"]["output_polar"][:, 2:, 0]
                true_l = batch["IVUS"]["output_polar"][:, 2:, 0]
                pred_l, true_l = self.radial_polygon_area(pred_l), self.radial_polygon_area(true_l)
                l_lum_a = self.losses['lumen_a'](pred_l, true_l).detach()
                l_lum_b = self.losses['lumen_b'](pred_l, true_l).detach()

                loss_array[i] = torch.tensor([l_dice, l_lum_a, l_lum_b])

            loss_arrays.append(loss_array.cpu())

        # find best parameters
        loss_arrays = torch.stack(loss_arrays, dim=0)
        loss_arrays -= loss_arrays.amin(dim=(0, 1), keepdim=True)
        loss_arrays /= loss_arrays.amax(dim=(0, 1), keepdim=True)
        loss_components = loss_arrays.clone()
        loss_arrays = loss_arrays.sum(dim=-1)

        best_losses = [torch.min(loss) for loss in loss_arrays]
        best_sz_idx = torch.argmin(torch.tensor(best_losses))
        best_dz_idx = torch.argmin(loss_arrays[best_sz_idx])

        # set optimal parameters
        self.model.sz.data.fill_(sz_values[best_sz_idx])
        self.model.dz.data.fill_(dz_arrays[best_sz_idx][best_dz_idx])
        self.model.v_flip.fill_(1.0) if sz_values[best_sz_idx] > 0 else self.model.v_flip.fill_(-1.0)


        print(f"Optimal sz: {sz_values[best_sz_idx]:.2f},"
              f" dz: {dz_arrays[best_sz_idx][best_dz_idx]:.2f},"
              f" v_flip: {self.model.v_flip.item():.0f}")

        # plot results
        vectors_dir = join(self.logger.log_dir, "vectors")
        self._ensure_dir(vectors_dir)

        plt.figure(figsize=(10, 5))
        for i, (sz, dz_array, loss) in enumerate(zip(sz_values, dz_arrays, loss_arrays)):
            plt.plot(dz_array.cpu().numpy(), loss.numpy(), label=f"sz = {sz:.2f}")
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('dz')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(join(vectors_dir, "loss_dz.png"))
        plt.close()

        component_labels = ["Normalized dice loss", "Normalized absolute area error", "Normalized NCC value"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for c, (ax, label) in enumerate(zip(axes, component_labels)):
            for sz, dz_array, comp in zip(sz_values, dz_arrays, loss_components):
                ax.plot(dz_array.cpu().numpy(), comp[:, c].numpy(), label=f"sz = {sz:.2f}")
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('dz')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(join(vectors_dir, "loss_components_dz.png"))
        plt.close(fig)

    def init_theta(self, batch, n_theta=72):
        """Initialize rotation parameters using cumulative rotation model."""
        self.log_metrics(batch, "pre_init_theta")

        images_dir = join(self.logger.log_dir, "images")
        self._ensure_dir(images_dir)
        pred = batch["CCTA"]["output_polar"]
        radii = pred[0, 2, 0] * batch["CCTA"]["image_polar"].shape[2]
        calc = torch.sigmoid(pred[0, 0, 0])
        bif = torch.sigmoid(pred[0, 1, 0])
        self.to_stl(
            self.model.get_current_frame()[0],
            radii.T,
            join(images_dir, "mesh_pre_init_theta"),
            calc_map=calc.T,
            bif_map=bif.T,
        )

        # grid search for optimal theta and flip values
        theta_range = 1.0 / self.c_m["multiplier"]
        theta_vals = torch.linspace(-theta_range, theta_range, n_theta, device=self.device)
        n_losses = 2 if self.losses.get('cls_theta_b') is not None else 1
        loss_array = torch.zeros([n_theta, n_losses], device="cpu")

        for i in range(n_theta):
            theta_val = theta_vals[i].item()
            self.model.set_constant_rotation(theta_val)

            # compute loss for current parameters
            batch["CCTA"]["image_polar"] = self.model(batch["CCTA"]["image"], mode="polar")
            batch["CCTA"]["output_polar"] = self.classify(batch["CCTA"]["image_polar"])

            # classification-based
            pred = batch["CCTA"]["output_polar"]
            true = batch["IVUS"]["output_polar"]
            if self.cls_bif_only:
                l_cls_a = self.losses['cls_theta_a'](pred[:, 1], true[:, 1]).detach()
            else:
                l_cls_a = (self.losses['cls_theta_a'](pred[:, 0], true[:, 0]).detach() +
                           self.losses['cls_theta_a'](pred[:, 1], true[:, 1]).detach()) / 2
            loss_array[i, 0] = l_cls_a
            if self.losses.get('cls_theta_b') is not None:
                if self.cls_bif_only:
                    l_cls_b = self.losses['cls_theta_b'](pred[:, 1], true[:, 1]).detach()
                else:
                    l_cls_b = (self.losses['cls_theta_b'](pred[:, 0], true[:, 0]).detach() +
                               self.losses['cls_theta_b'](pred[:, 1], true[:, 1]).detach()) / 2
                loss_array[i, 1] = l_cls_b

        # normalize each loss component to [0, 1] and sum
        loss_array -= loss_array.amin(dim=0, keepdim=True)
        loss_array /= loss_array.amax(dim=0, keepdim=True)
        loss = loss_array.sum(dim=-1)

        # Find best parameters
        index = torch.argmin(loss)
        theta_opt = theta_vals[index].item()

        # Plot loss curves
        plt.figure(figsize=(10, 5))
        eff_angles = (theta_vals * self.c_m["multiplier"] * 180).cpu().numpy()
        plt.plot(eff_angles, loss.numpy())

        plt.axvline(x=theta_opt * self.c_m["multiplier"] * 180, color='r', linestyle='--')
        plt.xlabel('Effective Rotation (degrees)')
        plt.ylabel('Loss')
        plt.title(f"Theta Initialization (cumulative model, factor={self.c_m['multiplier']})")
        plt.savefig(f"{self.logger.log_dir}/vectors/loss_theta.png")

        # Set optimal parameters using cumulative formulation
        self.model.set_constant_rotation(theta_opt)
        eff_angle = theta_opt * self.c_m["multiplier"]* 180
        print(f"Theta: {theta_opt:.4f} (effective: {eff_angle:.1f}, cumulative model)")

    def training_step(self, batch, batch_idx):
        # global initialization
        if self.current_epoch == 0:
            self.init_dz(batch)
            self.init_theta(batch, n_theta=72)

        batch["CCTA"]["image_polar"] = self.model(batch["CCTA"]["image"], mode="polar")
        batch["CCTA"]["output_polar"] = self.classify(batch["CCTA"]["image_polar"])

        # calculate losses
        pred = batch["CCTA"]["output_polar"]
        true = batch["IVUS"]["output_polar"]
        if self.cls_bif_only:
            loss_cls_a = self.losses['cls_refine_a'](pred[:, 1], true[:, 1])
        else:
            loss_cls_a = (self.losses['cls_refine_a'](pred[:, 0], true[:, 0]) +
                          self.losses['cls_refine_a'](pred[:, 1], true[:, 1])) / 2
        loss_cls_b = None
        if self.losses.get('cls_refine_b') is not None:
            if self.cls_bif_only:
                loss_cls_b = self.losses['cls_refine_b'](pred[:, 1], true[:, 1])
            else:
                loss_cls_b = (self.losses['cls_refine_b'](pred[:, 0], true[:, 0]) +
                              self.losses['cls_refine_b'](pred[:, 1], true[:, 1])) / 2
        loss_sim = None
        if self.losses['similarity'] is not None:
            loss_sim = self.losses['similarity'](batch["CCTA"]["image_polar"]
                                                 [..., self.c_d['padding']:-self.c_d['padding'], :],
                                                 batch["IVUS"]["image_polar"],
                                                 batch["IVUS"]["mask_polar"])
        loss_lum = None
        if self.losses['lum_refine'] is not None:
            loss_lum = self.losses['lum_refine'](pred[:, 2], true[:, 2])

        loss_reg = self.regularize()
        loss = self.log_losses(loss_cls_a, loss_cls_b, loss_sim, loss_lum, loss_reg)

        if self.current_epoch == 0 or (self.current_epoch + 1) % 10 == 0:
            self.log_metrics(batch)
            self.log_params()
            self.log_maps(pred, true)

        if self.current_epoch == 0 or (self.current_epoch + 1) % 50 == 0:
            self.log_images(batch)

        return loss

    def plot_centerlines(self, batch, true, pred, name=None):
        """Plot centerlines and frame vectors."""
        # Create output directory
        vectors_dir = join(self.logger.log_dir, "vectors")
        self._ensure_dir(vectors_dir)

        # Extract data
        ctl_true, u_true, v_true = true
        ctl_pred, u_pred, v_pred = pred

        # Filter out empty points
        mask = ctl_pred[:, 0] != 0
        ctl_pred, u_pred, v_pred = ctl_pred[mask], u_pred[mask], v_pred[mask]

        # Convert to numpy
        ctl_true = ctl_true.cpu().detach().numpy()
        u_true = u_true.cpu().detach().numpy()
        v_true = v_true.cpu().detach().numpy()
        ctl_pred = ctl_pred.cpu().detach().numpy()
        u_pred = u_pred.cpu().detach().numpy()
        v_pred = v_pred.cpu().detach().numpy()

        # Generate filename
        if name:
            filename = f"vectors_{name}"
        else:
            filename = f"vectors_{str(self.current_epoch).zfill(3)}"

        # Save vector data
        if "output_polar" in batch["CCTA"]:
            np.savez(
                join(vectors_dir, f"{filename}.npz"),
                ctl_true=ctl_true, u_true=u_true, v_true=v_true,
                ctl_pred=ctl_pred, u_pred=u_pred, v_pred=v_pred,
                ccta_pred=batch["CCTA"]["output_polar"].cpu().detach().numpy(),
                ivus_pred=batch["IVUS"]["output_polar"].cpu().detach().numpy()
            )
        else:
            np.savez(
                join(vectors_dir, f"{filename}.npz"),
                ctl_true=ctl_true, u_true=u_true, v_true=v_true,
                ctl_pred=ctl_pred, u_pred=u_pred, v_pred=v_pred
            )

        # create plot
        fig = plt.figure(figsize=(10, 5))
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        fig.suptitle("Centerlines and vector planes")

        # plot true vectors
        ax0.scatter(*ctl_true[0], color='g', s=10)
        ax0.scatter(*ctl_true[-1], color='k', s=10)
        ax0.quiver(*ctl_true.T, *u_true.T, color='b', length=2)
        ax0.quiver(*ctl_true.T, *v_true.T, color='r', length=2)
        ax0.set_title("True vector planes")

        # plot predicted vectors
        ax1.scatter(*ctl_pred[0], color='g', s=10)
        ax1.scatter(*ctl_pred[-1], color='k', s=10)
        ax1.quiver(*ctl_pred.T, *u_pred.T, color='b', length=2)
        ax1.quiver(*ctl_pred.T, *v_pred.T, color='r', length=2)
        ax1.set_title("Predicted vector planes")

        # set plot limits
        if self.min_ is None or self.max_ is None:
            x_min = min(ctl_true[:, 0].min(), ctl_pred[:, 0].min())
            x_max = max(ctl_true[:, 0].max(), ctl_pred[:, 0].max())
            y_min = min(ctl_true[:, 1].min(), ctl_pred[:, 1].min())
            y_max = max(ctl_true[:, 1].max(), ctl_pred[:, 1].max())
            z_min = min(ctl_true[:, 2].min(), ctl_pred[:, 2].min())
            z_max = max(ctl_true[:, 2].max(), ctl_pred[:, 2].max())
            self.min_ = (x_min, y_min, z_min)
            self.max_ = (x_max, y_max, z_max)

        # apply limits to both axes
        for ax in [ax0, ax1]:
            ax.set_xlim(self.min_[0], self.max_[0])
            ax.set_ylim(self.min_[1], self.max_[1])
            ax.set_zlim(self.min_[2], self.max_[2])

        # save figure
        plt.tight_layout()
        plt.savefig(join(vectors_dir, f"{filename}.png"))
        plt.close()

    def log_losses(self, loss_cls_a, loss_cls_b, loss_sim, loss_lum, loss_reg):
        """Log losses and compute weighted total loss."""
        self.log("loss/cls_a", loss_cls_a, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        weight_a = self.c_o["reg"]["alpha_cls_a"]
        if loss_cls_b is not None:
            self.log("loss/cls_b", loss_cls_b, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            weight_b = self.c_o["reg"]["alpha_cls_b"]
            loss_cls = 0.5 * (loss_cls_a * weight_a) + 0.5 * (loss_cls_b * weight_b)
        else:
            loss_cls = loss_cls_a * weight_a

        self.log("loss/cls", loss_cls, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        if loss_sim is not None:
            self.log("loss/sim", loss_sim.mean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)
        if loss_lum is not None:
            self.log("loss/lum", loss_lum.mean(), on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # unpack regularization
        reg_theta, reg_u, reg_v, norm_u, norm_v, reg_s = loss_reg
        self.log("loss/grad_theta", reg_theta, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log("loss/grad_x", reg_u, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log("loss/grad_y", reg_v, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log("loss/norm_x", norm_u, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log("loss/norm_y", norm_v, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log("loss/grad_s", reg_s, on_epoch=True, on_step=False, prog_bar=False, logger=True)

        reg = (reg_theta * self.c_o["reg"]["grad_theta"] +
               (reg_u + reg_v) / 2 * self.c_o["reg"]["grad_r"] +
               (norm_u + norm_v) / 2 * self.c_o["reg"]["norm_r"] +
               reg_s * self.c_o["reg"]["grad_s"])

        total_loss = loss_cls + reg
        if loss_lum is not None:
            total_loss = total_loss + self.c_o["reg"]["alpha_lum"] * loss_lum.mean()
        if loss_sim is not None:
            total_loss = total_loss + self.c_o["reg"]["alpha_sim"] * loss_sim.mean()

        self.log("loss/total", total_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return total_loss

    def classify(self, x):
        """Apply classifier to input image."""

        # prepare image and apply padding
        pad = self.c_d["padding"]
        x = F.pad(x, (pad, pad, 0, 0, 0, 0), mode="replicate")

        # run classification
        for m in range(len(self.classifiers)):
            self.classifiers[m].eval()
            if m == 0:
                output = self.classifiers[m](x)
            else:
                output += self.classifiers[m](x)

        # average predictions
        output = output / (m + 1)
        output[:, 3] = torch.sigmoid(output[:, 3])

        return output[:, 1:]

    def log_metrics(self, batch, name=None):
        """Log evaluation metrics."""
        metrics_dir = join(self.logger.log_dir, "metrics")
        self._ensure_dir(join(metrics_dir))

        # get reference frame from original centerline
        ctl_true = batch["CTL"]["R"]["ctl"][0]
        t_true, u_true, v_true = self.model.get_coordinate_frame_from_tu(batch["CTL"]["R"]["tu"][0], return_tangents=True)

        # get current frame from model
        ctl_pred, u_pred, v_pred = self.model.get_current_frame()

        # plot centerlines and vector planes
        self.plot_centerlines(batch, (ctl_true, u_true, v_true), (ctl_pred, u_pred, v_pred), name)

        # calculate centerline metrics
        R, P, F1 = self.get_ctl_metrics(ctl_true, ctl_pred)

        # calculate lumen dice
        if "output_polar" not in batch["CCTA"]:
            batch["CCTA"]["image_polar"] = self.model(batch["CCTA"]["image"], mode="polar")
            batch["CCTA"]["output_polar"] = self.classify(batch["CCTA"]["image_polar"])
        dice_l = self.radial_dice(batch["IVUS"]["output_polar"][:, 2], batch["CCTA"]["output_polar"][:, 2])

        # calculate vector similarity metrics
        cos_u = F.cosine_similarity(u_true, u_pred, dim=-1).cpu().detach().numpy()
        cos_v = F.cosine_similarity(v_true, v_pred, dim=-1).cpu().detach().numpy()

        # generate filename
        if name:
            filename = f"metrics_{name}"
        else:
            filename = f"metrics_{str(self.current_epoch).zfill(3)}"

        # save metrics to JSON
        metrics = {
            "recall": R.cpu().detach().item(),
            "precision": P.cpu().detach().item(),
            "F1": F1.cpu().detach().item(),
            "cos_u": cos_u.tolist(),
            "cos_v": cos_v.tolist(),
            "dice_lumen": dice_l.cpu().detach().item()
        }

        with open(join(metrics_dir, f"{filename}.json"), "w") as f:
            json.dump(metrics, f)

        # create plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            f"Recall: {R.cpu().detach().item():2f}, "
            f"Precision: {P.cpu().detach().item():2f}, "
            f"F1: {F1.cpu().detach().item():2f}, "
            f"Lumen dice: {dice_l.cpu().detach().item():.2f}"
        )

        # plot cosine similarities
        ax[0].plot(cos_u, 'o')
        ax[0].set_title("cos_u")
        ax[0].set_ylim(-1.1, 1.1)

        ax[1].plot(cos_v, 'o')
        ax[1].set_title("cos_v")
        ax[1].set_ylim(-1.1, 1.1)

        plt.tight_layout()
        plt.savefig(join(metrics_dir, f"{filename}.png"))
        plt.close()

    @staticmethod
    def get_ctl_metrics(true, pred, threshold=2):
        """Calculate metrics for centerline registration quality."""
        try:
            # calculate distances between points
            A_expanded = pred[None].unsqueeze(2)
            B_expanded = true[None].unsqueeze(1)

            # compute squared Euclidean distances
            distances = torch.sum(
                (A_expanded - B_expanded) ** 2, dim=3
            )

            # find minimum distances in both directions
            dist_A_to_B = torch.min(distances, dim=2)[0][0]
            dist_B_to_A = torch.min(distances, dim=1)[0][0]

            # calculate metrics
            R = torch.sum(dist_B_to_A < threshold) / max(len(true), 1)  # Recall
            P = torch.sum(dist_A_to_B < threshold) / max(len(pred), 1)  # Precision
            F1 = 2 * R * P / (R + P + 1e-8)  # F1 score

            return R, P, F1
        except Exception as e:
            print(f"Error calculating centerline metrics: {e}")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def log_params(self):
        """Log current parameter values."""
        # Create output directory
        params_dir = join(self.logger.log_dir, "params")
        self._ensure_dir(params_dir)

        # Get parameter values
        du = self.model.du.cpu().detach().numpy()
        dv = self.model.dv.cpu().detach().numpy()
        theta_raw = self.model.theta.cpu().detach().numpy()
        theta_deg = theta_raw * self.c_m["multiplier"] * 180

        theta = (self.model.theta.cpu().detach().numpy() * math.pi + math.pi) % (2 * math.pi)

        # Get s residual (B-spline deformation only)
        s_residual = self.model._get_arclength_residual().cpu().detach().numpy()

        # Create plot
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        theta_offset_deg = self.model.theta_offset.cpu().detach().item() * self.c_m["multiplier"] * 180
        fig.suptitle(
            f"sr: {self.model.sr.cpu().detach().item():.2f}, "
            f"sz: {self.model.sz.cpu().detach().item():.2f}, "
            f"dz: {self.model.dz.cpu().detach().item():.2f}, "
            f"theta_offset: {theta_offset_deg:.1f}"
        )

        # Plot parameters
        ax[0, 0].plot(du, 'o')
        ax[0, 0].set_title("du")
        ax[0, 0].set_ylim(-0.5, 0.5)

        ax[0, 1].plot(dv, 'o')
        ax[0, 1].set_title("dv")
        ax[0, 1].set_ylim(-0.5, 0.5)

        ax[1, 0].plot(theta_deg, 'o')
        ax[1, 0].set_title("theta")
        ax[1, 0].set_ylabel("deg")
        ax[1, 0].set_ylim(-180, 180)
        ax[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Plot s residual (B-spline deformation)
        ax[1, 1].plot(s_residual, 'o')
        ax[1, 1].set_title("s residual (B-spline)")
        ax[1, 1].set_ylabel("residual deformation")
        ax[1, 1].set_xlabel("frame index")
        ax[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(join(params_dir, f"params_{str(self.current_epoch).zfill(3)}.png"))
        plt.close()

    def log_maps(self, pred, true):
        """Log calcification and bifurcation prediction maps over time."""
        maps_dir = join(self.logger.log_dir, "maps")
        self._ensure_dir(maps_dir)

        # Apply sigmoid to pred since it's not sigmoided yet
        pred_sigmoid = torch.sigmoid(pred[:, :2]).cpu().detach().numpy()
        true_np = true[:, :2].cpu().detach().numpy()

        # Extract maps (batch index 0) and squeeze any extra dimensions
        pred_calc = np.squeeze(pred_sigmoid[0, 0])
        pred_bif = np.squeeze(pred_sigmoid[0, 1])
        true_calc = np.squeeze(true_np[0, 0])
        true_bif = np.squeeze(true_np[0, 1])

        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Calcification & Bifurcation Maps - Epoch {self.current_epoch}")

        # Plot calcification maps
        im0 = axes[0, 0].imshow(pred_calc, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, 0].set_title("Predicted Calcification")
        axes[0, 0].set_xlabel("Z (frames)")
        axes[0, 0].set_ylabel("Angle")
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(true_calc, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title("True Calcification")
        axes[0, 1].set_xlabel("Z (frames)")
        axes[0, 1].set_ylabel("Angle")
        plt.colorbar(im1, ax=axes[0, 1])

        # Plot bifurcation maps
        im2 = axes[1, 0].imshow(pred_bif, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title("Predicted Bifurcation")
        axes[1, 0].set_xlabel("Z (frames)")
        axes[1, 0].set_ylabel("Angle")
        plt.colorbar(im2, ax=axes[1, 0])

        im3 = axes[1, 1].imshow(true_bif, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title("True Bifurcation")
        axes[1, 1].set_xlabel("Z (frames)")
        axes[1, 1].set_ylabel("Angle")
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(join(maps_dir, f"maps_{str(self.current_epoch).zfill(3)}.png"), dpi=150)
        plt.close()

        # Also save raw data for later analysis
        np.savez(
            join(maps_dir, f"maps_{str(self.current_epoch).zfill(3)}.npz"),
            pred_calc=pred_calc,
            pred_bif=pred_bif,
            true_calc=true_calc,
            true_bif=true_bif
        )

    def log_images(self, batch):
        """Save transformed images as NIFTI files."""
        # Create output directory
        images_dir = join(self.logger.log_dir, "images")
        self._ensure_dir(images_dir)

        # Get spacing information
        spacing = np.array([float(s) for s in self.model.spacing_mpr.cpu().detach().numpy()])

        # Save reference images in first epoch
        if self.current_epoch == 0:
            # Save IVUS image
            ivus = batch["IVUS"]["image"][0].cpu().detach().numpy()
            ivus = ivus * 255
            numpy_to_sitk(ivus, spacing, np.zeros(3), join(images_dir, "ivus.nii.gz"))

            # Save IVUS segmentation
            lumen = batch["IVUS"]["lumen"][0].cpu().detach().numpy()
            numpy_to_sitk(lumen, spacing, np.zeros(3), join(images_dir, "ivus_lum.nii.gz"))

            # Save MPR image
            mpr = batch["MPR"]["image"][0].cpu().detach().numpy()
            mpr = mpr * 2000 - 760
            numpy_to_sitk(mpr, spacing, np.zeros(3), join(images_dir, "mpr.nii.gz"))

        # Generate filename prefix
        epoch_str = str(self.current_epoch + 1).zfill(3)

        # Save transformed MPR image
        mpr_warped = self.model(batch["CCTA"]["image"], mode="default")
        mpr_warped = mpr_warped[0, 0].cpu().detach().numpy()
        mpr_warped = mpr_warped * 2000 - 760
        numpy_to_sitk(
            mpr_warped,
            spacing,
            np.zeros(3),
            join(images_dir, f"mpr_warped_{epoch_str}.nii.gz")
        )

        # Save MPR segmentation
        lumen_warped = self.model.forward_image(batch["MPR"]["lumen"])
        lumen_warped = lumen_warped[0, 0].cpu().detach().numpy()
        numpy_to_sitk(lumen_warped, spacing, np.zeros(3), join(images_dir, f"mpr_lum_warped_{epoch_str}.nii.gz"))

        # Export artery meshes with calcification/bifurcation maps
        if "output_polar" in batch["CCTA"]:
            # predicted mesh
            pred = batch["CCTA"]["output_polar"]
            radii = pred[0, 2, 0] * batch["CCTA"]["image_polar"].shape[2]
            calc = torch.sigmoid(pred[0, 0, 0])
            bif = torch.sigmoid(pred[0, 1, 0])
            self.to_stl(
                self.model.get_current_frame()[0],
                radii.T,
                join(images_dir, f"mesh_{epoch_str}"),
                calc_map=calc.T,
                bif_map=bif.T,
            )

            # reference mesh - only on first epoch (static)
            if self.current_epoch == 0:
                true = batch["IVUS"]["output_polar"]
                radii_ref = true[0, 2, 0] * batch["IVUS"]["image_polar"].shape[2]
                calc_ref = true[0, 0, 0]
                bif_ref = true[0, 1, 0]
                ctl_ref = batch["CTL"]["R"]["ctl"][0]
                self.to_stl(
                    ctl_ref,
                    radii_ref.T,
                    join(images_dir, "mesh_ref"),
                    calc_map=calc_ref.T,
                    bif_map=bif_ref.T,
                )

    def to_stl(self, centerline, radii, filename, calc_map=None, bif_map=None):
        """
        Export artery mesh as STL, optionally with calcification/bifurcation
        probability maps saved as VTK point data.
        """
        n_theta = radii.shape[1]

        # get tangents and rotated frame vectors
        tangents = self.model.compute_tangents(centerline)
        u_vectors, _ = self.model.compute_continuous_frame(tangents)
        angles = torch.linspace(0, 2 * torch.pi, n_theta, device=self.device)
        rotated_vectors = self.model._rotate_vectors_around_axis(tangents, u_vectors, angles)

        # move everything to CPU for mesh construction
        centerline = centerline.cpu().detach()
        radii = radii.cpu().detach()
        rotated_vectors = rotated_vectors.cpu().detach()

        # compute surface vertices
        centerline_expanded = centerline.unsqueeze(1).expand(-1, n_theta, -1)
        radius_expanded = radii.unsqueeze(2)
        vertices = centerline_expanded + radius_expanded * rotated_vectors * self.model.spacing_polar[0]

        # build triangle faces for the tube
        z = centerline.shape[0]
        z_indices = torch.arange(z - 1)
        theta_indices = torch.arange(n_theta)
        z_grid, theta_grid = torch.meshgrid(z_indices, theta_indices, indexing="ij")

        current = (z_grid * n_theta + theta_grid).flatten()
        next_theta = (z_grid * n_theta + (theta_grid + 1) % n_theta).flatten()
        next_z = ((z_grid + 1) * n_theta + theta_grid).flatten()
        next_both = ((z_grid + 1) * n_theta + (theta_grid + 1) % n_theta).flatten()

        triangles1 = torch.stack([current, next_theta, next_z], dim=1)
        triangles2 = torch.stack([next_theta, next_both, next_z], dim=1)
        faces = torch.cat([triangles1, triangles2], dim=0)

        vertices = vertices.reshape(-1, 3).numpy()
        faces_np = faces.numpy()

        # save plain STL
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces_np, process=False)
        mesh.export(f"{filename}.stl")

        # save VTK with calcification/bifurcation point data
        if calc_map is not None or bif_map is not None:
            faces_pv = np.hstack([np.full((faces_np.shape[0], 1), 3), faces_np])
            mesh_pv = pv.PolyData(vertices, faces_pv)

            if calc_map is not None:
                mesh_pv.point_data["calcification"] = calc_map.reshape(-1).cpu().detach().numpy()
            if bif_map is not None:
                mesh_pv.point_data["bifurcation"] = bif_map.reshape(-1).cpu().detach().numpy()

            mesh_pv.save(f"{filename}.vtk")

    def regularize(self):
        grad_theta = self._calc_parameter_gradient(self.model.theta)
        grad_u = self._calc_parameter_gradient(self.model.du)
        grad_v = self._calc_parameter_gradient(self.model.dv)
        norm_u = torch.max(self.model.du ** 2)
        norm_v = torch.max(self.model.dv ** 2)

        # non-rigid longitudinal smoothness
        s_residual = self.model._get_arclength_residual()
        grad_s = self._calc_parameter_gradient(s_residual)

        return grad_theta, grad_u, grad_v, norm_u, norm_v, grad_s

    @staticmethod
    def radial_polygon_area(radii):
        d_theta = torch.tensor(2 * torch.pi / radii.shape[2], device=radii.device, dtype=radii.dtype)
        r_next = torch.roll(radii, shifts=-1, dims=2)
        return 0.5 * torch.sin(d_theta) * torch.sum(radii * r_next, dim=2)

    def radial_dice(self, reference, prediction):
        intersection_radii = torch.minimum(reference, prediction)
        area_ref = self.radial_polygon_area(reference)
        area_pred = self.radial_polygon_area(prediction)
        area_intersection = self.radial_polygon_area(intersection_radii)

        area_ref = torch.sum(area_ref)
        area_pred = torch.sum(area_pred)
        area_intersection = torch.sum(area_intersection)
        tp_area = area_intersection
        dice = 2 * tp_area / (area_pred + area_ref + 1e-8)
        return dice
