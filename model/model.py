# import necessary libraries
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from einops import rearrange
from gem_cnn.nn import GemResNetBlock
from gem_cnn.transform import ScaleMask, GemPrecomp
from gem_cnn.transform.scale_mask import mask_idx, invert_index
from gem_cnn.utils.rep_act import rep_act
from torch.nn import *
from torch_geometric.nn import MessagePassing


class DownConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, (kernel_size, 1 ,1), bias=False, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, (kernel_size, 1 ,1), bias=False, **kwargs),
            nn.Conv3d(out_channels, out_channels, (kernel_size, 1 ,1), (2, 1, 1), bias=False, **kwargs),
            nn.BatchNorm3d(out_channels)
        )
        self.downsample = nn.Conv3d(in_channels, out_channels, (kernel_size, 1 ,1), (2, 1, 1), bias=False, **kwargs)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.sequential(x)

        if x.shape[3] == 1 and x.shape[4] == 1:
            out = out[:, :, :, 3:4, 3:4] + identity[:, :, :, 1:2, 1:2]
        else:
            out += identity
        return F.leaky_relu(out, 0.2)


class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = config["encoder"]

        self.dim = config["dim"]
        self.dim_out = self.dim[-1] * config["l"] // (2 ** len(self.dim))
        self.l_in = config["l"]
        self.n_channels = config["n_channels"]

        # model layers
        self.encoder = self.make_encoder()

    def make_encoder(self):
        layers = [DownConv1D(self.n_channels, self.dim[0], 3, padding=1)]
        for i in range(len(self.dim) - 1):
            layers.append(DownConv1D(self.dim[i], self.dim[i+1], 3, padding=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x[:, None, :, None, None])
        return rearrange(x, "b c l x y -> (b x y) (c l)")


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, do_identity=True, **kwargs):
        super().__init__()
        self.do_identity = do_identity

        self.sequential = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm3d(out_channels)
        )
        if in_channels != out_channels:
            self.identity = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        out = self.sequential(x)
        if self.do_identity:
            out += identity[:, :, 2:-2, 2:-2, 2:-2]
        return F.leaky_relu(out, 0.2)


class SpatialAttention(nn.Module):
    """SE-style channel attention over the full spatial volume.

    Squeezes (B, C, D, H, W) into a per-channel descriptor using both mean and
    max pooling, runs it through a shared bottleneck, and gates the original
    feature map with the resulting weights. This lets every spatial location
    see a summary of the entire volume, counteracting the limited receptive
    field of the local convolutions.
    """
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        mean_pool = x.mean(dim=(2, 3, 4))       # (B, C)
        max_pool  = x.amax(dim=(2, 3, 4))       # (B, C)
        pooled    = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2C)
        attn      = self.fc(pooled)[:, :, None, None, None]  # (B, C, 1, 1, 1)
        return x * attn


class FanCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        # initialize
        self.dim = config["dim"]
        self.n_channels = config["n_channels"] if "n_channels" in config else 1
        self.n_classes = config["n_classes"]

        self.cnn = self.make()
        self.conv_final = nn.Conv3d(self.dim[-1], self.n_classes, 1, bias=False)

        if "load" in config:
            ckpt = torch.load(config["load"])
            state_dict = ckpt["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict)

    def make(self):
        layers = [DownConv1D(self.n_channels, self.dim[0], 3, padding=(1, 0, 0))]
        for i in range(len(self.dim) - 1):
            layers.append(DoubleConv3D(self.dim[i], self.dim[i+1], 3))
        return nn.Sequential(*layers)

    def forward(self, inp):
        x = self.cnn(inp)

        x = torch.mean(x, dim=2, keepdim=True)
        return self.conv_final(x)


class GEMUNet(nn.Module):
    def __init__(self, config, dim_in):
        super().__init__()
        self.max_order = config["max_order"]
        self.n_classes = config["n_classes"]
        self.conv_dict = {
            "batch_norm": True,
            "checkpoint": True,
            "num_samples": 7,
            "n_rings": config["n_rings"]
        }

        # pre-compute each forward pass
        self.scale_transforms = [T.Compose([ScaleMask(i),
                                            GemPrecomp(n_rings=config["n_rings"], max_order=config["max_order"])])
                                 for i in range(3)]
        dim = config["dim"]

        # Encoder
        self.conv01 = GemResNetBlock(dim_in, dim[0], 0, self.max_order, **self.conv_dict)
        self.conv02 = GemResNetBlock(dim[0], dim[0], self.max_order, self.max_order, **self.conv_dict)

        # Downstream
        self.pool1 = ParallelTransportPool(1, unpool=False)
        self.conv11 = GemResNetBlock(dim[0], dim[1], self.max_order, self.max_order, **self.conv_dict)
        self.conv12 = GemResNetBlock(dim[1], dim[1], self.max_order, self.max_order, **self.conv_dict)

        self.pool2 = ParallelTransportPool(2, unpool=False)
        self.conv21 = GemResNetBlock(dim[1], dim[2], self.max_order, self.max_order, **self.conv_dict)
        self.conv22 = GemResNetBlock(dim[2], dim[2], self.max_order, self.max_order, **self.conv_dict)

        # Up-stream
        self.unpool2 = ParallelTransportPool(2, unpool=True)
        self.conv13 = GemResNetBlock(dim[1] + dim[2], dim[1], self.max_order, self.max_order, **self.conv_dict)
        self.conv14 = GemResNetBlock(dim[1], dim[1], self.max_order, self.max_order, **self.conv_dict)

        # Decoder
        self.unpool1 = ParallelTransportPool(1, unpool=True)
        self.conv03 = GemResNetBlock(dim[0] + dim[1], dim[0], self.max_order, self.max_order, **self.conv_dict)
        self.conv04 = GemResNetBlock(dim[0], dim[0], self.max_order, self.max_order, **self.conv_dict)
        self.conv05 = GemResNetBlock(dim[0], self.n_classes, self.max_order, 0, last_layer=True, **self.conv_dict)

    def forward(self, x, data):
        # scale graphs
        scale_data = [s(data) for s in self.scale_transforms]
        scale_attr = [(d.edge_index, d.precomp, d.connection) for d in scale_data]

        # encoder
        x = self.conv01(x, *scale_attr[0])
        x = self.conv02(x, *scale_attr[0])

        # downstream
        copy0 = x.clone()
        x = self.pool1(x, data)
        x = self.conv11(x, *scale_attr[1])
        x = self.conv12(x, *scale_attr[1])

        copy1 = x.clone()
        x = self.pool2(x, data)
        x = self.conv21(x, *scale_attr[2])
        x = self.conv22(x, *scale_attr[2])

        # upstream
        x = self.unpool2(x, data)
        x = torch.cat((x, copy1), dim=1)  # "copy/cat"
        x = self.conv13(x, *scale_attr[1])
        x = self.conv14(x, *scale_attr[1])

        x = self.unpool1(x, data)
        x = torch.cat((x, copy0), dim=1)  # "copy/cat"
        x = self.conv03(x, *scale_attr[0])
        x = self.conv04(x, *scale_attr[0])

        # decoder
        x = self.conv05(x, *scale_attr[0])

        return x


class FullModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config["encoder"]["n_channels"]
        self.n_classes = config["probe"]["n_classes"]
        self.probe_name = config["probe"]["name"]

        # set encoder
        self.encoder = self.str_to_attr(config["decoder"]["name"])(config)
        config["dim"] = self.encoder.dim_out

        # set probe
        self.probe = self.str_to_attr(self.probe_name)(config["probe"], config["dim"])

    def forward(self, batch):
        x = batch.x
        x = self.encoder(x)
        x = self.probe(x[:, :, None], batch)

        return x[:, :, 0]


    @staticmethod
    def str_to_attr(attr_name):
        return getattr(sys.modules[__name__], attr_name)


"""Adapted from the gem_cnn library, this module contains a corrected Parallel Transport pool class."""

class ParallelTransportPool(MessagePassing):
    def __init__(self, coarse_lvl, *, unpool):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)
        self.coarse_lvl = coarse_lvl
        self.unpool = unpool

    def forward(self, x, data):
        pool_edge_mask = mask_idx(2 * self.coarse_lvl, data.edge_mask)
        node_idx_fine = torch.nonzero(data.node_mask >= self.coarse_lvl - 1).view(-1)
        node_idx_coarse = torch.nonzero(data.node_mask >= self.coarse_lvl).view(-1)
        node_idx_all_to_fine = invert_index(node_idx_fine, data.num_nodes)
        node_idx_all_to_coarse = invert_index(node_idx_coarse, data.num_nodes)

        coarse, fine = data.edge_index[:, pool_edge_mask]
        coarse_idx_coarse = node_idx_all_to_coarse[coarse]
        fine_idx_fine = node_idx_all_to_fine[fine]

        num_fine, num_coarse = node_idx_fine.shape[0], node_idx_coarse.shape[0]

        if self.unpool:
            connection = -data.connection[pool_edge_mask]  # Parallel transport inverse
            edge_index = torch.stack([fine_idx_fine, coarse_idx_coarse])  # Coarse to fine
            size = (num_fine, num_coarse)
        else:  # Pool
            connection = data.connection[pool_edge_mask]  # Parallel transport
            edge_index = torch.stack([coarse_idx_coarse, fine_idx_fine])  # Fine to coarse
            size = (num_coarse, num_fine)

        out = self.propagate(edge_index=edge_index, x=x, connection=connection, size=size)
        return out

    def message(self, x_j, connection):
        x_j_transported = rep_act(x_j, connection)
        return x_j_transported
