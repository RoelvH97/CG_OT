# import necessary libraries
import h5py
import json
import lightning.pytorch as pl
import numpy as np
import random
import sys
import torch

from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_multiplicative, augment_gamma
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur, augment_gaussian_noise
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from einops import rearrange, repeat
from gem_cnn.transform import SimpleGeometry
from itertools import cycle, islice
from model import FanCNN
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as DataLoaderGeo
from utils import MPRTransform, PolarTransform, sitk_to_numpy


class IVUSDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode

        self.geometric = config["geometric"]
        self.modality = config["modality"]
        self.pad = config["padding"]

        # set transform
        config["polar_transform"]["p_theta"] = self.pad
        self.transform = MPRTransform(config["mpr_transform"], device="cpu")
        self.transform_polar = PolarTransform(config["polar_transform"], device="cpu")

        # data dicts
        self.id_list, self.images = self.setup()
        self.h5_cache = {}

        if self.mode == "train" or self.mode == "train_val":
            self.id_list = list(islice(cycle(self.id_list), self.config["size"]))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        path_image = self.images[id_]
        path_h5 = path_image.replace("images", "h5").replace("_0000.nii.gz", ".h5")

        if self.mode == "train" or self.mode == "train_val":
            return self.load_h5(path_h5, augment=True)
        else:
            return self.load_h5(path_h5)

    def setup(self):
        fold = json.load(open(join(self.config["root_dir"], "splits_final.json")))[self.config["fold"]]

        if self.mode == "train_val":
            id_list = fold["train"] + fold["val"]
        else:
            id_list = fold[self.mode]

        # remove any excluded samples
        for excl in self.config.get("exclude", []):
            if excl in id_list:
                id_list.remove(excl)

        # load images
        images = {}
        for id_ in id_list:
            images[id_] = join(self.config["root_dir"], f"imagesTr", f"{id_}_0000.nii.gz")

        return id_list, images

    def load_h5(self, h5_path, augment=False):
        h5_file = self._get_h5_file(h5_path)
        image = h5_file[self.modality]["image"]
        label = h5_file[self.modality]["label"]
        label_l = h5_file[self.modality]["label2"]

        # pad image and label
        image = np.pad(image, ((0, 0), (0, 0), (self.pad, self.pad)), mode="edge")
        label = np.pad(label, ((0, 0), (0, 0), (self.pad, self.pad)), mode="constant")
        lumen = np.pad(label_l, ((0, 0), (0, 0), (self.pad, self.pad)), mode="constant")

        locs, l = self.get_location(label_l)
        if augment:
            image = image[:, :, l - self.pad:l + self.pad * 3 + 1]
            label = label[:, :, l - self.pad:l + self.pad * 3 + 1]
            lumen = lumen[:, :, l - self.pad:l + self.pad * 3 + 1]
        else:
            image = image[:, :, :]
            label = label[:, :, :]
            lumen = lumen[:, :, :]

        image = self.norm(image)
        if augment:
            image = self.augment_intensities(image[None])[0]
        data = np.stack([image, label == 1, label == 2, label == 3, lumen == 1],
                        axis=0, dtype=np.float32)
        data = torch.from_numpy(data)

        if augment:
            data = self.transform_polar(data, offset_range=(-12, 12))
        else:
            data = self.transform_polar(data)
        data = rearrange(data, "1 C Z R θ -> C R θ Z")

        if augment:
            if random.random() < 0.5:
                data = torch.flip(data, [2])
            if random.random() < 0.5:
                data = torch.flip(data, [3])

        y = torch.max(data[1:4], dim=1)[0]
        r = torch.sum(data[4:], dim=1) / data.shape[1]
        y = torch.cat([y, r], dim=0)

        if self.geometric:
            x_ = rearrange(data[0, :, self.pad:-self.pad, self.pad:-self.pad], "R θ Z -> (Z θ) R")
            y_ = rearrange(y[:, self.pad:-self.pad, self.pad:-self.pad], "C θ Z -> (Z θ) C")

            locs = np.unique(np.concatenate((locs[0], locs[1])))
            locs = torch.tensor(locs, dtype=torch.long)

            # get ctl data
            vertices, normals = self.transform.tube_vertices(data.shape[-1] - 2 * self.pad)
            edges, edge_attrs = self.transform._generate_multiscale_edges(data.shape[-1] - 2 * self.pad, [0, 3, 7])
            data = Data(pos=vertices, normal=normals)
            data = self.transform.multiscale_tube_graph(data, edges, edge_attrs)
            data = SimpleGeometry(gauge_def="random")(data)

            return Data(
                x=x_,
                y=y_,
                locs=locs,
                connection=data.connection,
                edge_coords=data.edge_coords,
                edge_index=data.edge_index,
                frame=data.frame,
                normal=data.normal,
                pos=data.pos,
                weight=data.weight,
                node_mask=data.node_mask,
                edge_mask=data.edge_mask,
            )
        else:
            data = data[:1]

        if augment:
            return data, y[:, self.pad:-self.pad, self.pad:-self.pad]
        else:
            locs = np.unique(np.concatenate((locs[0], locs[1])))
            return data, y[:, self.pad:-self.pad, self.pad:-self.pad], locs

    def get_location(self, label_l):
        if self.modality == "IVUS":
            locs = [label_l.attrs["loc_wire"], label_l.attrs["loc_calc"],
                    label_l.attrs["loc_wire_not_calc"], label_l.attrs["loc_bif"]]

            if random.random() < 0.33 and len(locs[1]):
                l = locs[1][(locs[1] > self.pad) & (locs[1] < label_l.shape[2] - self.pad)]
            elif random.random() < 0.66 and len(locs[3]):
                l = locs[3][(locs[3] > self.pad) & (locs[3] < label_l.shape[2] - self.pad)]
            else:
                l = locs[2][(locs[2] > self.pad) & (locs[2] < label_l.shape[2] - self.pad)]
        else:
            locs = [label_l.attrs["loc_calc"], label_l.attrs["loc_bif"]]

            if random.random() < 0.5 and len(locs[0]):
                l = locs[0][(locs[0] > self.pad) & (locs[0] < label_l.shape[2] - self.pad)]
            else:
                l = locs[1][(locs[1] > self.pad) & (locs[1] < label_l.shape[2] - self.pad)]

        if not len(l):  # failsafe
            l = np.arange(self.pad + 1, label_l.shape[2] - self.pad - 1)
        return locs, random.choice(l)

    def norm(self, x: np.ndarray):
        if self.config["modality"] == "IVUS":
            x = x / 255
        else:
            x = x - 1024 if x.min() >= -10 else x

            x = np.clip(x, -760, 1240)
            x = (x + 760) / 2000

        return x

    @staticmethod
    def augment_intensities(image):
        if np.random.random() < 0.1:
            image = augment_gaussian_noise(image, (0, 0.1))
        if np.random.random() < 0.1:
            image = augment_gaussian_blur(image, (0.5, 1.5))
        if np.random.random() < 0.15:
            image = augment_brightness_multiplicative(image, (0.75, 1.25))
        if np.random.random() < 0.15:
            image = augment_contrast(image, (0.75, 1.25))
        if np.random.random() < 0.125:
            image = augment_linear_downsampling_scipy(image)
        if np.random.random() < 0.1:
            image = augment_gamma(image, (0.7, 1.5), retain_stats=True, invert_image=True)
        if np.random.random() < 0.3:
            image = augment_gamma(image, (0.7, 1.5), retain_stats=True)

        return image

    def _get_h5_file(self, h5_path):
        # if the file is not in the cache, open it and cache the handle
        if h5_path not in self.h5_cache:
            self.h5_cache[h5_path] = h5py.File(h5_path, "r")
        return self.h5_cache[h5_path]

    def __del__(self):
        # ensure that all open h5py files are closed when the dataset is destroyed
        if hasattr(self, '_h5_cache'):
            for f in self.h5_cache.values():
                try:
                    f.close()
                except Exception:
                    pass


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["n_workers"]

        # Datasets will be created in the setup stage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if self.config["geometric"]:
            self.dataloader = DataLoaderGeo
        else:
            self.dataloader = DataLoader

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.config["mode"] == "train_val":
                self.train_dataset = IVUSDataset(self.config, mode="train_val")
            else:
                self.train_dataset = IVUSDataset(self.config, mode="train")
            self.val_dataset = IVUSDataset(self.config, mode="val")

    def train_dataloader(self):
        return self.dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # shuffle for training
        )

    def val_dataloader(self):
        return self.dataloader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class RegDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.c_d, self.c_m, self.c_o = config["DATA"], config["MODEL"], config["OPTIMIZATION"]
        self.pad = self.c_d["padding"]

        # set transforms
        self.classifier = self.str_to_attr(self.c_m["name"]["classifier"])(self.c_m)
        self.transform = MPRTransform(self.c_d["mpr_transform"], device="cpu")
        self.transform_polar = PolarTransform(self.c_d["polar_transform"], device=f"cuda:{self.c_o['gpu']}")

        # data dicts
        self.id_list, self.data = self.setup()

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        return self.data[self.id_list[index]]

    def setup(self):
        id_list = [self.c_d["sample_id"]]

        # load images
        data = {}
        for id_ in id_list:
            path_CCTA = join(self.c_d["root_dir"], f"CTTr", f"{id_.split('_')[0]}_0000.nii.gz")
            path_CLUM = join(self.c_d["root_dir"], f"MPRlabels2Tr", f"{id_}.nii.gz")
            path_IVUS = join(self.c_d["root_dir"], f"IVUSTr", f"{id_}_0000.nii.gz")
            path_ILUM = join(self.c_d["root_dir"], f"IVUSlabels2Tr", f"{id_}.nii.gz")
            path_CTL = [join(self.c_d["root_dir"], f"ctlTr", f"{id_}_0000.csv"),
                        join(self.c_d["root_dir"], f"ctlTr", f"{id_}.npy")]

            # IVUS data
            ivus, spacing, _ = sitk_to_numpy(path_IVUS)
            segm, _, _ = sitk_to_numpy(path_ILUM)
            ivus = self.norm(ivus, modality="IVUS")
            data_ivus = {"image": torch.tensor(ivus, dtype=torch.float32),
                         "lumen": torch.tensor(segm, dtype=torch.float32),
                         "spacing": torch.tensor(spacing),
                         "offset": torch.zeros(3),
                         "image_polar": (self.classify(ivus))[0],
                         "output_polar": (self.classify(ivus))[1],
                         "mask_polar": (self.classify(ivus))[2]}

            # CCTA data
            ccta, spacing, offset = sitk_to_numpy(path_CCTA)
            ccta = self.norm(ccta, modality="CCTA")
            data_ccta = {"image": torch.tensor(ccta, dtype=torch.float32),
                         "spacing": torch.tensor(spacing), "offset": torch.tensor(offset)}

            # centerline data
            ctlR = np.loadtxt(path_CTL[0], delimiter=",", dtype=float)
            ctlJ = np.load(path_CTL[1]) - np.array(offset)
            ctl_data = {"R": {"ctl": torch.tensor(ctlR[:, :3] - np.array(offset), dtype=torch.float32),
                              "tu": torch.tensor(ctlR[:, 3:], dtype=torch.float32)},
                        "J": torch.tensor(ctlJ, dtype=torch.float32)}
            self.transform.spacing[2] = np.linalg.norm(ctlR[:-1, :3] - ctlR[1:, :3], axis=1).mean()
            self.transform.spacing_polar[2] = torch.tensor(self.transform.spacing[2], dtype=torch.float32)

            # initial MPR transform
            mpr = self.transform((data_ccta["image"], data_ccta["spacing"]), ctl_data["J"])
            seg, _, _ = sitk_to_numpy(path_CLUM)
            data_mpr = {"image": mpr,
                        "lumen": torch.tensor(seg.astype(np.float32), dtype=torch.float32),
                        "spacing": torch.tensor(self.transform.spacing, dtype=torch.float32), "offset": torch.zeros(3)}

            data[id_] = {"IVUS": data_ivus, "CCTA": data_ccta, "CTL": ctl_data, "MPR": data_mpr}

        # everything to cpu
        for id_ in id_list:
            for k in data[id_]:
                for k2 in data[id_][k]:
                    if isinstance(data[id_][k][k2], torch.Tensor):
                        data[id_][k][k2] = data[id_][k][k2].cpu().detach()

        return id_list, data

    @staticmethod
    def norm(x: np.ndarray, modality: str):
        if modality == "IVUS":
            x = x / 255
        else:
            x = x - 1024 if x.min() >= -10 else x

            x = np.clip(x, -760, 1240)
            x = (x + 760) / 2000

        return x

    @staticmethod
    def str_to_attr(name):
        return getattr(sys.modules[__name__], name)

    def classify(self, x):
        x = np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad)), mode="edge")
        x = self.transform_polar(torch.from_numpy(x).float().to(f"cuda:{self.c_o['gpu']}"))
        x = rearrange(x, "1 C Z R θ -> 1 C R θ Z")

        # apply classifier to the input data
        self.classifier.eval().to(f"cuda:{self.c_o['gpu']}")
        with torch.no_grad():
            for m, model in enumerate(self.c_m["ckpts_IVUS"]):
                state_dict = torch.load(join(model, "model", "last.ckpt"), map_location=f"cuda:{self.c_o['gpu']}")
                state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
                self.classifier.load_state_dict(state_dict)
                if m == 0:
                    output = self.classifier(x)
                else:
                    output += self.classifier(x)

        output = rearrange(output, "B C R θ Z -> (B C) R θ Z")
        output = torch.sigmoid(output / (m + 1))

        # get mask
        mask = 1 - output[:1] > 0.75
        mask = repeat(mask, "C R θ Z -> C (n R) θ Z", n=x.shape[2]).clone()
        mask[:, :6], mask[:, 56:] = 0, 0

        return x[0, ..., self.pad:-self.pad, self.pad:-self.pad], output[1:], mask


class RegDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["DATA"]["batch_size"]
        self.num_workers = config["DATA"]["n_workers"]

        # Datasets will be created in the setup stage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = RegDataset(self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # we are simply optimizing a registration pair
        )
