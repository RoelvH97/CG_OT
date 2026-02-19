# import necessary libraries
import argparse
import copy
import datetime
import gc
import json
import lightning.pytorch as pl
import numpy as np
import os
import torch

from data import RegDataModule
from glob import glob
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from registration import LightningReg
from os.path import dirname, join


def register(config):
    now = datetime.datetime.now()
    now = f"{str(now.day).zfill(2)}-{str(now.month).zfill(2)}-{now.year}_"\
          f"{str(now.hour).zfill(2)}-{str(now.minute).zfill(2)}-{str(now.second).zfill(2)}"
    config_d, config_g, config_m, config_o = config["DATA"], config["GENERAL"], config["MODEL"], config["OPTIMIZATION"]
    config_d["padding"] = (len(config_m["dim"]) - 1) * 2
    config_d["polar_transform"]["p_theta"] = config_d["padding"]
    config_d["mpr_transform"]["p_theta"] = config_d["padding"]

    # init
    pl.seed_everything(config["GENERAL"]["seed"])

    # save config
    config_file = join(os.getcwd(), f"lightning_logs/{config_g['name']}/{config_d['sample_id']}/{now}", "config.json")
    os.makedirs(dirname(config_file), exist_ok=True)
    with open(config_file, "w") as outfile:
        json.dump(config, outfile)

    data_module = RegDataModule(config)
    data_module.setup()
    model = LightningReg(data_module.train_dataset[0], config)

    if config_o["lr_policy"] == "CosineAnnealingWarmRestarts":
        max_epochs = config_o["n_warmup"] + sum(config_o["scheduler"]["T_0"] * config_o["scheduler"]["T_mult"] ** i
                                                for i in range(config_o["n_cycles"]))
    else:
        max_epochs = config_o["max_epochs"]

    # outline trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath= f"lightning_logs/{config_g['name']}/{config_d['sample_id']}/{now}",
        save_top_k=5,
        save_last=True,
        monitor=f"val/{config_m['metric']}",
        mode="max",
    )
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{config_g['name']}/{config_d['sample_id']}",
        version=now,
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback,
                   LearningRateMonitor(logging_interval='epoch'),
                   RichProgressBar()],
        check_val_every_n_epoch=config_o["eval_every"],
        deterministic=False,
        devices=[config_o["gpu"]],
        gradient_clip_algorithm="norm",
        gradient_clip_val=config_o["clip_grad"],
        logger=logger,
        max_epochs=max_epochs,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, data_module)

    del data_module, model, trainer
    gc.collect()
    torch.cuda.empty_cache()

SAMPLE_IDS = ["sample_001"]


def cap_tube_mesh(mesh):
    """Close open ends of a tube mesh by adding fan caps at each boundary loop."""
    import trimesh

    # find boundary edges (edges belonging to only one face)
    edges_sorted = np.sort(mesh.edges, axis=1)
    edge_keys = edges_sorted[:, 0] * len(mesh.vertices) + edges_sorted[:, 1]
    unique_keys, counts = np.unique(edge_keys, return_counts=True)
    boundary_keys = unique_keys[counts == 1]
    boundary_edges = np.column_stack([
        boundary_keys // len(mesh.vertices),
        boundary_keys % len(mesh.vertices),
    ]).astype(int)

    # build adjacency and trace boundary loops
    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = start
        while True:
            nbs = [n for n in adj[cur] if n not in visited]
            if not nbs:
                break
            cur = nbs[0]
            loop.append(cur)
            visited.add(cur)
        loops.append(loop)

    # add a fan cap for each loop
    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()
    for loop in loops:
        center = vertices[loop].mean(axis=0)
        center_idx = len(vertices)
        vertices = np.vstack([vertices, center.reshape(1, 3)])
        for i in range(len(loop)):
            faces.append([loop[i], loop[(i + 1) % len(loop)], center_idx])

    capped = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    trimesh.repair.fix_normals(capped)
    return capped


def compute_mesh_dice(mesh_path_a, mesh_path_b, voxel_size=0.3):
    """
    Compute volumetric Dice score between two open tube meshes.

    Caps both meshes to make them watertight, voxelizes them into a shared
    256^3 volume using VTK stencil rasterization, and computes overlap.

    Args:
        mesh_path_a: path to first STL file
        mesh_path_b: path to second STL file
        voxel_size: isotropic voxel spacing (same units as the meshes)

    Returns:
        dice: float in [0, 1]
    """
    import trimesh
    import pyvista as pv
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    mesh_a = cap_tube_mesh(trimesh.load(mesh_path_a))
    mesh_b = cap_tube_mesh(trimesh.load(mesh_path_b))

    # center both meshes in a 256^3 volume
    lo = np.minimum(mesh_a.bounds[0], mesh_b.bounds[0])
    hi = np.maximum(mesh_a.bounds[1], mesh_b.bounds[1])
    center = (lo + hi) / 2
    shape = (256, 256, 256)
    spacing = (voxel_size, voxel_size, voxel_size)
    offset = center - np.array(shape) * voxel_size / 2

    def to_mask(mesh):
        # trimesh -> pyvista PolyData (which subclasses vtkPolyData)
        n = len(mesh.faces)
        faces_pv = np.column_stack([np.full(n, 3, dtype=int), mesh.faces]).ravel()
        polydata = pv.PolyData(np.ascontiguousarray(mesh.vertices), faces_pv)

        transform = vtk.vtkTransform()
        transform.Translate([-o for o in offset])

        tf = vtk.vtkTransformFilter()
        tf.SetInputData(polydata)
        tf.SetTransform(transform)

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputConnection(tf.GetOutputPort())
        stencil.SetOutputSpacing(*spacing)
        stencil.SetOutputOrigin(0, 0, 0)
        stencil.SetOutputWholeExtent(0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)

        to_image = vtk.vtkImageStencilToImage()
        to_image.SetInputConnection(stencil.GetOutputPort())
        to_image.SetOutsideValue(0)
        to_image.SetInsideValue(1)
        to_image.Update()

        vtk_img = to_image.GetOutput()
        dims = vtk_img.GetDimensions()
        mask = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
        mask = mask.reshape(dims[2], dims[1], dims[0])
        return np.transpose(mask, (2, 1, 0))

    mask_a = to_mask(mesh_a)
    mask_b = to_mask(mesh_b)

    intersection = np.sum((mask_a > 0) & (mask_b > 0))
    denom = np.sum(mask_a > 0) + np.sum(mask_b > 0)
    if denom == 0:
        return 0.0
    return float(2.0 * intersection / denom)


def eval(config):
    from scipy.stats import wilcoxon
    from itertools import combinations
    import pandas as pd

    methods = {
        "method_a": "lightning_logs/method_a",
        "method_b": "lightning_logs/method_b",
    }

    metric_names = ["F1", "dice_lumen", "dice_mesh", "cos_u", "cos_v"]
    method_results = {name: {m: [] for m in metric_names} for name in methods}

    for method_name, dir_ in methods.items():
        print(f"\n{'=' * 60}")
        print(f"Collecting results: {method_name}")
        print(f"{'=' * 60}")

        id_list = sorted(os.listdir(dir_))
        for id_ in id_list:
            files_ = glob(f"{dir_}/{id_}/*/metrics/metrics_249.json")
            metrics = json.load(open(files_[-1], "r"))
            run_dir = dirname(dirname(files_[-1]))

            method_results[method_name]["F1"].append(metrics["F1"])
            method_results[method_name]["dice_lumen"].append(metrics["dice_lumen"])
            method_results[method_name]["cos_u"].append(np.mean(metrics["cos_u"]))
            method_results[method_name]["cos_v"].append(np.mean(metrics["cos_v"]))

            # mesh Dice between reference and final prediction (cached)
            dice_cache = join(run_dir, "metrics", "mesh_dice.json")
            if os.path.exists(dice_cache):
                dice_mesh = json.load(open(dice_cache, "r"))["dice_mesh"]
                print(f"  {id_}: dice_mesh = {dice_mesh:.4f} (cached)")
            else:
                mesh_ref = join(run_dir, "images", "mesh_ref.stl")
                mesh_pred = join(run_dir, "images", "mesh_250.stl")
                if os.path.exists(mesh_ref) and os.path.exists(mesh_pred):
                    try:
                        dice_mesh = compute_mesh_dice(mesh_ref, mesh_pred, voxel_size=0.3)
                    except IndexError:
                        print(f"  {id_}: Error computing mesh Dice, likely due to invalid STL files.")
                        dice_mesh = 0.0
                    with open(dice_cache, "w") as f:
                        json.dump({"dice_mesh": dice_mesh}, f)
                    print(f"  {id_}: dice_mesh = {dice_mesh:.4f}")
                else:
                    dice_mesh = float("nan")
                    print(f"  {id_}: mesh files not found, skipping dice_mesh")
            method_results[method_name]["dice_mesh"].append(dice_mesh)

    # Statistical comparison
    method_names = list(method_results.keys())
    n_comparisons = max(1, len(list(combinations(method_names, 2))) * len(metric_names))
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    print(f"\n{'=' * 60}")
    print("STATISTICAL COMPARISON")
    print(f"{'=' * 60}")
    print(f"\nUsing Bonferroni correction: alpha = {alpha}/{n_comparisons} = {bonferroni_alpha:.6f}")

    pvalue_summary = []

    for metric_name in metric_names:
        print(f"\n{'-' * 60}")
        print(f"  {metric_name}:")
        print(f"{'-' * 60}")

        print("    Means +/- Std:")
        for method_name in method_names:
            values = method_results[method_name][metric_name]
            if values:
                q25, q75 = np.percentile(values, 25), np.percentile(values, 75)
                print(f"      {method_name:25s}: {np.mean(values):.4f} +/- {np.std(values):.4f},"
                      f" median {np.median(values):.4f} IQR [{q25:.4f}, {q75:.4f}] (n={len(values)})")

        print("\n    Pairwise comparisons (Wilcoxon signed-rank test):")
        for method1, method2 in combinations(method_names, 2):
            values1 = np.array(method_results[method1][metric_name])
            values2 = np.array(method_results[method2][metric_name])

            if len(values1) == 0 or len(values2) == 0:
                continue

            min_len = min(len(values1), len(values2))
            values1 = values1[:min_len]
            values2 = values2[:min_len]

            try:
                _, p_value = wilcoxon(values1, values2)
                p_value = float(p_value)
                mean_diff = float(np.mean(values1) - np.mean(values2))

                significance = "***" if p_value < bonferroni_alpha else \
                    "**" if p_value < 0.01 else \
                        "*" if p_value < 0.05 else "ns"

                print(f"      {method1:25s} vs {method2:25s}: p={p_value:.6f} {significance:3s} (d={mean_diff:+.4f})")

                pvalue_summary.append({
                    'Metric': metric_name,
                    'Method1': method1,
                    'Method2': method2,
                    'p_value': p_value,
                    'mean_diff': mean_diff,
                    'significance': significance
                })
            except Exception as e:
                print(f"      {method1:25s} vs {method2:25s}: Error - {str(e)}")

    # Summary
    df_pvalues = pd.DataFrame(pvalue_summary)
    df_pvalues.to_csv('registration_statistical_comparison.csv', index=False)
    print(f"\nFull statistical results saved to 'registration_statistical_comparison.csv'")

    print(f"\n{'=' * 60}")
    print("SIGNIFICANT DIFFERENCES (p < 0.05)")
    print(f"{'=' * 60}")
    significant = df_pvalues[df_pvalues['p_value'] < 0.05].sort_values('p_value')
    if len(significant) > 0:
        for _, row in significant.iterrows():
            print(f"{row['Metric']:15s} | "
                  f"{row['Method1']:25s} vs {row['Method2']:25s} | "
                  f"p={row['p_value']:.6f} | d={row['mean_diff']:+.4f}")
    else:
        print("No significant differences found.")

    return method_results, df_pvalues

if __name__ == "__main__":
    # parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The configuration file')

    args = parser.parse_args()
    config = json.load(open(args.config))

    if config['GENERAL']['mode'] == "register":
        for i, sample_id in enumerate(SAMPLE_IDS):
            print("=" * 40)
            print(f"Processing {sample_id} ({i + 1}/{len(SAMPLE_IDS)})")
            print("=" * 40)

            run_config = copy.deepcopy(config)
            run_config["DATA"]["sample_id"] = sample_id
            register(run_config)

            print(f"Completed {sample_id}\n")
    else:
        eval(config)
