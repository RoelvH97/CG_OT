# import necessary libraries
import argparse
import copy
import datetime
import json
import lightning.pytorch as pl
import numpy as np
import os
import torch

from data import DataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from model import LightningFan
from os.path import join, dirname
from tqdm import tqdm


def train(config):
    now = datetime.datetime.now()
    now = f"{str(now.day).zfill(2)}-{str(now.month).zfill(2)}-{now.year}_"\
          f"{str(now.hour).zfill(2)}-{str(now.minute).zfill(2)}-{str(now.second).zfill(2)}"
    config_d, config_g, config_m, config_o = config["DATA"], config["GENERAL"], config["MODEL"], config["OPTIMIZATION"]
    config_d["padding"] = (len(config_m["dim"]) - 1) * 2

    # init
    pl.seed_everything(config["GENERAL"]["seed"])

    # save config
    config_file = join(os.getcwd(), f"lightning_logs/{config_g['name']}/{now}", "config.json")
    os.makedirs(dirname(config_file), exist_ok=True)
    with open(config_file, "w") as outfile:
        json.dump(config, outfile)

    data_module = DataModule(config_d)
    model = LightningFan(config)

    # outline trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/{config_g['name']}/{now}/model",
        save_top_k=5,
        save_last=True,
        monitor=f"val/TotalBettiError",
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=config_g["name"],
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
        max_epochs=config_o["n_warmup"] + sum(config_o["scheduler"]["T_0"] * config_o["scheduler"]["T_mult"] ** i
                                              for i in range(config_o["n_cycles"])),
        num_sanity_val_steps=2,
    )

    ckpt = join(os.getcwd(), "lightning_logs", config_o["resume"], "model", "last.ckpt") if config_o["resume"] else None
    trainer.fit(model, data_module, ckpt_path=ckpt)


def eval_multiple_methods(configs_dict):
    """
    Evaluate multiple methods and perform statistical comparison.

    Args:
        configs_dict: Dictionary mapping method names to config dicts
                     e.g., {"BCEDice": config1, "Dice": config2, ...}
    """
    from monai.metrics import DiceMetric
    from metrics.betti_error import BettiNumberMetric
    from metrics.cldice import ClDiceMetric
    from torchmetrics.classification import BinaryAUROC, BinaryF1Score
    from scipy.stats import wilcoxon
    from itertools import combinations
    import pandas as pd

    def radial_polygon_area(radii):
        d_theta = torch.tensor(2 * torch.pi / radii.shape[2], device=radii.device, dtype=radii.dtype)
        r_next = torch.roll(radii, shifts=-1, dims=2)
        return 0.5 * torch.sin(d_theta) * torch.sum(radii * r_next, dim=2)

    def radial_dice(reference, prediction):
        intersection_radii = torch.minimum(reference, prediction)
        area_ref = radial_polygon_area(reference)
        area_pred = radial_polygon_area(prediction)
        area_intersection = radial_polygon_area(intersection_radii)

        area_ref = torch.sum(area_ref)
        area_pred = torch.sum(area_pred)
        area_intersection = torch.sum(area_intersection)
        tp_area = area_intersection
        dice = 2 * tp_area / (area_pred + area_ref + 1e-8)
        return dice

    labels = ["Guidewire", "Calcifications", "Bifurcations", "Lumen"]

    # Store results per method, per patient
    # Structure: method_results[method_name][label][metric_name] = [patient_1_value, patient_2_value, ...]
    method_results = {method_name: {label: {
        "asd": [],
        "auroc": [],
        "f1_macro": [],
        "dice": [],
        "cldice": [],
        "betti_0_error": [],
        "betti_1_error": [],
        "betti_matching_error": []
    } for label in labels} for method_name in configs_dict.keys()}

    # Evaluate each method
    for method_name, config in configs_dict.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating Method: {method_name}")
        print(f"{'=' * 60}\n")

        config_d, config_m, config_o = config["DATA"], config["MODEL"], config["OPTIMIZATION"]
        config_d["padding"] = (len(config_m["dim"]) - 1) * 2
        device = f"cuda:{config_o['gpu']}" if torch.cuda.is_available() else "cpu"

        if config_d["modality"] == "IVUS":
            spacing = 0.02273
        else:
            spacing = 0.02
        factor = config_d["polar_transform"]["l_rad"] / config_d["polar_transform"]["n_rad"] * spacing

        # Set up metrics
        auroc_metric = BinaryAUROC().to(device)
        f1_metric = BinaryF1Score().to(device)
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        cldice_metric = ClDiceMetric(ignore_background=False)
        betti_number_metric = BettiNumberMetric(
            num_processes=0,
            ignore_background=False,
            eight_connectivity=True
        )

        checkpoints = config_m["ckpt"] if isinstance(config_m["ckpt"], list) else [config_m["ckpt"]]

        for fold_idx, ckpt_path in enumerate(checkpoints):
            print(f"\nEvaluating Fold {fold_idx + 1}/{len(checkpoints)}")

            config_d["fold"] = fold_idx
            model = LightningFan.load_from_checkpoint(ckpt_path, config=config)
            model.eval()
            model = model.to(device)

            data_module = DataModule(config_d)
            data_module.setup(stage="fit")
            dataloader = data_module.val_dataloader() if config_d["mode"] == "val" else data_module.test_dataloader()

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Fold {fold_idx + 1}")):
                    y_pred = model.model(batch[0].to(device))
                    y_true = batch[1].to(device)

                    y_pred = torch.sigmoid(y_pred)
                    y_pred_pre = y_pred.clone()
                    y_pred[:, :3] = y_pred[:, :3] > 0.5
                    y_true[:, :3] = y_true[:, :3] > 0.5

                    for i in range(y_pred.shape[1] - 1):
                        label = labels[i]
                        a_ = y_pred[:, i:i + 1, 0]
                        b_ = y_true[:, i:i + 1]
                        c_ = y_pred_pre[:, i:i + 1, 0]

                        if i == 0 and config_d["modality"] == "IVUS":
                            a_ = a_[..., batch[2]][..., 0, :]
                            b_ = b_[..., batch[2]][..., 0, :]
                            c_ = c_[..., batch[2]][..., 0, :]
                            cldice = cldice_metric(a_, b_)
                            cldice_val = cldice.item() if hasattr(cldice, 'item') else cldice
                            method_results[method_name][label]["cldice"].append(cldice_val)
                        if i == 0 and config_d["modality"] == "MPR":
                            continue

                        auroc = auroc_metric(c_.flatten(), b_.flatten())
                        f1_macro = f1_metric(torch.max(c_, dim=2)[0].flatten(), torch.max(b_, dim=2)[0].flatten())
                        dice = dice_metric(a_, b_)
                        betti_error = betti_number_metric(a_, b_)

                        auroc_val = auroc.item() if hasattr(auroc, 'item') else auroc
                        method_results[method_name][label]["auroc"].append(auroc_val)

                        f1_macro_val = f1_macro.item() if hasattr(f1_macro, 'item') else f1_macro
                        method_results[method_name][label]["f1_macro"].append(f1_macro_val)

                        dice_val = dice.item() if hasattr(dice, 'item') else dice
                        if not np.isnan(dice_val):
                            method_results[method_name][label]["dice"].append(dice_val)

                        if isinstance(betti_error, list):
                            method_results[method_name][label]["betti_0_error"].append(betti_error[0])
                            method_results[method_name][label]["betti_1_error"].append(betti_error[1])
                            method_results[method_name][label]["betti_matching_error"].append(betti_error[2])

                    dice = radial_dice(y_true[:, -1:], y_pred[:, -1:, 0])
                    dice_val = dice.item() if hasattr(dice, 'item') else dice
                    method_results[method_name]["Lumen"]["dice"].append(dice_val)

                    asd = torch.mean(torch.abs((y_true[:, -1:] - y_pred[:, -1:, 0]) * factor))
                    asd_val = asd.item() if hasattr(asd, 'item') else asd
                    method_results[method_name]["Lumen"]["asd"].append(asd_val)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Now perform statistical comparisons
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON")
    print("=" * 60)

    # Get all method pairs
    method_names = list(configs_dict.keys())
    n_comparisons = len(list(combinations(method_names, 2))) * len(labels) * 8  # rough estimate
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    print(f"\nUsing Bonferroni correction: alpha = {alpha}/{n_comparisons} = {bonferroni_alpha:.6f}")

    # Store p-values for heatmap
    pvalue_summary = []

    for label in labels:
        print(f"\n{'-' * 60}")
        print(f"{label}")
        print(f"{'-' * 60}")

        for metric_name in ["asd", "auroc", "f1_macro", "dice", "cldice",
                            "betti_0_error", "betti_1_error", "betti_matching_error"]:

            # Check if any method has this metric
            has_metric = any(method_results[m][label][metric_name] for m in method_names)
            if not has_metric:
                continue

            print(f"\n  {metric_name}:")

            # First print means +/- stds
            print("    Means +/- Std:")
            for method_name in method_names:
                values = method_results[method_name][label][metric_name]
                if values:
                    print(f"      {method_name:20s}: {np.mean(values):.4f} +/- {np.std(values):.4f} (n={len(values)})")

            # Then pairwise comparisons
            print("\n    Pairwise comparisons (Wilcoxon signed-rank test):")
            for method1, method2 in combinations(method_names, 2):
                values1 = np.array(method_results[method1][label][metric_name])
                values2 = np.array(method_results[method2][label][metric_name])

                if len(values1) == 0 or len(values2) == 0:
                    continue

                # Ensure same length (should be if same patients)
                min_len = min(len(values1), len(values2))
                values1 = values1[:min_len]
                values2 = values2[:min_len]

                # Wilcoxon signed-rank test (paired)
                try:
                    statistic, p_value = wilcoxon(values1, values2)

                    p_value = float(p_value)
                    mean_diff = float(np.mean(values1) - np.mean(values2))

                    significance = "***" if p_value < bonferroni_alpha else \
                        "**" if p_value < 0.01 else \
                            "*" if p_value < 0.05 else "ns"

                    print(
                        f"      {method1:20s} vs {method2:20s}: p={p_value:.6f} {significance:3s} (d={mean_diff:+.4f})")

                    pvalue_summary.append({
                        'Structure': label,
                        'Metric': metric_name,
                        'Method1': method1,
                        'Method2': method2,
                        'p_value': p_value,
                        'mean_diff': mean_diff,
                        'significance': significance
                    })
                except Exception as e:
                    print(f"      {method1:20s} vs {method2:20s}: Error - {str(e)}")

    # Create summary DataFrame
    df_pvalues = pd.DataFrame(pvalue_summary)

    # Save to CSV
    df_pvalues.to_csv('statistical_comparison.csv', index=False)
    print(f"\n\nFull statistical results saved to 'statistical_comparison.csv'")

    # Print significant findings
    print("\n" + "=" * 60)
    print("SIGNIFICANT DIFFERENCES (p < 0.05)")
    print("=" * 60)
    significant = df_pvalues[df_pvalues['p_value'] < 0.05].sort_values('p_value')
    if len(significant) > 0:
        for _, row in significant.iterrows():
            print(f"{row['Structure']:15s} | {row['Metric']:20s} | "
                  f"{row['Method1']:15s} vs {row['Method2']:15s} | "
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
    print(json.dumps(config, sort_keys=True, indent=4))

    if config['GENERAL']['mode'] == "train":
        name = copy.deepcopy(config['GENERAL']["name"])
        for i in range(5):
            config['GENERAL']["name"] = f"{name}_{i}"
            config["DATA"]["fold"] = i
            train(config)
    elif config['GENERAL']['mode'] == "eval":
        configs_dict = {
            config['GENERAL']['name']: config,
        }
        results, pvalues_df = eval_multiple_methods(configs_dict)
