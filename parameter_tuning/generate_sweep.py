"""
generate_sweep.py

Generates experiment configs and submits SLURM jobs.

Usage:
    python generate_sweep.py             # runs all stages
    python generate_sweep.py --stage 1   # runs only stage 1
"""

import json
import os
import itertools
import subprocess
import random
import argparse

random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_CFG = {
    "ckpt":             "",
    "ckpt_save_loc":    "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/parameter_tuning/models/",
    "log_dir":          "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/parameter_tuning/logs/",
    "datapointer":      "/scratch3/BMC/gpu-ghpcs/Rey.Koki/SmokeViz/deep_learning/dataset_pointers/pseudo/pseudo.pkl",
    "encoder_weights":  "None",
    "num_workers":      4,
    "n_epochs":         30,
    # defaults — overridden per stage
    "stage":                 1,
    "lr":                    1e-4,
    "batch_size":            32,
    "weight_decay":          1e-5,
    "encoder_lr_multiplier": 0.1,
    "loss_high_weight":      3,
    "loss_med_weight":       2,
    "train_augmentations":   {"rhf": 0.5, "rvf": 0.3},
}

SLURM_SCRIPT = "run_train.script"
CONFIG_DIR   = "configs"

# ── stage definitions ──────────────────────────────────────────────────────────
# Run stages in order. Fix the best result from each stage before moving on.
# Comment out stages you've already completed.

STAGES = {
    # Stage 1: find best encoder for PSPNet
    1: {
        "n_jobs": 6,
        "search": {
            "encoder": ["timm-efficientnet-b2", "timm-efficientnet-b4", "resnet50", "resnet34", "mit_b2", "mit_b3"],
        },
        "overrides": {
            "stage": 1,
            "architecture": "PSPNet",
            "n_epochs": 15,
        }
    },

    # Stage 2: tune optimizer
    6: {
        "n_jobs": 12,
        "search": {
            "lr":         [1e-3, 5e-4, 2.5e-4, 1e-4],
            "batch_size": [32, 64, 128],
            "weight_decay":          [1e-5],
            "encoder_lr_multiplier": [0.1],
        },
        "overrides": {
            "stage": 6,
            "architecture": "PSPNet",
            #"encoder":      "timm-efficientnet-b2",  # <-- update after stage 1
            "encoder":      "resnet34",  # <-- update after stage 1
        }
    },

    # Stage 3: tune enc_mult around exp57/58 sweet spot
    3: {
        "n_jobs": 2,
        "search": {
            "lr":                    [1e-3],
            "batch_size":            [128],
            "weight_decay":          [1e-5],
            "encoder_lr_multiplier": [0.05, 0.1],
        },
        "overrides": {
            "stage": 3,
            "architecture": "PSPNet",
            "encoder":      "resnet34",
        }
    },

    # Stage 4: loss weights + augmentation
    5: {
        "n_jobs": 12,
        "search": {
            "loss_high_weight":    [2, 3, 4],
            "loss_med_weight":     [1, 2],
            "train_augmentations": [
                {},
                {"rhf": 0.5},
                {"rhf": 0.5, "rvf": 0.3},
            ],
        },
        "overrides": {
            "stage": 5,
            "architecture":          "PSPNet",
            "encoder":               "resnet34",  # <-- update after stage 1
            "lr":                    1e-3,                    # <-- update after stage 2
            "weight_decay":          1e-5,                   # <-- update after stage 2
            "encoder_lr_multiplier": 0.1,                     # <-- update after stage 2
            "batch_size":            128,
        }
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────

def get_exp_num():
    """Find the next available experiment number across all existing configs."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    existing = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split('exp')[-1].split('.json')[0]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


def sample_combos(search_space, n_jobs):
    """Random sample from search space, or full grid if smaller than n_jobs."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_combos = list(itertools.product(*values))
    n = min(n_jobs, len(all_combos))
    selected = random.sample(all_combos, n)
    return [dict(zip(keys, combo)) for combo in selected]



def submit_job(config_fn, cfg, exp_num, stage_num):
    arch    = cfg['architecture']
    encoder = cfg['encoder']
    name    = f"s{stage_num}_{arch}_{encoder}_exp{exp_num}"[:30]  # SLURM has a 30 char limit
    result = subprocess.run(
        ["sbatch",
         f"--job-name={name}",
         f"--output=./logs/{name}_%j.log",
         f"--export=CONFIG_FN={config_fn}",
         SLURM_SCRIPT],
        capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print("ERROR:", result.stderr.strip())


def run_stage(stage_num):
    stage = STAGES[stage_num]
    combos = sample_combos(stage["search"], stage["n_jobs"])
    exp_num = get_exp_num()

    print(f"\n=== Stage {stage_num}: submitting {len(combos)} jobs ===")
    for combo in combos:
        cfg = BASE_CFG.copy()
        cfg.update(stage.get("overrides", {}))
        cfg.update(combo)

        config_fn = os.path.join(CONFIG_DIR, f"exp{exp_num}.json")
        with open(config_fn, "w") as f:
            json.dump(cfg, f, indent=4)

        print(f"  exp{exp_num}: {combo}")
        submit_job(config_fn, cfg, exp_num, stage_num)
        exp_num += 1


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=None, help="Run a specific stage (1-4). Default: run all.")
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
    else:
        for stage_num in sorted(STAGES.keys()):
            run_stage(stage_num)
