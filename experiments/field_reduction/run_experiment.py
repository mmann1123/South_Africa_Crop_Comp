"""Orchestrator for the field reduction experiment.

Dispatches training scripts via subprocess using the correct conda environment,
then runs OOS prediction and analysis.

Usage:
    python run_experiment.py                    # Run all models (skips TabNet)
    python run_experiment.py --models tabnet_pixel  # Run only TabNet
    python run_experiment.py --skip-tabnet      # Explicitly skip TabNet
    python run_experiment.py --fractions 0.50 0.75  # Specific fractions
    python run_experiment.py --skip-training    # Only predict + analyze
    python run_experiment.py --dry-run          # Show commands without executing
"""

import argparse
import os
import subprocess
import sys
import time

from experiment_config import (
    EXPERIMENT_DIR,
    MODELS_DIR,
    FRACTIONS,
    MODEL_ENV_MAP,
    MODEL_SCRIPT_MAP,
    L2_REG_LAMBDA,
)

sys.stdout.reconfigure(line_buffering=True)

ALL_MODELS = list(MODEL_SCRIPT_MAP.keys())

# Estimated minutes per fraction for each model
TIME_ESTIMATES = {
    "tabnet_pixel": 150,
    "ltae_pixel": 26,
    "ltae_field": 1,
    "xgboost_field": 2,
    "base_lgbm_pixel": 0.5,
    "base_lr_pixel": 8,
    "xgboost_field_l2": 2,
    "base_lgbm_pixel_l2": 9,
}


def build_train_cmd(model_name, fraction, python_exe):
    """Build the subprocess command for a training script."""
    script = MODEL_SCRIPT_MAP[model_name]
    frac_str = f"{fraction:.2f}"
    is_l2 = model_name.endswith("_l2")

    if model_name in ("base_lgbm_pixel", "base_lr_pixel"):
        # base_ml.py trains both LightGBM and LR together
        lgbm_dir = os.path.join(MODELS_DIR, "base_lgbm_pixel", f"frac_{frac_str}")
        lr_dir = os.path.join(MODELS_DIR, "base_lr_pixel", f"frac_{frac_str}")
        return [
            python_exe, script,
            "--fraction", str(fraction),
            "--output-dir-lgbm", lgbm_dir,
            "--output-dir-lr", lr_dir,
        ]
    elif model_name == "base_lgbm_pixel_l2":
        # L2 variant: trains LightGBM with reg_lambda, LR output goes to a throwaway dir
        lgbm_dir = os.path.join(MODELS_DIR, "base_lgbm_pixel_l2", f"frac_{frac_str}")
        lr_dir = os.path.join(MODELS_DIR, "base_lr_pixel", f"frac_{frac_str}")
        return [
            python_exe, script,
            "--fraction", str(fraction),
            "--output-dir-lgbm", lgbm_dir,
            "--output-dir-lr", lr_dir,
            "--reg-lambda", str(L2_REG_LAMBDA),
        ]
    else:
        output_dir = os.path.join(MODELS_DIR, model_name, f"frac_{frac_str}")
        cmd = [
            python_exe, script,
            "--fraction", str(fraction),
            "--output-dir", output_dir,
        ]
        if is_l2:
            cmd += ["--reg-lambda", str(L2_REG_LAMBDA)]
        return cmd


def run_command(cmd, dry_run=False):
    """Run a command, printing it first."""
    cmd_str = " ".join(cmd)
    print(f"\n$ {cmd_str}")
    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=EXPERIMENT_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        return False
    else:
        print(f"  OK ({elapsed:.0f}s / {elapsed/60:.1f}min)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Field reduction experiment orchestrator")
    parser.add_argument('--models', nargs='*', default=None,
                        help="Models to run (default: all except tabnet)")
    parser.add_argument('--fractions', nargs='*', type=float, default=None,
                        help=f"Fractions to test (default: {FRACTIONS})")
    parser.add_argument('--skip-tabnet', action='store_true',
                        help="Skip TabNet (slowest model)")
    parser.add_argument('--skip-training', action='store_true',
                        help="Skip training, only run prediction + analysis")
    parser.add_argument('--skip-prediction', action='store_true',
                        help="Skip prediction, only run analysis")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print commands without executing")
    args = parser.parse_args()

    fractions = args.fractions or FRACTIONS
    models = args.models or ALL_MODELS

    if args.skip_tabnet and "tabnet_pixel" in models:
        models = [m for m in models if m != "tabnet_pixel"]

    # Default: skip tabnet unless explicitly requested
    if args.models is None and not any(m == "tabnet_pixel" for m in (args.models or [])):
        if "tabnet_pixel" in models and not args.skip_tabnet:
            # Only include tabnet if user explicitly asked for it
            pass

    # Deduplicate base_ml (train_base_ml.py trains both LightGBM and LR)
    # Only deduplicate the non-L2 pair; L2 variant runs separately
    train_models = []
    base_ml_added = False
    for m in models:
        if m in ("base_lgbm_pixel", "base_lr_pixel"):
            if not base_ml_added:
                train_models.append(m)
                base_ml_added = True
        else:
            train_models.append(m)

    # Print plan
    print("=" * 60)
    print("FIELD REDUCTION EXPERIMENT")
    print("=" * 60)
    print(f"Models:    {models}")
    print(f"Fractions: {fractions}")
    total_min = sum(TIME_ESTIMATES.get(m, 5) * len(fractions) for m in models)
    print(f"Estimated: ~{total_min:.0f} min ({total_min/60:.1f} hrs)")
    print("=" * 60)

    # =================== Training ===================
    if not args.skip_training:
        print("\n\n===== PHASE 1: TRAINING =====")
        for frac in fractions:
            for model_name in train_models:
                python_exe = MODEL_ENV_MAP[model_name]
                cmd = build_train_cmd(model_name, frac, python_exe)
                ok = run_command(cmd, dry_run=args.dry_run)
                if not ok:
                    print(f"  WARNING: {model_name} frac={frac} failed, continuing...")
    else:
        print("\n[SKIP] Training phase")

    # =================== Prediction ===================
    if not args.skip_prediction:
        print("\n\n===== PHASE 2: OOS PREDICTION =====")
        # Use the DL python for predict_oos.py (it handles both DL + ML models)
        predict_script = os.path.join(EXPERIMENT_DIR, "predict_oos.py")

        # Separate DL and ML models for correct conda env
        dl_models = [m for m in models if MODEL_ENV_MAP[m] != MODEL_ENV_MAP.get("xgboost_field")]
        ml_models = [m for m in models if MODEL_ENV_MAP[m] == MODEL_ENV_MAP.get("xgboost_field")]

        frac_args = [str(f) for f in fractions]

        if dl_models:
            dl_python = MODEL_ENV_MAP[dl_models[0]]
            cmd = [dl_python, predict_script, "--models"] + dl_models + ["--fractions"] + frac_args
            run_command(cmd, dry_run=args.dry_run)

        if ml_models:
            ml_python = MODEL_ENV_MAP[ml_models[0]]
            cmd = [ml_python, predict_script, "--models"] + ml_models + ["--fractions"] + frac_args
            run_command(cmd, dry_run=args.dry_run)
    else:
        print("\n[SKIP] Prediction phase")

    # =================== Analysis ===================
    print("\n\n===== PHASE 3: ANALYSIS =====")
    analyze_script = os.path.join(EXPERIMENT_DIR, "analyze_results.py")
    # Use ML python for analysis (lighter dependencies)
    ml_python = MODEL_ENV_MAP.get("xgboost_field", sys.executable)
    run_command([ml_python, analyze_script], dry_run=args.dry_run)

    print("\n\nDone!")


if __name__ == "__main__":
    main()
