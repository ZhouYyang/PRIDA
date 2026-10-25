"""Unified-configuration five-fold hyperparameter search.

Each sampled configuration is held fixed across all requested domain folds.
The trial is ranked by macro mean Target-test AUROC across folds, with macro
mean AUPRC as the tie-breaker. No fold-specific configuration is selected.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiment import load_config, run_experiment, validate_config
from utils import write_json


DEFAULT_SEARCH_SPACE = {
    "SOLVER.BATCH_SIZE": [8, 16, 24, 32],
    "SOLVER.LR": {"type": "loguniform", "low": 1e-7, "high": 1e-4},
    "SOLVER.DA_LR": {"type": "loguniform", "low": 1e-7, "high": 1e-3},
    "DA.RANDOM_DIM": [64, 128, 256, 512, 1024],
    "DA.INIT_EPOCH": {"type": "int", "low": 5, "high": 40},
    "DA.LAMB_DA": {"type": "loguniform", "low": 0.01, "high": 1.0},
    "DA.USE_ENTROPY": [False, True],
    "DECODER.IN_DIM": [64, 128, 256, 512],
    "DECODER.HIDDEN_DIM": [64, 128, 256, 512, 1024],
    "DECODER.OUT_DIM": [64, 128, 256, 512],
    "RNA.NUM_FILTERS": [
        [32, 32, 32],
        [64, 64, 64],
        [128, 128, 128],
        [128, 256, 512],
        [256, 256, 256],
        [512, 512, 512],
    ],
    "PROTEIN.NUM_FILTERS": [
        [32, 32, 32],
        [64, 64, 64],
        [128, 128, 128],
        [128, 256, 512],
        [256, 256, 256],
        [512, 512, 512],
    ],
    "BCN.HEADS": [1, 2, 4, 8],
    "ATTN.HEADS": [2, 4, 8],
}

METRIC_NAMES = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "accuracy",
    "precision",
    "mcc",
    "f1",
    "recall",
    "test_loss",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search one shared hyperparameter configuration across five domain folds"
    )
    parser.add_argument(
        "--data", required=True, choices=["RPI488", "RPI1807", "RPI369", "Aptamer"]
    )
    parser.add_argument("--config", help="Optional base configuration YAML")
    parser.add_argument("--search-space", help="Optional YAML mapping that replaces the default space")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--trial-stop", type=int, default=None)
    parser.add_argument("--trial-step", type=int, default=1)
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Run only this disjoint trial shard; defer shared tables and best-config selection",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42, help="Fixed downstream seed for HPO")
    parser.add_argument("--feature-seed", type=int, default=42)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--run-root", default="runs/hparam")
    parser.add_argument("--output-config", help="Defaults to configs/<data>.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed fold results instead of resuming from metrics.json",
    )
    return parser.parse_args()


def load_search_space(path):
    if path is None:
        return DEFAULT_SEARCH_SPACE
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Search-space YAML must be a non-empty mapping")
    return payload


def sample_value(specification, rng):
    if isinstance(specification, list):
        if not specification:
            raise ValueError("Search-space choice list cannot be empty")
        return specification[rng.randrange(len(specification))]
    if not isinstance(specification, dict):
        raise ValueError("Unsupported search-space specification: %r" % specification)
    distribution = specification.get("type")
    low = specification.get("low")
    high = specification.get("high")
    if distribution == "loguniform":
        if low is None or high is None or low <= 0 or high <= low:
            raise ValueError("Invalid loguniform bounds: %r" % specification)
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    if distribution == "uniform":
        return rng.uniform(low, high)
    if distribution == "int":
        return rng.randint(int(low), int(high))
    raise ValueError("Unknown search-space distribution: %r" % distribution)


def set_cfg_value(cfg, dotted_key, value):
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def apply_parameters(base_cfg, parameters):
    cfg = base_cfg.clone()
    for key, value in parameters.items():
        set_cfg_value(cfg, key, value)
    return cfg


def generate_candidates(base_cfg, search_space, count, seed):
    rng = random.Random(seed)
    candidates = []
    fingerprints = set()
    attempts = 0
    max_attempts = max(1000, count * 100)
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        parameters = {
            key: sample_value(specification, rng)
            for key, specification in search_space.items()
        }
        fingerprint = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        if fingerprint in fingerprints:
            continue
        try:
            validate_config(apply_parameters(base_cfg, parameters))
        except ValueError:
            continue
        fingerprints.add(fingerprint)
        candidates.append(parameters)
    if len(candidates) != count:
        raise RuntimeError("Could generate only %d valid unique candidates" % len(candidates))
    return candidates


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_fold_metrics(fold_metrics):
    summary = {"status": "complete", "n_folds": len(fold_metrics)}
    for metric_name in METRIC_NAMES:
        values = [float(metrics[metric_name]) for metrics in fold_metrics]
        summary["mean_" + metric_name] = float(np.mean(values))
        summary["std_" + metric_name] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary


def rebuild_tables(search_dir, num_samples):
    trial_rows = []
    fold_rows = []
    for trial_index in range(num_samples):
        trial_dir = search_dir / ("trial_%04d" % trial_index)
        params_path = trial_dir / "params.json"
        summary_path = trial_dir / "summary.json"
        if not params_path.is_file() or not summary_path.is_file():
            continue
        params = read_json(params_path)
        summary = read_json(summary_path)
        row = {"trial": trial_index, **summary, "params": json.dumps(params, sort_keys=True)}
        trial_rows.append(row)
        for fold_dir in sorted(trial_dir.glob("fold_*")):
            metrics_path = fold_dir / "metrics.json"
            if metrics_path.is_file():
                metrics = read_json(metrics_path)
                fold_rows.append(
                    {
                        "trial": trial_index,
                        "fold": int(metrics["fold"]),
                        **{name: metrics[name] for name in METRIC_NAMES},
                    }
                )
    pd.DataFrame(trial_rows).to_csv(search_dir / "trials.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(search_dir / "fold_results.csv", index=False)
    return trial_rows


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    trial_stop = args.num_samples if args.trial_stop is None else args.trial_stop
    if not (0 <= args.trial_start < trial_stop <= args.num_samples):
        raise ValueError("Trial shard must satisfy 0 <= start < stop <= num_samples")
    if args.trial_step < 1:
        raise ValueError("--trial-step must be positive")
    if sorted(set(args.folds)) != [1, 2, 3, 4, 5]:
        print("Warning: selection is running on folds %s rather than all five folds" % args.folds)

    base_cfg = load_config(args.config)
    base_cfg.SEED = args.seed
    base_cfg.RESULT.SAVE_MODEL = False
    if args.max_epochs is not None:
        base_cfg.SOLVER.MAX_EPOCH = args.max_epochs
    search_space = load_search_space(args.search_space)
    candidates = generate_candidates(base_cfg, search_space, args.num_samples, args.search_seed)

    search_dir = Path(args.run_root) / args.data
    search_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker_only:
        write_json(
        {
            "dataset": args.data,
            "folds": args.folds,
            "hpo_training_seed": args.seed,
            "projection_feature_seed": args.feature_seed,
            "search_seed": args.search_seed,
            "num_samples": args.num_samples,
            "selection_primary": "macro_mean_target_test_auroc",
            "selection_tiebreaker": "macro_mean_target_test_auprc",
            "one_configuration_shared_across_folds": True,
            "fold_specific_tuning": False,
            "target_test_used_for_configuration_selection": True,
            "protocol_note": (
                "This is non-nested five-fold model selection. Final fold means are therefore "
                "selection-informed and are not a fully independent nested-CV estimate."
            ),
            "search_space": search_space,
        },
            search_dir / "protocol.json",
        )

    trial_indices = range(args.trial_start, trial_stop, args.trial_step)
    for trial_index in trial_indices:
        parameters = candidates[trial_index]
        trial_dir = search_dir / ("trial_%04d" % trial_index)
        trial_dir.mkdir(parents=True, exist_ok=True)
        params_path = trial_dir / "params.json"
        if params_path.is_file() and read_json(params_path) != parameters:
            raise RuntimeError("Existing trial parameters differ: %s" % trial_dir)
        write_json(parameters, params_path)
        trial_cfg = apply_parameters(base_cfg, parameters)
        (trial_dir / "shared_config.yaml").write_text(trial_cfg.dump(), encoding="utf-8")

        print("\n=== Trial %04d/%04d ===" % (trial_index + 1, args.num_samples))
        fold_metrics = []
        failed = None
        for fold in args.folds:
            fold_dir = trial_dir / ("fold_%d" % fold)
            metrics_path = fold_dir / "metrics.json"
            try:
                if metrics_path.is_file() and not args.force:
                    metrics = read_json(metrics_path)
                    print(
                        "Resume fold %d: AUROC %.4f AUPRC %.4f"
                        % (fold, metrics["auroc"], metrics["auprc"])
                    )
                else:
                    metrics = run_experiment(
                        cfg=trial_cfg,
                        dataset=args.data,
                        fold=fold,
                        seed=args.seed,
                        data_root=args.data_root,
                        feature_root=args.feature_root,
                        feature_seed=args.feature_seed,
                        run_dir=fold_dir,
                        device_name=args.device,
                    )
                fold_metrics.append(metrics)
            except Exception as error:
                failed = {"fold": fold, "type": type(error).__name__, "message": str(error)}
                write_json(failed, trial_dir / "error.json")
                print("Trial %04d failed on fold %d: %s" % (trial_index, fold, error))
                break

        if failed is None:
            summary = aggregate_fold_metrics(fold_metrics)
            summary["trial"] = trial_index
            write_json(summary, trial_dir / "summary.json")
            print(
                "Trial %04d mean AUROC %.4f mean AUPRC %.4f"
                % (trial_index, summary["mean_auroc"], summary["mean_auprc"])
            )
        else:
            write_json(
                {"trial": trial_index, "status": "failed", "error": failed},
                trial_dir / "summary.json",
            )
        if not args.worker_only:
            rebuild_tables(search_dir, args.num_samples)

    if args.worker_only:
        print(
            "Completed worker shard start=%d stop=%d step=%d"
            % (args.trial_start, trial_stop, args.trial_step)
        )
        return

    trial_rows = rebuild_tables(search_dir, args.num_samples)
    completed = [row for row in trial_rows if row.get("status") == "complete"]
    if not completed:
        raise RuntimeError("No hyperparameter trial completed successfully")
    completed.sort(
        key=lambda row: (float(row["mean_auroc"]), float(row["mean_auprc"])),
        reverse=True,
    )
    best = completed[0]
    best_trial = int(best["trial"])
    best_parameters = read_json(search_dir / ("trial_%04d" % best_trial) / "params.json")
    best_cfg = apply_parameters(base_cfg, best_parameters)
    output_config = Path(args.output_config or ("configs/%s.yaml" % args.data))
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(best_cfg.dump(), encoding="utf-8")
    write_json(
        {
            "best_trial": best_trial,
            "mean_auroc": float(best["mean_auroc"]),
            "mean_auprc": float(best["mean_auprc"]),
            "parameters": best_parameters,
            "output_config": str(output_config),
        },
        search_dir / "selection.json",
    )
    print(
        "\nSelected trial %04d: mean AUROC %.4f, mean AUPRC %.4f\nSaved %s"
        % (best_trial, best["mean_auroc"], best["mean_auprc"], output_config)
    )


if __name__ == "__main__":
    main()
