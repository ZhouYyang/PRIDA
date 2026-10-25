import argparse
import json
from pathlib import Path

from experiment import load_config, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run one homology-constrained PRIDA fold")
    parser.add_argument("--data", required=True, help="RPI488, RPI1807, RPI369 or Aptamer")
    parser.add_argument("--fold", type=int, default=1, help="Domain fold (1-5)")
    parser.add_argument("--seed", type=int, default=42, help="Downstream training seed")
    parser.add_argument(
        "--feature-seed",
        type=int,
        default=None,
        help="Projection pretraining seed; defaults to --seed",
    )
    parser.add_argument("--config", help="YAML configuration; defaults to configs/<data>.yaml if present")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--run-root", default="runs/final")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--threshold-rule",
        choices=["f1_optimal", "fixed"],
        default=None,
        help="Target-test decision threshold; defaults to the configuration value.",
    )
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    default_config = Path("configs") / (args.data + ".yaml")
    config_path = args.config or (str(default_config) if default_config.is_file() else None)
    cfg = load_config(config_path)
    cfg.SEED = args.seed
    if args.max_epochs is not None:
        cfg.SOLVER.MAX_EPOCH = args.max_epochs
    if args.threshold_rule:
        cfg.RESULT.THRESHOLD_RULE = args.threshold_rule
    if args.save_model:
        cfg.RESULT.SAVE_MODEL = True

    feature_seed = args.seed if args.feature_seed is None else args.feature_seed
    run_dir = (
        Path(args.run_root)
        / args.data
        / ("fold_%d" % args.fold)
        / ("seed_%d" % args.seed)
    )
    metrics = run_experiment(
        cfg=cfg,
        dataset=args.data,
        fold=args.fold,
        seed=args.seed,
        data_root=args.data_root,
        feature_root=args.feature_root,
        feature_seed=feature_seed,
        run_dir=run_dir,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
