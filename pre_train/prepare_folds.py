import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pre_train.common import feature_dir, load_fold_sequences, sha256_file
from pre_train.extract_protein import extract_protein_features
from pre_train.extract_rna import extract_rna_features
from pre_train.train import train_projection_for_fold
from utils import write_json


def validate_feature_cache(data_root, feature_root, dataset, fold, seed):
    _, _, protein_records, rna_records = load_fold_sequences(data_root, dataset, fold)
    root = feature_dir(feature_root, dataset, fold, seed)
    missing = []
    for modality, records in (("Protein", protein_records), ("RNA", rna_records)):
        for sequence_id in records:
            path = root / modality / (sequence_id + ".pt")
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 20:
                    break
    if missing:
        raise FileNotFoundError("Incomplete feature cache:\n  " + "\n  ".join(missing))
    return {
        "protein_features": len(protein_records),
        "rna_features": len(rna_records),
        "feature_dir": str(root.resolve()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Source-only pretraining and projected-feature extraction for domain folds"
    )
    parser.add_argument("--data", required=True, choices=["RPI488", "RPI1807", "RPI369", "Aptamer"])
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--artifact-root", default="artifacts/pretrain")
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--esm2-path", default="checkpoints/ESM2")
    parser.add_argument("--rinalmo-path", default="checkpoints/RiNALMo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-pretrain", action="store_true")
    parser.add_argument("--force-extraction", action="store_true")
    return parser.parse_args()


def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    for fold in args.folds:
        print("\n=== Prepare %s fold_%d seed_%d ===" % (args.data, fold, args.seed))
        checkpoint = train_projection_for_fold(
            dataset=args.data,
            fold=fold,
            seed=args.seed,
            data_root=args.data_root,
            artifact_root=args.artifact_root,
            esm2_path=args.esm2_path,
            rinalmo_path=args.rinalmo_path,
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            projection_dim=args.projection_dim,
            validation_fraction=args.validation_fraction,
            patience=args.patience,
            epochs=args.epochs,
            num_workers=args.num_workers,
            force=args.force_pretrain,
        )
        release_cuda()
        extract_protein_features(
            dataset=args.data,
            fold=fold,
            seed=args.seed,
            data_root=args.data_root,
            artifact_root=args.artifact_root,
            feature_root=args.feature_root,
            esm2_path=args.esm2_path,
            checkpoint_path=checkpoint,
            device=args.device,
            force=args.force_extraction,
        )
        release_cuda()
        extract_rna_features(
            dataset=args.data,
            fold=fold,
            seed=args.seed,
            data_root=args.data_root,
            artifact_root=args.artifact_root,
            feature_root=args.feature_root,
            rinalmo_path=args.rinalmo_path,
            checkpoint_path=checkpoint,
            device=args.device,
            force=args.force_extraction,
        )
        release_cuda()
        validation = validate_feature_cache(
            args.data_root, args.feature_root, args.data, fold, args.seed
        )
        validation.update(
            {
                "dataset": args.data,
                "fold": int(fold),
                "seed": int(args.seed),
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_only_projection_pretraining": True,
                "target_labels_used": False,
            }
        )
        write_json(
            validation,
            feature_dir(args.feature_root, args.data, fold, args.seed)
            / "cache_validation.json",
        )
        print("Validated %s" % validation["feature_dir"])


if __name__ == "__main__":
    main()
