import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from pre_train.common import artifact_dir, feature_dir, load_fold_sequences, sha256_file
from pre_train.train import ESMEncoder, ProjectionHead, masked_mean_pool
from utils import write_json


def extract_protein_features(
    dataset,
    fold,
    seed,
    data_root="datasets",
    artifact_root="artifacts/pretrain",
    feature_root="features",
    esm2_path="checkpoints/ESM2",
    checkpoint_path=None,
    device="cuda:0",
    force=False,
):
    device_object = torch.device(device)
    if device_object.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for protein extraction but is unavailable")
        torch.cuda.set_device(0 if device_object.index is None else device_object.index)

    _, _, protein_records, _ = load_fold_sequences(data_root, dataset, fold)
    if checkpoint_path is None:
        checkpoint_path = artifact_dir(artifact_root, dataset, fold, seed) / "best_pretrain_proj.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Projection checkpoint does not exist: %s" % checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=device_object)
    projection_state = checkpoint["prot_proj"]
    projection_dim = int(projection_state["proj.0.weight"].shape[0])

    encoder = ESMEncoder(esm2_path).to(device_object).eval()
    projection = ProjectionHead(encoder.output_dim, projection_dim).to(device_object).eval()
    projection.load_state_dict(projection_state, strict=True)
    output_dir = feature_dir(feature_root, dataset, fold, seed) / "Protein"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    reused = 0
    with torch.no_grad():
        for sequence_id in tqdm(sorted(protein_records), desc="Projected ESM-2 features"):
            output_path = output_dir / (sequence_id + ".pt")
            if output_path.is_file() and not force:
                reused += 1
                continue
            sequence = protein_records[sequence_id]
            with autocast(enabled=device_object.type == "cuda"):
                tokens, mask = encoder([sequence])
                projected = projection(tokens)
                embedding = masked_mean_pool(projected, mask).squeeze(0).float().cpu()
            if embedding.ndim != 1 or embedding.numel() != projection_dim:
                raise RuntimeError("Unexpected protein feature shape for %s" % sequence_id)
            torch.save(
                {
                    "sequence_id": sequence_id,
                    "sequence": sequence,
                    "mean_representations": embedding,
                    "modality": "protein",
                    "dataset": dataset,
                    "fold": int(fold),
                    "projection_seed": int(seed),
                },
                output_path,
            )
            written += 1

    metadata = {
        "dataset": dataset,
        "fold": int(fold),
        "seed": int(seed),
        "modality": "protein",
        "n_required": len(protein_records),
        "n_written": written,
        "n_reused": reused,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "base_model": str(Path(esm2_path).resolve()),
        "projection_dim": projection_dim,
        "pooling": "mean_over_attention_mask_nonpadding_tokens",
    }
    write_json(metadata, output_dir / "extraction_metadata.json")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Extract fold-specific projected protein features")
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--artifact-root", default="artifacts/pretrain")
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--esm2-path", default="checkpoints/ESM2")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = extract_protein_features(
        dataset=args.data,
        fold=args.fold,
        seed=args.seed,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        feature_root=args.feature_root,
        esm2_path=args.esm2_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        force=args.force,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
