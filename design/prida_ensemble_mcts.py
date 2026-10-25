"""Generate 20-nt aptamers using the mean score of five PRIDA folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import load_config  # noqa: E402
from models import PRIDA  # noqa: E402
from mcts_v2 import CachedBatchScorer, CorrectedMCTS  # noqa: E402
from pre_train.train import ProjectionHead, RNAEncoder, masked_mean_pool  # noqa: E402


def parse_fasta(path):
    records, key, chunks = [], None, []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if key is not None:
                records.append((key, "".join(chunks)))
            key, chunks = line[1:].split()[0], []
        else:
            chunks.append(line)
    if key is not None:
        records.append((key, "".join(chunks)))
    return records


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class FiveFoldReward:
    def __init__(self, device, config_path, weight_root, artifact_root, feature_root,
                 target_sequence, sequence_to_id, rna_encoder):
        self.device = device
        self.rna_encoder = rna_encoder
        self.models, self.rna_projections, self.protein_features = [], [], []
        cfg = load_config(config_path)
        protein_id = sequence_to_id[target_sequence]
        for fold in range(1, 6):
            checkpoint_path = Path(weight_root) / "Aptamer" / f"fold_{fold}/seed_42/best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = PRIDA(**cfg).to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            projection_checkpoint = torch.load(
                Path(artifact_root) / f"Aptamer/fold_{fold}/seed_42/best_pretrain_proj.pt",
                map_location=device,
            )
            projection = ProjectionHead(rna_encoder.output_dim, 512).to(device)
            projection.load_state_dict(projection_checkpoint["rna_proj"], strict=True)
            projection.eval()
            for parameter in projection.parameters():
                parameter.requires_grad = False
            protein_payload = torch.load(
                Path(feature_root) / f"Aptamer/fold_{fold}/seed_42/Protein/{protein_id}.pt",
                map_location="cpu",
            )
            protein_feature = protein_payload["mean_representations"].float().to(device).unsqueeze(0)
            self.models.append(model)
            self.rna_projections.append(projection)
            self.protein_features.append(protein_feature)
        self.last_fold_scores = None

    def raw_score_many_with_folds(self, sequences):
        with torch.inference_mode(), autocast(enabled=self.device.type == "cuda"):
            tokens, mask = self.rna_encoder(list(sequences))
            fold_scores = []
            for model, projection, protein in zip(
                self.models, self.rna_projections, self.protein_features
            ):
                rna = masked_mean_pool(projection(tokens), mask).float()
                pro = protein.expand(rna.size(0), -1)
                _, _, _, logits, _ = model(rna, pro)
                fold_scores.append(F.softmax(logits, dim=1)[:, 1])
            matrix = torch.stack(fold_scores, dim=1)
        self.last_fold_scores = matrix.detach().cpu().numpy()
        return matrix.mean(dim=1).detach().cpu().tolist(), self.last_fold_scores

    def raw_score_many(self, sequences):
        return self.raw_score_many_with_folds(sequences)[0]

    def scorer(self, batch_size):
        return CachedBatchScorer(self.raw_score_many, batch_size=batch_size,
                                 use_cache=True, group_key=len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target-fasta",
        default=str(ROOT / "datasets/Aptamer/target_proteins.fasta"),
        help="Protein FASTA holding the design targets",
    )
    parser.add_argument(
        "--dataset-fasta",
        default=str(ROOT / "datasets/Aptamer/target_proteins.fasta"),
        help="Aptamer Target FASTA used to map a target sequence back to its ID",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/Aptamer.yaml"),
        help="Shared PRIDA configuration of the five folds",
    )
    parser.add_argument(
        "--weight-root",
        required=True,
        help="Run root holding Aptamer/fold_<k>/seed_42/best_model.pt for the five splits",
    )
    parser.add_argument("--artifact-root", default=str(ROOT / "artifacts/pretrain"))
    parser.add_argument("--feature-root", default=str(ROOT / "features"))
    parser.add_argument(
        "--rinalmo-path",
        default=str(ROOT / "checkpoints/RiNALMo"),
        help="Frozen RiNALMo checkpoint directory",
    )
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--search-budget", type=int, default=300)
    parser.add_argument("--num-candidates", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--leaf-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    target_fasta = Path(args.target_fasta)
    dataset_fasta = Path(args.dataset_fasta)
    targets = list(enumerate(parse_fasta(target_fasta)))
    if args.target_limit:
        targets = targets[:args.target_limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    targets = [item for item in targets if item[0] % args.shard_count == args.shard_index]
    sequence_to_id = {}
    for identifier, sequence in parse_fasta(dataset_fasta):
        sequence_to_id.setdefault(sequence, identifier)
    missing = [identifier for _, (identifier, sequence) in targets if sequence not in sequence_to_id]
    if missing:
        raise RuntimeError("Targets absent from Aptamer target FASTA: " + ",".join(missing))

    config_path = Path(args.config)
    weight_root = Path(args.weight_root)
    artifact_root = Path(args.artifact_root)
    feature_root = Path(args.feature_root)
    rna_encoder = RNAEncoder(args.rinalmo_path).to(device).eval()
    for parameter in rna_encoder.parameters():
        parameter.requires_grad = False

    all_path, selected_path = output / "all_candidates.csv", output / "selected_top_k.csv"
    all_rows = list(csv.DictReader(all_path.open())) if all_path.exists() else []
    selected_rows = list(csv.DictReader(selected_path.open())) if selected_path.exists() else []
    completed = {row["target_id"] for row in selected_rows}

    for shard_position, (target_index, (target_id, target_sequence)) in enumerate(targets):
        if target_id in completed:
            print(f"[{shard_position+1}/{len(targets)}] resume {target_id}", flush=True)
            continue
        print(f"[{shard_position+1}/{len(targets)}] target={target_id} original_index={target_index}", flush=True)
        reward = FiveFoldReward(device, config_path, weight_root, artifact_root,
                                feature_root, target_sequence, sequence_to_id, rna_encoder)
        scorer = reward.scorer(args.batch_size)
        target_rows, unique = [], set()
        for attempt in range(args.num_candidates * 3):
            if len(unique) >= args.num_candidates:
                break
            seed = args.seed + target_index * 10000 + attempt
            start = time.time()
            result = CorrectedMCTS(
                scorer, sequence_length=20, final_policy="visits",
                batch_size=args.leaf_batch_size, seed=seed,
            ).search(args.search_budget)
            mean_scores, fold_matrix = reward.raw_score_many_with_folds([result.candidate])
            row = {
                "target_id": target_id, "target_length": len(target_sequence),
                "attempt": attempt, "seed": seed, "candidate": result.candidate,
                "ensemble_mean_score": mean_scores[0],
                **{f"fold_{i+1}_score": float(fold_matrix[0, i]) for i in range(5)},
                "search_budget_requests": result.score_requests,
                "wall_seconds": time.time() - start,
                "duplicate_candidate": result.candidate in unique,
            }
            unique.add(result.candidate); target_rows.append(row); all_rows.append(row)
        best = {}
        for row in target_rows:
            sequence = row["candidate"]
            if sequence not in best or float(row["ensemble_mean_score"]) > float(best[sequence]["ensemble_mean_score"]):
                best[sequence] = row
        ranked = sorted(best.values(), key=lambda row: float(row["ensemble_mean_score"]), reverse=True)[:args.top_k]
        if len(ranked) != args.top_k:
            raise RuntimeError(f"{target_id}: only {len(ranked)} unique candidates")
        with (output / f"{target_id}_sorted.txt").open("w") as handle:
            for rank, row in enumerate(ranked, 1):
                selected = dict(row); selected["rank"] = rank; selected_rows.append(selected)
                handle.write(f"{row['candidate']}\t{float(row['ensemble_mean_score']):.10f}\n")
        write_csv(all_path, all_rows); write_csv(selected_path, selected_rows)
        del reward; torch.cuda.empty_cache()

    manifest = {
        "target_count": len(targets), "top_k": args.top_k,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "sequence_length": 20, "ensemble": "arithmetic mean of five PRIDA fold probabilities",
        "trial": 261, "seed": args.seed, "search_budget": args.search_budget,
        "config_sha256": sha256(config_path),
        "weight_sha256": [sha256(weight_root / f"Aptamer/fold_{i}/seed_42/best_model.pt") for i in range(1, 6)],
        "target_fasta_sha256": sha256(target_fasta),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
