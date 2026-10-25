import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import multimolecule  # Registers RiNALMo with transformers AutoModel/AutoTokenizer.
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.functional import normalize
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoModel,
    AutoTokenizer,
    EsmModel,
    EsmTokenizer,
    get_linear_schedule_with_warmup,
)

from pre_train.common import artifact_dir, sha256_file, source_sequence_table
from utils import set_seed, write_json


class SourceSequenceDataset(Dataset):
    def __init__(self, source_table):
        self.table = source_table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        return (
            row["Protein_sequence"],
            row["RNA_sequence"],
            torch.tensor(float(row["Y"]), dtype=torch.float32),
        )


class ESMEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = EsmModel.from_pretrained(
            model_path, local_files_only=True, add_pooling_layer=False
        )
        self.output_dim = int(self.model.config.hidden_size)
        self.max_length = int(self.model.config.max_position_embeddings)

    def forward(self, sequences):
        inputs = self.tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, inputs["attention_mask"]


class RNAEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, add_pooling_layer=False
        )
        if self.model.config.model_type != "rinalmo":
            raise ValueError(
                "Expected a RiNALMo checkpoint, found model_type=%s"
                % self.model.config.model_type
            )
        self.output_dim = int(self.model.config.hidden_size)
        self.max_length = int(self.model.config.max_position_embeddings)

    def forward(self, sequences):
        inputs = self.tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, inputs["attention_mask"]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, token_embeddings):
        return normalize(self.proj(token_embeddings), dim=-1)


def masked_mean_pool(token_embeddings, attention_mask):
    """Mean-pool non-padding tokens; unlike the old code, padding has zero weight."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class ProjectionTrainer(object):
    def __init__(
        self,
        source_table,
        output_dir,
        esm2_path,
        rinalmo_path,
        seed=42,
        batch_size=4,
        learning_rate=2e-5,
        projection_dim=512,
        validation_fraction=0.2,
        patience=5,
        warmup_ratio=0.1,
        num_workers=0,
        device="cuda:0",
    ):
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for projection pretraining but is unavailable")
            torch.cuda.set_device(0 if self.device.index is None else self.device.index)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = int(patience)
        self.warmup_ratio = float(warmup_ratio)

        print("Loading frozen ESM-2 and RiNALMo encoders")
        self.protein_encoder = ESMEncoder(esm2_path).to(self.device)
        self.rna_encoder = RNAEncoder(rinalmo_path).to(self.device)
        self.protein_encoder.eval()
        self.rna_encoder.eval()
        for parameter in self.protein_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.rna_encoder.parameters():
            parameter.requires_grad = False

        self.protein_projection = ProjectionHead(
            self.protein_encoder.output_dim, projection_dim
        ).to(self.device)
        self.rna_projection = ProjectionHead(self.rna_encoder.output_dim, projection_dim).to(
            self.device
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(
            list(self.protein_projection.parameters())
            + list(self.rna_projection.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

        dataset = SourceSequenceDataset(source_table)
        labels = source_table["Y"].astype(int).to_numpy()
        indices = np.arange(len(dataset))
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=self.seed,
            shuffle=True,
            stratify=labels,
        )
        generator = torch.Generator().manual_seed(self.seed)
        self.train_loader = DataLoader(
            Subset(dataset, train_indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        self.validation_loader = DataLoader(
            Subset(dataset, validation_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.split_metadata = {
            "n_source_pairs": len(dataset),
            "n_projection_train": len(train_indices),
            "n_projection_validation": len(validation_indices),
            "projection_train_indices": train_indices.tolist(),
            "projection_validation_indices": validation_indices.tolist(),
        }

    def _forward_batch(self, protein_sequences, rna_sequences):
        with torch.no_grad(), autocast(enabled=self.device.type == "cuda"):
            protein_tokens, protein_mask = self.protein_encoder(protein_sequences)
            rna_tokens, rna_mask = self.rna_encoder(rna_sequences)
        with autocast(enabled=self.device.type == "cuda"):
            protein_projected = self.protein_projection(protein_tokens)
            rna_projected = self.rna_projection(rna_tokens)
            protein_features = masked_mean_pool(protein_projected, protein_mask)
            rna_features = masked_mean_pool(rna_projected, rna_mask)
            logits = torch.sum(protein_features * rna_features, dim=-1)
        return logits

    def validate(self):
        self.protein_projection.eval()
        self.rna_projection.eval()
        labels_all = []
        scores_all = []
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for protein_sequences, rna_sequences, labels in self.validation_loader:
                labels = labels.to(self.device)
                logits = self._forward_batch(protein_sequences, rna_sequences)
                loss = self.criterion(logits.float(), labels)
                probabilities = torch.sigmoid(logits)
                total_loss += loss.item() * labels.size(0)
                total_examples += labels.size(0)
                labels_all.extend(labels.cpu().tolist())
                scores_all.extend(probabilities.float().cpu().tolist())
        return {
            "val_loss": total_loss / total_examples,
            "auroc": float(roc_auc_score(labels_all, scores_all)),
            "auprc": float(average_precision_score(labels_all, scores_all)),
        }

    def train(self, epochs=30, metadata=None):
        total_steps = len(self.train_loader) * int(epochs)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(total_steps * self.warmup_ratio), total_steps
        )
        best_auroc = -float("inf")
        patience_counter = 0
        history = []
        checkpoint_path = self.output_dir / "best_pretrain_proj.pt"

        for epoch in range(1, int(epochs) + 1):
            self.protein_projection.train()
            self.rna_projection.train()
            total_loss = 0.0
            total_examples = 0
            for protein_sequences, rna_sequences, labels in self.train_loader:
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self._forward_batch(protein_sequences, rna_sequences)
                loss = self.criterion(logits.float(), labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()
                total_loss += loss.item() * labels.size(0)
                total_examples += labels.size(0)

            validation = self.validate()
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total_examples,
                **validation,
            }
            history.append(row)
            print(
                "Projection epoch %d train %.6f val %.6f AUROC %.4f AUPRC %.4f"
                % (
                    epoch,
                    row["train_loss"],
                    row["val_loss"],
                    row["auroc"],
                    row["auprc"],
                )
            )

            if validation["auroc"] > best_auroc:
                best_auroc = validation["auroc"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "prot_proj": self.protein_projection.state_dict(),
                        "rna_proj": self.rna_projection.state_dict(),
                        "metrics": validation,
                        "metadata": metadata or {},
                        "pooling": "mean_over_attention_mask_nonpadding_tokens",
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        pd.DataFrame(history).to_csv(self.output_dir / "pretrain_history.csv", index=False)
        return checkpoint_path


def train_projection_for_fold(
    dataset,
    fold,
    seed,
    data_root="datasets",
    artifact_root="artifacts/pretrain",
    esm2_path="checkpoints/ESM2",
    rinalmo_path="checkpoints/RiNALMo",
    device="cuda:0",
    batch_size=4,
    learning_rate=2e-5,
    projection_dim=512,
    validation_fraction=0.2,
    patience=5,
    epochs=30,
    num_workers=0,
    force=False,
):
    set_seed(seed)
    split_dir, source_table = source_sequence_table(data_root, dataset, fold)
    output_dir = artifact_dir(artifact_root, dataset, fold, seed)
    checkpoint_path = output_dir / "best_pretrain_proj.pt"
    if checkpoint_path.is_file() and not force:
        print("Reuse projection checkpoint %s" % checkpoint_path)
        return checkpoint_path

    source_csv = split_dir / "source_train.csv"
    metadata = {
        "dataset": dataset,
        "fold": int(fold),
        "seed": int(seed),
        "source_csv": str(source_csv.resolve()),
        "source_csv_sha256": sha256_file(source_csv),
        "target_rows_used": 0,
        "target_labels_used": False,
        "esm2_path": str(Path(esm2_path).resolve()),
        "rinalmo_path": str(Path(rinalmo_path).resolve()),
        "projection_dim": int(projection_dim),
    }
    trainer = ProjectionTrainer(
        source_table=source_table,
        output_dir=output_dir,
        esm2_path=esm2_path,
        rinalmo_path=rinalmo_path,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        projection_dim=projection_dim,
        validation_fraction=validation_fraction,
        patience=patience,
        num_workers=num_workers,
        device=device,
    )
    metadata.update(trainer.split_metadata)
    checkpoint_path = trainer.train(epochs=epochs, metadata=metadata)
    write_json(metadata, output_dir / "pretrain_metadata.json")
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="Source-only projection pretraining for one fold")
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--artifact-root", default="artifacts/pretrain")
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = train_projection_for_fold(
        dataset=args.data,
        fold=args.fold,
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
        force=args.force,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
