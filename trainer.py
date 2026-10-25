import copy
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm

from domain_adaptator import ReverseLayerF
from models import RandomLayer, cross_entropy_logits, entropy_logits
from utils import write_json


class Trainer(object):
    """Train one dataset/fold/seed run.

    Model selection uses Source training loss only. Target labels are consumed
    once after training by :meth:`test`; target_train batches contain no label.
    """

    def __init__(
        self,
        data,
        model,
        optim,
        device,
        train_dataloader,
        test_dataloader,
        opt_da,
        discriminator,
        fold=1,
        seed=42,
        run_dir=None,
        alpha=1.0,
        **config
    ):
        self.data = data
        self.fold = int(fold)
        self.seed = int(seed)
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = int(config["SOLVER"]["MAX_EPOCH"])
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = float(alpha)
        self.n_class = int(config["DECODER"]["BINARY"])
        self.da_init_epoch = int(config["DA"]["INIT_EPOCH"])
        self.lambda_da = float(config["DA"]["LAMB_DA"])
        self.use_da_entropy = bool(config["DA"]["USE_ENTROPY"])
        self.threshold = float(config["RESULT"].get("THRESHOLD", 0.5))
        self.threshold_rule = str(config["RESULT"].get("THRESHOLD_RULE", "f1_optimal"))
        if self.threshold_rule not in {"f1_optimal", "fixed"}:
            raise ValueError("RESULT.THRESHOLD_RULE must be 'f1_optimal' or 'fixed'")
        self.save_model = bool(config["RESULT"].get("SAVE_MODEL", False))
        self.run_dir = Path(run_dir or config["RESULT"]["OUTPUT_DIR"])
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if opt_da is None or discriminator is None:
            raise ValueError("CDAN requires a discriminator and its optimizer")
        self.optim_da = opt_da
        self.domain_dmm = discriminator
        self.random_layer = RandomLayer(
            [int(config["DECODER"]["IN_DIM"]), self.n_class],
            int(config["DA"]["RANDOM_DIM"]),
        ).to(device)

        self.best_epoch = 0
        self.best_loss = float("inf")
        self.history = []

    @staticmethod
    def _f1_optimal_threshold(y_true, y_score):
        """Threshold on the Target-test scores that maximises F1.

        Uses the ``score >= threshold`` convention of :meth:`test`. Ties resolve
        to the smallest qualifying threshold.
        """
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_score, pos_label=1
        )
        precision, recall = precision[:-1], recall[:-1]
        denominator = precision + recall
        f1_scores = np.divide(
            2.0 * precision * recall,
            denominator,
            out=np.zeros(len(thresholds), dtype=float),
            where=denominator > 0,
        )
        return float(thresholds[int(np.argmax(f1_scores))])

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        return 1.0 + torch.exp(-entropy)

    def _domain_logits(self, features, class_logits):
        """Condition the reversed features on the class posterior (CDAN)."""
        reverse_features = ReverseLayerF.apply(features, self.alpha)
        probabilities = torch.softmax(class_logits, dim=1).detach()
        conditioned = self.random_layer([reverse_features, probabilities])
        return self.domain_dmm(conditioned)

    def train(self):
        patience = 5
        epochs_without_improvement = 0
        selection_start = self.da_init_epoch
        best_state = copy.deepcopy(self.model.state_dict())

        for _ in range(self.epochs):
            self.current_epoch += 1
            train_loss, model_loss, da_loss, lambda_da = self.train_da_epoch()

            row = {
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "model_loss": model_loss,
                "da_loss": da_loss,
                "lambda_da": lambda_da,
                "selection_loss": model_loss,
            }
            self.history.append(row)

            if self.current_epoch >= selection_start:
                if model_loss < self.best_loss:
                    self.best_loss = model_loss
                    self.best_epoch = self.current_epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        if self.best_epoch == 0:
            self.best_epoch = self.current_epoch
            self.best_loss = self.history[-1]["selection_loss"]
            best_state = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        pd.DataFrame(self.history).to_csv(self.run_dir / "train_history.csv", index=False)
        if self.save_model:
            torch.save(
                {
                    "model": best_state,
                    "dataset": self.data,
                    "fold": self.fold,
                    "seed": self.seed,
                    "best_epoch": self.best_epoch,
                    "best_source_classification_loss": self.best_loss,
                },
                self.run_dir / "best_model.pt",
            )

        metrics = self.test()
        metrics.update(
            {
                "dataset": self.data,
                "fold": self.fold,
                "seed": self.seed,
                "best_epoch": self.best_epoch,
                "best_source_classification_loss": self.best_loss,
                "target_labels_used_for_training": False,
                "selection_criterion": "source_classification_loss",
            }
        )
        write_json(metrics, self.run_dir / "metrics.json")
        print(
            "Test fold %d: AUROC %.4f AUPRC %.4f (best epoch %d)"
            % (self.fold, metrics["auroc"], metrics["auprc"], self.best_epoch)
        )
        return metrics

    def train_da_epoch(self):
        self.model.train()
        self.domain_dmm.train()
        total_loss = 0.0
        total_model_loss = 0.0
        total_da_loss = 0.0
        total_source_examples = 0
        da_active = self.current_epoch >= self.da_init_epoch
        applied_lambda = self.lambda_da if da_active else 0.0

        for batch_source, batch_target in tqdm(
            self.train_dataloader, desc="Source + unlabelled Target", leave=False
        ):
            rna, protein, labels = batch_source
            rna_target, protein_target = batch_target
            rna = rna.to(self.device)
            protein = protein.to(self.device)
            labels = labels.long().to(self.device)
            rna_target = rna_target.to(self.device)
            protein_target = protein_target.to(self.device)

            self.optim.zero_grad()
            if da_active:
                self.optim_da.zero_grad()

            _, _, source_features, source_scores, _ = self.model(rna, protein)
            _, model_loss = cross_entropy_logits(source_scores, labels)
            da_loss = torch.zeros((), device=self.device)

            if da_active:
                _, _, target_features, target_scores, _ = self.model(rna_target, protein_target)
                source_domain_scores = self._domain_logits(source_features, source_scores)
                target_domain_scores = self._domain_logits(target_features, target_scores)

                if self.use_da_entropy:
                    source_weight = self._compute_entropy_weights(source_scores)
                    source_weight = source_weight / source_weight.sum()
                    target_weight = self._compute_entropy_weights(target_scores)
                    target_weight = target_weight / target_weight.sum()
                else:
                    source_weight = None
                    target_weight = None

                source_domain_labels = torch.ones(
                    source_domain_scores.size(0), dtype=torch.long, device=self.device
                )
                target_domain_labels = torch.zeros(
                    target_domain_scores.size(0), dtype=torch.long, device=self.device
                )
                _, source_da_loss = cross_entropy_logits(
                    source_domain_scores, source_domain_labels, source_weight
                )
                _, target_da_loss = cross_entropy_logits(
                    target_domain_scores, target_domain_labels, target_weight
                )
                da_loss = source_da_loss + target_da_loss

            loss = model_loss + applied_lambda * da_loss
            loss.backward()
            self.optim.step()
            if da_active:
                self.optim_da.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_model_loss += model_loss.item() * batch_size
            total_da_loss += da_loss.item() * batch_size
            total_source_examples += batch_size

        if total_source_examples == 0:
            raise RuntimeError("Domain-adaptation DataLoader yielded no Source examples")
        averages = (
            total_loss / total_source_examples,
            total_model_loss / total_source_examples,
            total_da_loss / total_source_examples,
            applied_lambda,
        )
        print(
            "Epoch %d total %.6f source %.6f domain %.6f lambda %.4f"
            % ((self.current_epoch,) + averages)
        )
        return averages

    def test(self):
        self.model.eval()
        y_true = []
        y_score = []
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for rna, protein, labels in self.test_dataloader:
                rna = rna.to(self.device)
                protein = protein.to(self.device)
                labels = labels.long().to(self.device)
                _, _, _, scores, _ = self.model(rna, protein)
                probabilities, loss = cross_entropy_logits(scores, labels)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size
                y_true.extend(labels.cpu().tolist())
                y_score.extend(probabilities.cpu().tolist())

        if total_examples == 0:
            raise RuntimeError("Target test DataLoader yielded no examples")
        y_true_array = np.asarray(y_true, dtype=int)
        y_score_array = np.asarray(y_score, dtype=float)
        if np.unique(y_true_array).size != 2:
            raise ValueError("Target test must contain both classes to calculate AUROC")

        fpr, tpr, _ = roc_curve(y_true_array, y_score_array)
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_array, y_score_array, pos_label=1
        )
        if self.threshold_rule == "f1_optimal":
            threshold = self._f1_optimal_threshold(y_true_array, y_score_array)
        else:
            threshold = self.threshold
        predictions = (y_score_array >= threshold).astype(int)
        matrix = confusion_matrix(y_true_array, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        precision = precision_score(y_true_array, predictions, zero_division=0)
        recall = recall_score(y_true_array, predictions, zero_division=0)

        metrics = {
            "auroc": float(auc(fpr, tpr)),
            "auprc": float(auc(recall_curve, precision_curve)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "accuracy": float((tp + tn) / matrix.sum()),
            "test_loss": float(total_loss / total_examples),
            "precision": float(precision),
            "mcc": float(matthews_corrcoef(y_true_array, predictions)),
            "f1": float(2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0,
            "recall": float(recall),
            "classification_threshold": float(threshold),
            "threshold_rule": self.threshold_rule,
            "n_target_test": int(total_examples),
        }

        pd.DataFrame(
            {
                "row_index": np.arange(total_examples),
                "y_true": y_true_array,
                "y_score": y_score_array,
                "y_pred": predictions,
            }
        ).to_csv(self.run_dir / "target_predictions.csv", index=False)
        return metrics
