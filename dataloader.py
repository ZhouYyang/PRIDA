from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data


class ProjectedFeatureDataset(data.Dataset):
    """Load fold-specific projected features for an interaction table.

    Target-adaptation datasets are created with ``labeled=False`` so their labels
    never enter the training batch, even though the construction CSV retains Y
    for auditing and stratified target_train/target_test construction.
    """

    def __init__(self, df, feature_dir, labeled=True, validate=True):
        self.df = df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.labeled = labeled
        self.rna_dir = self.feature_dir / "RNA"
        self.protein_dir = self.feature_dir / "Protein"

        required = {"RNA", "Protein"}
        if labeled:
            required.add("Y")
        missing_columns = required.difference(self.df.columns)
        if missing_columns:
            raise ValueError("Missing dataset columns: %s" % sorted(missing_columns))

        if validate:
            self._validate_feature_files()

    def _validate_feature_files(self):
        missing = []
        for column, directory in (("RNA", self.rna_dir), ("Protein", self.protein_dir)):
            for sequence_id in self.df[column].astype(str).unique():
                path = directory / (sequence_id + ".pt")
                if not path.is_file():
                    missing.append(str(path))
                    if len(missing) >= 10:
                        break
        if missing:
            raise FileNotFoundError(
                "Fold-specific projected features are missing. First examples:\n  "
                + "\n  ".join(missing)
            )

    @staticmethod
    def _load_feature(path):
        payload = torch.load(str(path), map_location="cpu")
        if "mean_representations" not in payload:
            raise KeyError("%s has no 'mean_representations' tensor" % path)
        feature = payload["mean_representations"]
        if not torch.is_tensor(feature):
            feature = torch.as_tensor(feature)
        return feature.float()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        rna_id = str(row["RNA"])
        protein_id = str(row["Protein"])
        rna = self._load_feature(self.rna_dir / (rna_id + ".pt"))
        protein = self._load_feature(self.protein_dir / (protein_id + ".pt"))
        if not self.labeled:
            return rna, protein
        return rna, protein, int(row["Y"])


class MultiDataLoader(object):
    """Cycle shorter domain loader so each epoch sees the longer loader once."""

    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = int(np.maximum(1, n_batches))

    def __iter__(self):
        iterators = [iter(loader) for loader in self._dataloaders]
        for _ in range(self._n_batches):
            batches = []
            for index, loader in enumerate(self._dataloaders):
                try:
                    batch = next(iterators[index])
                except StopIteration:
                    iterators[index] = iter(loader)
                    batch = next(iterators[index])
                batches.append(batch)
            yield batches

    def __len__(self):
        return self._n_batches
