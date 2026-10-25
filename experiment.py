from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import MultiDataLoader, ProjectedFeatureDataset
from domain_adaptator import Discriminator
from models import PRIDA
from trainer import Trainer
from utils import set_seed, write_json


def load_config(config_path=None):
    cfg = get_cfg_defaults()
    if config_path:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError("Configuration file does not exist: %s" % config_path)
        cfg.merge_from_file(str(config_path))
    return cfg


def resolve_split_dir(data_root, dataset, fold):
    dataset_dir = Path(data_root) / dataset
    fold_dir = dataset_dir / "folds" / ("fold_%d" % int(fold))
    if fold_dir.is_dir():
        return fold_dir
    if int(fold) != 1:
        raise FileNotFoundError("%s has no fold_%d" % (dataset, int(fold)))
    required = [dataset_dir / name for name in ("source_train.csv", "target_train.csv", "target_test.csv")]
    if all(path.is_file() for path in required):
        return dataset_dir
    raise FileNotFoundError("Cannot resolve split directory for %s fold %s" % (dataset, fold))


def resolve_feature_dir(feature_root, dataset, fold, feature_seed):
    return Path(feature_root) / dataset / ("fold_%d" % int(fold)) / ("seed_%d" % int(feature_seed))


def resolve_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else device.index
        torch.cuda.set_device(index)
    return device


def validate_config(cfg):
    attention_heads = int(cfg.ATTN.HEADS)
    for branch in ("RNA", "PROTEIN"):
        filters = list(cfg[branch].NUM_FILTERS)
        if len(filters) != 3:
            raise ValueError("%s.NUM_FILTERS must contain exactly three values" % branch)
        if filters[-1] % attention_heads != 0:
            raise ValueError(
                "%s.NUM_FILTERS[-1]=%d is not divisible by ATTN.HEADS=%d"
                % (branch, filters[-1], attention_heads)
            )
    if int(cfg.SOLVER.BATCH_SIZE) < 2:
        raise ValueError("SOLVER.BATCH_SIZE must be at least 2 because the model uses BatchNorm")
    if cfg.DA.RANDOM_DIM is None:
        raise ValueError("DA.RANDOM_DIM is required for CDAN")


def _drop_singleton_last_batch(n_examples, batch_size):
    return n_examples > 1 and n_examples % batch_size == 1


def _make_loader(dataset, batch_size, shuffle, seed, num_workers, training):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=_drop_singleton_last_batch(len(dataset), batch_size) if training else False,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def run_experiment(
    cfg,
    dataset,
    fold,
    seed,
    data_root,
    feature_root,
    feature_seed,
    run_dir,
    device_name="cuda:0",
):
    """Run one fold with fold-specific projected features and return metrics."""
    cfg = cfg.clone()
    cfg.SEED = int(seed)
    cfg.RESULT.OUTPUT_DIR = str(run_dir)
    validate_config(cfg)
    set_seed(seed)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    split_dir = resolve_split_dir(data_root, dataset, fold)
    feature_dir = resolve_feature_dir(feature_root, dataset, fold, feature_seed)
    source_df = pd.read_csv(split_dir / "source_train.csv").reset_index(drop=True)
    target_train_df = pd.read_csv(split_dir / "target_train.csv").reset_index(drop=True)
    target_test_df = pd.read_csv(split_dir / "target_test.csv").reset_index(drop=True)

    source_dataset = ProjectedFeatureDataset(source_df, feature_dir, labeled=True)
    target_adaptation_dataset = ProjectedFeatureDataset(
        target_train_df, feature_dir, labeled=False
    )
    target_test_dataset = ProjectedFeatureDataset(target_test_df, feature_dir, labeled=True)

    batch_size = int(cfg.SOLVER.BATCH_SIZE)
    num_workers = int(cfg.SOLVER.NUM_WORKERS)
    source_loader = _make_loader(
        source_dataset, batch_size, True, seed, num_workers, training=True
    )
    test_loader = _make_loader(
        target_test_dataset, batch_size, False, seed, num_workers, training=False
    )

    target_loader = _make_loader(
        target_adaptation_dataset,
        batch_size,
        True,
        seed + 10000,
        num_workers,
        training=True,
    )
    train_loader = MultiDataLoader(
        [source_loader, target_loader], max(len(source_loader), len(target_loader))
    )

    model = PRIDA(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.SOLVER.LR))
    discriminator = Discriminator(
        input_size=int(cfg.DA.RANDOM_DIM), n_class=int(cfg.DECODER.BINARY)
    ).to(device)
    optimizer_da = torch.optim.Adam(
        discriminator.parameters(), lr=float(cfg.SOLVER.DA_LR)
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    write_json(
        {
            "dataset": dataset,
            "fold": int(fold),
            "seed": int(seed),
            "feature_seed": int(feature_seed),
            "split_dir": str(split_dir.resolve()),
            "feature_dir": str(feature_dir.resolve()),
            "device": str(device),
            "source_rows": len(source_df),
            "target_train_rows": len(target_train_df),
            "target_test_rows": len(target_test_df),
            "target_train_labels_exposed_to_trainer": False,
        },
        run_dir / "run_metadata.json",
    )

    trainer = Trainer(
        dataset,
        model,
        optimizer,
        device,
        train_loader,
        test_loader,
        opt_da=optimizer_da,
        discriminator=discriminator,
        fold=fold,
        seed=seed,
        run_dir=run_dir,
        **cfg
    )
    metrics = trainer.train()

    del trainer, model, optimizer, discriminator, optimizer_da
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics
