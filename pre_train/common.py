import hashlib
from pathlib import Path

import pandas as pd
from Bio import SeqIO


RNA_FASTA_NAMES = {
    "RPI488": "RPI488_rna.fasta",
    "RPI1807": "RPI1807_rna_seq.fa",
    "RPI369": "RPI369_rna.fa",
    "Aptamer": "total_rna.fasta",
}


def resolve_split_dir(data_root, dataset, fold):
    dataset_dir = Path(data_root) / dataset
    fold_dir = dataset_dir / "folds" / ("fold_%d" % int(fold))
    if fold_dir.is_dir():
        return fold_dir
    if int(fold) == 1 and (dataset_dir / "source_train.csv").is_file():
        return dataset_dir
    raise FileNotFoundError("Cannot resolve %s fold_%d" % (dataset, int(fold)))


def resolve_rna_fasta(data_root, dataset):
    if dataset not in RNA_FASTA_NAMES:
        raise ValueError("No RNA FASTA mapping for dataset %s" % dataset)
    path = Path(data_root) / dataset / RNA_FASTA_NAMES[dataset]
    if not path.is_file():
        raise FileNotFoundError("RNA FASTA does not exist: %s" % path)
    return path


def read_fasta(paths):
    records = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("FASTA does not exist: %s" % path)
        for record in SeqIO.parse(str(path), "fasta"):
            sequence_id = str(record.id)
            sequence = str(record.seq).upper().replace("T", "U") if "rna" in path.name.lower() else str(record.seq).upper()
            if sequence_id in records and records[sequence_id] != sequence:
                raise ValueError("Conflicting FASTA records for ID %s" % sequence_id)
            records[sequence_id] = sequence
    return records


def load_fold_tables(data_root, dataset, fold):
    split_dir = resolve_split_dir(data_root, dataset, fold)
    tables = {
        name: pd.read_csv(split_dir / (name + ".csv")).reset_index(drop=True)
        for name in ("source_train", "target_train", "target_test")
    }
    for name, table in tables.items():
        missing = {"Protein", "RNA", "Y"}.difference(table.columns)
        if missing:
            raise ValueError("%s is missing columns %s" % (name, sorted(missing)))
    return split_dir, tables


def load_fold_sequences(data_root, dataset, fold):
    split_dir, tables = load_fold_tables(data_root, dataset, fold)
    protein_paths = [split_dir / "source_proteins.fasta", split_dir / "target_proteins.fasta"]
    protein_records = read_fasta(protein_paths)
    rna_records = read_fasta([resolve_rna_fasta(data_root, dataset)])

    all_table = pd.concat(list(tables.values()), ignore_index=True)
    required_proteins = set(all_table["Protein"].astype(str))
    required_rnas = set(all_table["RNA"].astype(str))
    missing_proteins = sorted(required_proteins.difference(protein_records))
    missing_rnas = sorted(required_rnas.difference(rna_records))
    if missing_proteins or missing_rnas:
        raise KeyError(
            "Missing FASTA IDs: proteins=%s rnas=%s"
            % (missing_proteins[:10], missing_rnas[:10])
        )

    source_protein_records = read_fasta([split_dir / "source_proteins.fasta"])
    target_protein_records = read_fasta([split_dir / "target_proteins.fasta"])
    source_ids = set(tables["source_train"]["Protein"].astype(str))
    if not source_ids.issubset(source_protein_records):
        raise AssertionError("Source CSV contains proteins absent from source_proteins.fasta")
    shared_sequences = set(source_protein_records.values()).intersection(target_protein_records.values())
    if shared_sequences:
        raise AssertionError("Source and Target protein FASTAs share exact sequences")

    protein_records = {sequence_id: protein_records[sequence_id] for sequence_id in required_proteins}
    rna_records = {sequence_id: rna_records[sequence_id] for sequence_id in required_rnas}
    return split_dir, tables, protein_records, rna_records


def source_sequence_table(data_root, dataset, fold):
    split_dir, tables, protein_records, rna_records = load_fold_sequences(
        data_root, dataset, fold
    )
    source = tables["source_train"].copy()
    source["Protein_sequence"] = source["Protein"].astype(str).map(protein_records)
    source["RNA_sequence"] = source["RNA"].astype(str).map(rna_records)
    if source[["Protein_sequence", "RNA_sequence"]].isna().any().any():
        raise AssertionError("Failed to map Source IDs to sequences")
    return split_dir, source


def artifact_dir(root, dataset, fold, seed):
    return Path(root) / dataset / ("fold_%d" % int(fold)) / ("seed_%d" % int(seed))


def feature_dir(root, dataset, fold, seed):
    return Path(root) / dataset / ("fold_%d" % int(fold)) / ("seed_%d" % int(seed))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
