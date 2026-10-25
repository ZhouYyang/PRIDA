# Introduction
This repository contains the PyTorch implementation of **PRIDA** framework, predicting RNA-protein interactions based on sequences and deep learning.

# Installion Guide
## create a new conda environment
$ conda env create -f environment.yml
$ conda activate PRIDA


## install according to the requirements

`environment.yml` pins `torch==1.10.0+cu111` and `torchvision==0.11.0+cu111`, which are not on PyPI. If the environment fails to solve, install them from the PyTorch index first and then create the environment:

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Substitute the CUDA build that matches your driver if cu111 does not.

# Checkpoints
1. ESM-2 Protein Language Model
Visit https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main and download the entire repository folder to: path/to/PRIDA/checkpoints/ESM2/

2. RiNALMo Language Model
Visit https://huggingface.co/multimolecule/rinalmo-giga/tree/main and download the entire repository folder to: path/to/PRIDA/checkpoints/RiNALMo/

# Datasets
The `datasets` folder contains all experimental data used in PRIDA: RPI1807, RPI488, RPI369 and Aptamer.

Each dataset provides five Source/Target domain splits under `datasets/dataset_name/folds/fold_1` ... `fold_5`. Every split contains `source_train.csv` (labelled Source interactions), `target_train.csv` (Target interactions used without labels for domain adaptation), `target_test.csv` (held-out Target evaluation set), and the protein FASTA of each domain. RNA sequences are in the dataset-level FASTA. All CSV files use the columns `Protein,RNA,Y`.

The splits are homology-constrained: protein homology groups are assigned wholly to Source or wholly to Target, so no Target protein is homologous to any Source protein within a split. The five splits are not mutually exclusive cross-validation folds, so run all five and report the mean and standard deviation.

# Source domain pre-training
Each split trains its own projection head on that split's `source_train.csv` labels only, then applies it to every sequence the split needs. Target labels are never used here.

Step 1: Pre-train the projection head and extract features. One command does both stages for all five splits:

```
python pre_train/prepare_folds.py --data dataset_name --folds 1 2 3 4 5 --seed 42 --device cuda:0
```

Repeat for each dataset. The command is resumable, and writes the checkpoint to `artifacts/pretrain/dataset_name/fold_k/seed_42/best_pretrain_proj.pt` and the features to `features/dataset_name/fold_k/seed_42/`.

Step 2: Run the stages separately if preferred.

```
python pre_train/train.py --data dataset_name --fold 1 --seed 42
python pre_train/extract_protein.py --data dataset_name --fold 1 --seed 42
python pre_train/extract_rna.py --data dataset_name --fold 1 --seed 42
```

For RNA sequences, run extract_rna.py.
For protein sequences, run extract_protein.py

If the language-model weights are not in `checkpoints/`, pass `--esm2-path` and `--rinalmo-path`.

# RNA-Protein interaction prediction
The `configs` folder provides optim hyperparameters and the optim hyperparameters are designed by "hy.py" through the command:
```
python hy.py --data dataset_name
```
One sampled configuration is run on all five splits and ranked by the mean Target-test AUROC, so there is no per-split tuning. The optimized parameter file are saved in folder `configs/`, overwriting the one shipped here.

Adjust the parameters in the corresponding dataset configuration file (.yaml) for main.py.

```
python main.py --data dataset_name --fold 1
```

Run `--fold 1` through `--fold 5` and average the five `metrics.json` files. Results are written to `runs/final/dataset_name/fold_k/seed_42/`, including the resolved configuration, training history, metrics and Target-test predictions.

AUROC and AUPRC do not depend on the decision threshold. The threshold-dependent metrics (sensitivity, specificity, accuracy, precision, recall, F1, MCC) are reported at the per-split F1-maximising threshold, which is the default. Use `--threshold-rule fixed` to score at `RESULT.THRESHOLD` instead, and apply the same rule to every method being compared.

# aptamer design

If you want to design nucleic acid aptamers for a target protein, first save a trained model by adding `--save-model` to main.py, which writes `best_model.pt` into the run directory.

Next, to proceed with aptamer design, configure the corresponding file paths in design/design.py.

```
python design/design.py
```

To score each candidate by the mean of the five split models instead of a single model:

```
python design/prida_ensemble_mcts.py --output-dir generated/aptamers --weight-root runs/final --config configs/Aptamer.yaml --device cuda:0
```

`--weight-root` must contain `Aptamer/fold_k/seed_42/best_model.pt` for k = 1..5.
