from yacs.config import CfgNode as CN


_C = CN()

_C.SEED = 42

_C.RNA = CN()
_C.RNA.NUM_FILTERS = [64, 64, 64]
_C.RNA.KERNEL_SIZE = [1, 1, 1]
_C.RNA.EMBEDDING_DIM = 512

_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [64, 64, 64]
_C.PROTEIN.KERNEL_SIZE = [1, 1, 1]
_C.PROTEIN.EMBEDDING_DIM = 512

_C.BCN = CN()
_C.BCN.HEADS = 7

_C.ATTN = CN()
_C.ATTN.HEADS = 4

_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 128
_C.DECODER.HIDDEN_DIM = 256
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 2

_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3

_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
# "f1_optimal" picks the per-split F1-maximising threshold on the Target-test
# scores; "fixed" uses RESULT.THRESHOLD. AUROC and AUPRC do not depend on it.
_C.RESULT.THRESHOLD_RULE = "f1_optimal"
_C.RESULT.THRESHOLD = 0.5
_C.RESULT.SAVE_MODEL = False

# Conditional adversarial domain adaptation (CDAN).
_C.DA = CN()
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1.0
_C.DA.RANDOM_DIM = 256
_C.DA.USE_ENTROPY = False


def get_cfg_defaults():
    return _C.clone()
