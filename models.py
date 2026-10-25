import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from ban import BANLayer


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class PRIDA(nn.Module):
    def __init__(self, **config):
        super(PRIDA, self).__init__()
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        rna_emb_dim = config["RNA"]["EMBEDDING_DIM"]
        rna_num_filters = config["RNA"]["NUM_FILTERS"]
        rna_kernel_size = config["RNA"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        attn_heads = config["ATTN"]["HEADS"]
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, attn_heads)
        self.rna_extractor = RnaCNN(rna_emb_dim, rna_num_filters, rna_kernel_size, attn_heads)

        self.fusion = weight_norm(
            BANLayer(
                v_dim=rna_num_filters[-1],
                q_dim=num_filters[-1],
                h_dim=mlp_in_dim,
                h_out=ban_heads,
            ),
            name="h_mat",
            dim=None,
        )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, rna, pro, mode="train"):
        
        rna = self.rna_extractor(rna)
        pro = self.protein_extractor(pro)
        f, att = self.fusion(rna, pro)
        score = self.mlp_classifier(f)
        if mode == "train":
            return rna, pro, f, score, att
        elif mode == "eval":
            return rna, pro, score, att

    def predict_score(self, rna, pro):
        rna_features = self.rna_extractor(rna)
        pro_features = self.protein_extractor(pro)
        fused_features, _ = self.fusion(rna_features, pro_features)
        score = self.mlp_classifier(fused_features)
        score = F.softmax(score, dim=1)[:, 1]
        return score.item()


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, attn_heads=4):
        super(ProteinCNN, self).__init__()
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size

        self.conv1 = nn.Conv1d(embedding_dim, out_channels=in_ch[1], kernel_size=kernels[0], padding=1)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding=1)
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding=1)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

        self.attn = nn.MultiheadAttention(embed_dim=in_ch[-1], num_heads=attn_heads, batch_first=True)
        self.ln = nn.LayerNorm(in_ch[-1])

    def forward(self, v):
        v = v.float()
        if v.ndim == 3:
            v = v.transpose(2, 1)
        else:
            v = v.unsqueeze(-1)

        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))  # [B, C, L]

        # permute to [B, L, C] for attention
        v = v.permute(0, 2, 1)
        attn_out, _ = self.attn(v, v, v)
        v = self.ln(v + attn_out)  # residual connection + normalization
        return v

class RnaCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, attn_heads=4):
        super(RnaCNN, self).__init__()
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size

        self.conv1 = nn.Conv1d(embedding_dim, out_channels=in_ch[1], kernel_size=kernels[0], padding=1)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding=1)
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding=1)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

        self.attn = nn.MultiheadAttention(embed_dim=in_ch[-1], num_heads=attn_heads, batch_first=True)
        self.ln = nn.LayerNorm(in_ch[-1])

    def forward(self, v):
        v = v.float()
        if v.ndim == 3:
            v = v.transpose(2, 1)
        else:
            v = v.unsqueeze(-1)

        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))  # [B, C, L]

        v = v.permute(0, 2, 1)               # [B, L, C] for attention
        attn_out, _ = self.attn(v, v, v)
        v = self.ln(v + attn_out)
        return v



class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class RandomLayer(nn.Module):
    """Randomised multilinear map conditioning the CDAN discriminator."""

    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        for index, input_dim in enumerate(input_dim_list):
            self.register_buffer(
                "random_matrix_%d" % index,
                torch.randn(input_dim, output_dim),
            )

    def forward(self, input_list):
        matrices = [getattr(self, "random_matrix_%d" % i) for i in range(self.input_num)]
        return_list = [torch.mm(input_list[i], matrices[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor
