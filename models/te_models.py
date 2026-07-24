import torch.nn as nn
from . import te_module
from itertools import combinations


def generate_combinations(n):
    """All 2^n-1 non-empty subsets of [1,...,n]."""
    result = []
    for r in range(1, n + 1):
        result.extend([list(comb) for comb in combinations(range(1, n + 1), r)])
    return result


def generate_amde_dim_list(n_amde_layer, n_dim=2):
    """
    dim_list for AMDE_Network: length = n_amde_layer.
    Entry i is the equivariant-dimension set used by the i-th AMDE block.
    """
    return [generate_combinations(n_dim) for _ in range(n_amde_layer)]


def generate_mde_dim_list(n_dim=2):
    """
    Equivariant-dimension set for a single MDE layer (e.g. SLPN mde_b on tensor B).
    For 2-D TE on (K, L), this is [[1], [2], [1, 2]].
    """
    return generate_combinations(n_dim)


def generate_ea_dim_list(n_dim=2):
    """
    Equivariant-dimension subsets for EA-MDE (same power set as MDE dims).
    Prefer this name when the list is passed as ea_dim.
    """
    return generate_mde_dim_list(n_dim)


class AMDE_Network(nn.Module):
    """
    Stack of T AMDE blocks (paper: f_AMDE^{x T}) with input/output FC.
    Works for any number of equivariant axes: [bs, M1, ..., MN, F].
    """

    def __init__(self, d_in, d_out, n_amde_layer, d_hidden, dim_list, fa_dim, ea_dim):
        super().__init__()
        self.pre = nn.Linear(d_in, d_hidden)
        self.layers = nn.ModuleList([
            AMDE_Block(
                d_hidden=d_hidden, dim_list=dim_list[n_layer],
                fa_dim=fa_dim, ea_dim=ea_dim)
            for n_layer in range(n_amde_layer)
        ])
        self.final = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        # x: [bs, M1, ..., MN, F_in] -> [bs, M1, ..., MN, F_out]
        x = self.pre(x)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)


class AMDE_Block(nn.Module):
    """
    One AMDE residual block (paper):
      X'  = PReLU(BN(f_MDE(X)))
      X'' = BN(f_MDE(X'))
      then FA-MDE + EA-MDE gates, residual add, PReLU.
    Shapes preserved: [bs, M1, ..., MN, d_hidden].
    """

    def __init__(self, d_hidden, dim_list, fa_dim, ea_dim):
        super().__init__()
        self.mde_1 = te_module.MDE_Module_LowFLOPs(
            in_features=d_hidden, out_features=d_hidden, dim=dim_list)
        self.mde_2 = te_module.MDE_Module_LowFLOPs(
            in_features=d_hidden, out_features=d_hidden, dim=dim_list)
        self.amde_attn = te_module.AMDE_Attention(
            out_features=d_hidden, fa_dim=fa_dim, ea_dim=ea_dim)
        self.bn_1 = te_module.FeatureBatchNorm(d_hidden)
        self.bn_2 = te_module.FeatureBatchNorm(d_hidden)
        self.act_1 = nn.PReLU()
        self.act_out = nn.PReLU()

    def forward(self, x):
        residual = x
        x = self.act_1(self.bn_1(self.mde_1(x)))
        x = self.bn_2(self.mde_2(x))
        x = self.amde_attn(x) + residual
        return self.act_out(x)
