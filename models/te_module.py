
import torch
import torch.nn as nn
import math
from einops import rearrange


class MDI_Module(nn.Module):
    """
    Multidimensional invariant module: stack of MDI layers, then drop pooled axes.
    After each MDI_Layer pools dim axes to size 1, reshape removes those axes
    (sorted descending) so output rank decreases by len(dim).
    """

    def __init__(self, d_feature, num_heads, dim):
        super(MDI_Module, self).__init__()
        self.dim = dim
        layers = []
        for d in dim:
            layers.append(MDI_Layer(d_feature, num_heads, d))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        :param x: [bs, M1, M2, ..., Mk, F]
        :return: same feature width F, with axes in self.dim removed
        """
        sp = x.size()
        sp = list(sp)
        for layer in self.layers:
            x = layer(x)
        dim = sorted(self.dim, reverse=True)
        for d in dim:
            del sp[d]
        x = torch.reshape(x, sp)
        return x


class MDI_Layer(nn.Module):
    """
    Single-axis permutation-invariant pooling via multi-head attention against
    a learnable query s.

    Args:
        d_feature (int): Feature width F.
        num_heads (int): Number of attention heads; must divide d_feature.
        dim (int): Axis index to pool.
    """

    def __init__(self, d_feature, num_heads, dim):
        super(MDI_Layer, self).__init__()
        self.s = nn.Parameter(torch.Tensor(1, 1, d_feature))
        nn.init.xavier_uniform_(self.s)

        self.d_feature = d_feature
        self.num_heads = num_heads
        self.fc_q = nn.Linear(d_feature, d_feature)
        self.fc_k = nn.Linear(d_feature, d_feature)
        self.fc_v = nn.Linear(d_feature, d_feature)
        self.fc_o = nn.Linear(d_feature, d_feature)
        self.relu=nn.PReLU()

        self.dim = dim

    def forward(self, x):
        """
        :param x: [bs, ..., Mj, ..., F], axis self.dim has length Mj
        :return: [bs, ..., 1, ..., F], same rank, pooled axis length set to 1
        """
        # Move pooled axis next to feature so flatten groups the rest into batch.
        x = torch.moveaxis(x, self.dim, -2)
        sp = x.size()
        x = x.flatten(start_dim=0, end_dim=-3)

        q = self.fc_q(self.s.repeat(x.size(0), 1, 1))
        k, v = self.fc_k(x), self.fc_v(x)
        dim_split = self.d_feature // self.num_heads
        q_ = torch.cat(q.split(dim_split, 2), 0)
        k_ = torch.cat(k.split(dim_split, 2), 0)
        v_ = torch.cat(v.split(dim_split, 2), 0)

        a = torch.softmax(q_.bmm(k_.transpose(1, 2)) / math.sqrt(self.d_feature), 2)
        o = torch.cat((q_ + a.bmm(v_)).split(q.size(0), 0), 2)
        o = o + self.relu(self.fc_o(o))

        sp = list(sp)
        sp[-2] = 1
        x = torch.reshape(o, sp)
        x = torch.moveaxis(x, -2, self.dim)
        return x


class HOE_2_1_Module(nn.Module):
    """
    2-to-1 high-order equivariance block on SLPN mat (paper HOE-BLK).
    Reduces the two user axes of mat [bs, K, K, L, D_c] to one user axis.

    Args:
        d_in (int): Input feature width.
        d_out (int): Output feature width.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = nn.Linear(5 * d_in, d_in)
        self.fc2 = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm2d(d_in)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        :param x: [bs, K, K, L, d_in]
        :return: [bs, K, L, d_out]
        """
        # x: [bs, K, K, L, F_in]
        sp = x.size()
        x = rearrange(x, 'bs ue1 ue2 l f -> (bs l) f ue1 ue2')
        x = ops_2_to_1(x)
        x = rearrange(x, '(bs l) f ue -> bs ue l f', bs=sp[0])
        x = self.fc1(x)

        x = rearrange(x, 'bs ue l f -> bs f ue l')
        x = self.act(self.bn(x))
        x = rearrange(x, 'bs f ue l -> bs ue l f')
        x = self.fc2(x)
        return x


def ops_2_to_1(inputs):
    """
    Five mean-based reductions on a K x K plane for HOE_2_1_Module.

    :param inputs: [B, F, K, K]
    :return: [B, 5F, K] — diag; tiled mean(diag); mean(rows); mean(cols);
             tiled mean(all), concatenated on the feature axis
    """
    dim = inputs.shape[-1]

    diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)
    mean_diag_part = diag_part.mean(dim=2, keepdims=True)
    mean_rows = inputs.mean(dim=3)
    mean_cols = inputs.mean(dim=2)
    mean_all = inputs.mean(dim=(2, 3))

    op1 = diag_part
    # Broadcast scalar-per-batch mean(diag) back to length K.
    op2 = torch.tile(mean_diag_part, (1, 1, dim))
    op3 = mean_rows
    op4 = mean_cols
    op5 = torch.tile(mean_all.unsqueeze(2), (1, 1, dim))
    ops = torch.cat([op1, op2, op3, op4, op5], 1)
    return ops


class MDE_Module(nn.Module):
    """
    Multidimensional equivariant module

    Args:
        in_features (int): Input feature width F_in.
        out_features (int): Output feature width F_out.
        dim (list): Axis indices or index lists for mean-pools.
            SLPN uses generate_mde_dim_list(n_dim=2) -> [[1], [2], [1, 2]].
    """

    def __init__(self, in_features, out_features, dim=[1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]):
        super(MDE_Module, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(in_features * (1 + len(dim)), out_features)

    def forward(self, x):
        """
        :param x: [bs, M1, ..., Mk, in_features]
        :return: [bs, M1, ..., Mk, out_features]
        """
        pooled = [torch.mean(x, d, keepdim=True).expand_as(x) for d in self.dim]
        state = torch.cat([x] + pooled, dim=-1)
        y = self.linear(state)
        return y


class MDE_Module_LowFLOPs(nn.Module):
    """
    Low-FLOP MDE.

    Args:
        in_features (int): Input feature width F.
        out_features (int): Output feature width.
        dim (list): Axis subsets for mean-pools.
    """

    def __init__(self, in_features, out_features, dim=[1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]):
        super(MDE_Module_LowFLOPs, self).__init__()
        self.dim = dim
        layers = []
        self.linear = nn.Linear(in_features, out_features)
        for d in dim:
            layers.append(nn.Linear(in_features, out_features, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        :param x: [bs, M1, ..., Mk, in_features]
        :return: [bs, M1, ..., Mk, out_features]
        """
        y = self.linear(x)
        for i, layer in enumerate(self.layers):
            # Mean over listed axes keeps keepdim; Linear then expand_as(y).
            y = y + layer(torch.mean(x, self.dim[i], keepdim=True)).expand_as(y)
        return y


class LinearRelu(nn.Module):
    """
    Linear followed by ReLU.

    Args:
        in_features (int): Input feature width.
        out_features (int): Output feature width.
    """

    def __init__(self, in_features, out_features):
        super(LinearRelu, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu=nn.ReLU()

    def forward(self,x):
        """
        :param x: [..., in_features]
        :return: [..., out_features]
        """
        x=self.linear(x)
        x=self.relu(x)
        return x


class FeatureBatchNorm(nn.Module):
    """
    BatchNorm on the last feature axis for [bs, M1, ..., Mk, F], any k >= 1.

    Args:
        num_features (int): Feature width F (BatchNorm1d channels).
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        """
        :param x: [bs, M1, ..., Mk, F]
        :return: [bs, M1, ..., Mk, F]
        """
        lead = x.shape[:-1]
        F = x.shape[-1]
        # [bs, prod(M), F] then [bs, F, prod(M)] for BatchNorm1d.
        x = x.reshape(lead[0], -1, F).transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2).reshape(*lead, F)
        return x


class FA_MDE(nn.Module):
    """
    Feature-attention MDE (paper f_FA-MDE).

    Args:
        d_feature (int): Feature width F.
        dim: Equivariant axes to pool.
    """

    def __init__(self, d_feature, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            nn.Linear(d_feature, d_feature, bias=False),
            nn.ReLU(),
            nn.Linear(d_feature, d_feature, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: [bs, M1, ..., MN, F]
        :return: feature-attention map, broadcastable to x
        """
        avg_out = self.fc(torch.mean(x, dim=self.dim, keepdim=True))
        max_out = self.fc(torch.amax(x, dim=self.dim, keepdim=True))
        return self.sigmoid(avg_out + max_out)


class RMDE_Module(nn.Module):
    """
    ReLU-MDE (paper f_RMDE): each branch is Linear+ReLU, then summed.
    Differs from MDE, which aggregates first and activates afterward.
    Used inside EA-MDE.

    Args:
        in_features (int): Input feature width.
        out_features (int): Output feature width.
        dim (list): Axis subsets for mean-pools, e.g. [[1], [2], [1, 2]].
    """

    def __init__(self, in_features, out_features, dim):
        super().__init__()
        self.dim = dim
        self.linear = LinearRelu(in_features, out_features)
        self.layers = nn.ModuleList([
            LinearRelu(in_features, out_features) for _ in dim
        ])
        self.bn = FeatureBatchNorm(out_features)

    def forward(self, x):
        """
        :param x: [bs, M1, ..., MN, in_features]
        :return: [bs, M1, ..., MN, out_features]
        """
        y = self.linear(x)
        for i, layer in enumerate(self.layers):
            y = y + layer(torch.mean(x, self.dim[i], keepdim=True).expand_as(x))
        y = self.bn(y)
        return y


class EA_MDE(nn.Module):
    """
    Equivariant-dimension attention MDE (paper f_EA-MDE).

    Args:
        dim (list): Axis subsets for the internal RMDE_Module.
    """

    def __init__(self, dim):
        super().__init__()
        self.rmde = RMDE_Module(2, 1, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: [bs, M1, ..., MN, F]
        :return: [bs, M1, ..., MN, 1]
        """
        avg_out = torch.mean(x, dim=-1, keepdim=True)
        max_out, _ = torch.max(x, dim=-1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=-1)
        x = self.rmde(x)
        return self.sigmoid(x)


class AMDE_Attention(nn.Module):
    """
    Decoupled TE attention inside AMDE: FA-MDE then EA-MDE.

    Args:
        out_features (int): Feature width F.
        fa_dim (tuple|list[int]): Axes pooled by FA-MDE, e.g. (1, 2).
        ea_dim (list): RMDE axis subsets for EA-MDE, e.g. [[1], [2], [1, 2]].
    """

    def __init__(self, out_features, fa_dim, ea_dim):
        super().__init__()
        self.fa_mde = FA_MDE(out_features, fa_dim)
        self.ea_mde = EA_MDE(ea_dim)

    def forward(self, x):
        """
        :param x: [bs, M1, ..., MN, F]
        :return: [bs, M1, ..., MN, F]
        """
        x = self.fa_mde(x) * x
        x = self.ea_mde(x) * x
        return x
