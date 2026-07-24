import torch.nn as nn
import torch
from models import te_models
from models import te_module
from func.prec_func import complex2real


class RSLPN_A(nn.Module):
    """
    Robust SLP network A (paper RSLPN-A): regresses non-negative psi [bs, K, L].
    Trained with MSE against psi labels;
    """

    def __init__(self, d_in, d_hidden_1, n_amde_1, amde_dim_list_1,
                 d_hidden_2, n_amde_2, amde_dim_list_2,
                 fa_dim_1, ea_dim_1, fa_dim_2, ea_dim_2):
        super().__init__()
        self.mdi_dims = [2]

        self.amde_3d = te_models.AMDE_Network(
            d_in, d_hidden_1, n_amde_1, d_hidden_1, amde_dim_list_1,
            fa_dim_1, ea_dim_1)
        self.bn_1 = te_module.FeatureBatchNorm(d_hidden_1)
        self.prelu_1 = nn.PReLU()

        self.mdi = te_module.MDI_Module(d_hidden_1, 4, self.mdi_dims)
        self.bn_2 = te_module.FeatureBatchNorm(d_hidden_1)
        self.prelu_2 = nn.PReLU()

        self.amde_2d = te_models.AMDE_Network(
            d_hidden_1, 1, n_amde_2, d_hidden_2, amde_dim_list_2,
            fa_dim_2, ea_dim_2)
        self.output_act = nn.Softplus()

    def input_layer(self, channel, txsig, sigma, alpha, U_mat):
        # complex inputs only (complex2real uses torch.real/imag)
        channel = complex2real(channel)  # [bs, K, N, 2]
        U_mat = complex2real(U_mat)            # [bs, K, N, 2]
        txsig = complex2real(txsig)      # [bs, K, L, 2]
        dims = list(channel.shape)
        dims_tx = list(txsig.shape)
        sigma = sigma.unsqueeze(2).unsqueeze(3).unsqueeze(-1).expand(
            -1, dims[1], dims[2], dims_tx[2], -1)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(-1).expand(
            -1, dims[1], dims[2], -1, -1)
        channel = channel.unsqueeze(3).expand(-1, -1, -1, dims_tx[2], -1)
        txsig = txsig.unsqueeze(2).expand(-1, -1, dims[2], -1, -1)
        U_mat = U_mat.unsqueeze(3).expand(-1, -1, -1, dims_tx[2], -1)
        # d_in=8: H(2) + s(2) + sigma(1) + alpha(1) + U_mat(2)
        return torch.cat([channel, txsig, sigma, alpha, U_mat], -1)  # [bs, K, N, L, 8]

    def forward(self, channel, txsig, sigma, alpha, U_mat):
        # channel: complex [bs, K, N]; txsig: complex [bs, K, L]
        # sigma: float [bs, 1] (noise power); alpha: float [bs, L]
        # U_mat: U_mat feature complex [bs, K, N]
        # return: float [bs, K, L]
        x = self.input_layer(channel, txsig, sigma, alpha, U_mat)
        x = self.amde_3d(x)
        x = self.bn_1(x)
        x = self.prelu_1(x)
        x = self.mdi(x)
        x = self.bn_2(x)
        x = self.prelu_2(x)
        x = self.amde_2d(x)
        x = self.output_act(x)
        return x.squeeze(-1)


class SLPN(nn.Module):
    """
    Symbol-level precoding network (paper SLPN): approximates G(B, C) = D*.
    Also used as RSLPN-B under imperfect CSI (alias RSLPN_B).
    Inputs from KKT feature builders:
      mat (C): [bs, K, K, L, D1]  (D1=8 in current train scripts)
      vec (B): [bs, K, L, D2]     (D2=4 in current train scripts)
    Output: float [bs, K, L, 2] — raw regression of the two real channels of delta
    ; trained with MSE against real-valued delta labels.
    """

    def __init__(self, d_c, d_b, d_hidden, n_amde_layer, amde_dim_list, mde_dim_list,
                 fa_dim, ea_dim):
        super().__init__()

        # HOE path on C -> C''
        self.hoe_blk = te_module.HOE_2_1_Module(d_c, d_hidden)
        self.bn_c = te_module.FeatureBatchNorm(d_hidden)
        self.prelu_c = nn.PReLU()

        # MDE path on B -> B'
        self.mde_b = te_module.MDE_Module(
            in_features=d_b, out_features=d_hidden, dim=mde_dim_list)
        self.bn_b = te_module.FeatureBatchNorm(d_hidden)
        self.prelu_b = nn.PReLU()

        # [C'', B'] -> AMDE stack -> D
        self.amde_stack = te_models.AMDE_Network(
            2 * d_hidden, 2, n_amde_layer, d_hidden, amde_dim_list,
            fa_dim, ea_dim)

    def forward(self, mat, vec):
        # mat (C): [bs, K, K, L, D1]; vec (B): [bs, K, L, D2]
        feat_c = self.prelu_c(self.bn_c(self.hoe_blk(mat)))
        feat_b = self.prelu_b(self.bn_b(self.mde_b(vec)))
        return self.amde_stack(torch.cat([feat_c, feat_b], -1))


# RSLPN-B shares the SLPN architecture (paper).
RSLPN_B = SLPN
