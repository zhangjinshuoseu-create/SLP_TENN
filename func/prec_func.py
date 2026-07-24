import torch
import math


def complex2real(x):
    """
    Append real/imag as the last dimension.
    :param x: complex tensor (torch.imag requires a complex dtype)
    :return: real float tensor with shape [..., 2]
    """
    sp = x.size()
    dim = len(sp)
    x = torch.cat([torch.real(x).unsqueeze(dim), torch.imag(x).unsqueeze(dim)], dim)
    return x


def mpsk_decision_thresholds(txsig, pskOrder):
    """
    M-PSK CIR half-plane directions from symbol angles (M=2**pskOrder).
    :param txsig: complex
    :return: threshold_upper, threshold_lower — unit-modulus complex, same shape as txsig
    """
    M = pow(2, pskOrder)
    delta_angle = math.pi / M
    theta = torch.angle(txsig)
    upper_theta = theta + delta_angle
    lower_theta = theta - delta_angle
    threshold_upper = torch.exp(1j * upper_theta)
    threshold_lower = torch.exp(1j * lower_theta)
    return threshold_upper, threshold_lower


def mpsk_decision_thresholds16QAM(txsig):
    """
    Rectangular 16-QAM CIR boundary directions (unit-energy constellation scaled by 3/sqrt(10)).
    Returns complex tensors whose imag (upper) / real (lower) parts are in {-1, 0, +1},
    same leading shape as txsig; used as multiplier directions in KKT features.
    """
    txnorm = txsig / (3 / math.sqrt(10))
    threshold_upper = (
        torch.where(abs(torch.imag(txnorm) - 1) < 1e-3, 1.0, 0.0)
        + torch.where(abs(torch.imag(txnorm) + 1) < 1e-3, -1.0, 0.0)
    )
    threshold_lower = (
        torch.where(abs(torch.real(txnorm) - 1) < 1e-3, 1.0, 0.0)
        + torch.where(abs(torch.real(txnorm) + 1) < 1e-3, -1.0, 0.0)
    )
    threshold_upper = torch.complex(
        torch.zeros(threshold_upper.shape).to(txsig.device), threshold_upper)
    threshold_lower = torch.complex(
        threshold_lower, torch.zeros(threshold_lower.shape).to(txsig.device))
    return threshold_upper, threshold_lower


def slp_cir_thresholds(txsig, constellation='QPSK', pskOrder=2):
    """
    CIR boundary directions used to build SLPN inputs (CIZF-DL / CIMMSE-DL).
    QPSK/PSK: angle-based M-PSK boundaries.
    16QAM: rectangular decision boundaries of unit-energy 16-QAM.
    """
    constellation = constellation.upper().replace('-', '').replace('_', '')
    if constellation in ('16QAM', 'QAM16'):
        return mpsk_decision_thresholds16QAM(txsig)
    if constellation in ('QPSK', '4QAM', 'QAM4', 'PSK', 'MPSK'):
        return mpsk_decision_thresholds(txsig, pskOrder)
    raise ValueError(f'Unsupported constellation: {constellation}')


def slp_kkt_features(upsilon, tx, threshold_upper, threshold_lower):
    """
    Construct SLPN inputs from upsilon, symbols and CIR boundaries.
    :param upsilon: complex [bs, K, K]
    :param tx: complex [bs, K, L]
    :param threshold_upper/lower: complex, broadcastable with tx
    :return: mat [bs, K, K, L, 8], vec [bs, K, L, 4]  (real; 4 complex pairs via complex2real)
    """
    mattemp = upsilon.unsqueeze(-1) * threshold_upper.unsqueeze(1)
    mat1 = torch.conj(threshold_upper.unsqueeze(2)) * mattemp
    mat2 = torch.conj(threshold_lower.unsqueeze(2)) * mattemp
    mattemp = upsilon.unsqueeze(-1) * threshold_lower.unsqueeze(1)
    mat3 = torch.conj(threshold_lower.unsqueeze(2)) * mattemp
    mat4 = torch.conj(threshold_upper.unsqueeze(2)) * mattemp
    mat = torch.cat(
        [complex2real(mat1), complex2real(mat2), complex2real(mat3), complex2real(mat4)], -1)

    vec1 = torch.matmul(upsilon, tx)
    vec2 = vec1
    vec1 = torch.conj(threshold_upper) * vec1
    vec2 = torch.conj(threshold_lower) * vec2
    vec = torch.cat([complex2real(vec1), complex2real(vec2)], -1)
    return mat, vec


def rslpn_kkt_features_from_upsilon(upsilon, tx, pskOrder=2):
    """
    Build RSLPN-B (SLPN) inputs from real-valued KKT upsilon.
    :param upsilon: float [bs, L, 2K, 2K]
    :param tx: complex [bs, K, L]
    :return: mat [bs, K, K, L, 8], vec [bs, K, L, 4]
    """
    KNum = tx.size(1)
    upsilon_real = upsilon[:, :, 0:KNum, 0:KNum]
    upsilon_imag = upsilon[:, :, KNum:, 0:KNum]
    upsilon_complex = upsilon_real + 1j * upsilon_imag

    square_sum = torch.sum(torch.abs(upsilon_complex) ** 2, dim=(2, 3), keepdim=True)
    upsilon_complex = upsilon_complex / torch.sqrt(square_sum.clamp_min(1e-12))
    upsilon_complex = upsilon_complex * KNum

    threshold_upper, threshold_lower = slp_cir_thresholds(
        tx, constellation='QPSK', pskOrder=pskOrder)

    # [bs, L, K, K] -> [bs, K, K, L] for mat construction
    upsilon_complex = upsilon_complex.permute(0, 2, 3, 1)
    mattemp = upsilon_complex * threshold_upper.unsqueeze(1)
    mat1 = torch.conj(threshold_upper.unsqueeze(2)) * mattemp
    mat2 = torch.conj(threshold_lower.unsqueeze(2)) * mattemp
    mattemp = upsilon_complex * threshold_lower.unsqueeze(1)
    mat3 = torch.conj(threshold_lower.unsqueeze(2)) * mattemp
    mat4 = torch.conj(threshold_upper.unsqueeze(2)) * mattemp
    mat = torch.cat(
        [complex2real(mat1), complex2real(mat2),
         complex2real(mat3), complex2real(mat4)], -1)

    # vec: per-symbol matmul with L-varying upsilon
    upsilon_complex = upsilon_complex.permute(0, 3, 1, 2)  # bs L K K
    tx_temp = tx.permute(0, 2, 1).unsqueeze(-1)  # bs L K 1
    vec1 = torch.matmul(upsilon_complex, tx_temp).squeeze(-1)  # bs L K
    vec1 = vec1.permute(0, 2, 1)  # bs K L
    vec2 = vec1
    vec1 = torch.conj(threshold_upper) * vec1
    vec2 = torch.conj(threshold_lower) * vec2
    vec = torch.cat([complex2real(vec1), complex2real(vec2)], -1)
    return mat, vec


def dft_matrix(txAntNum):
    """
    Unitary DFT matrix (1/sqrt(N) normalized).
    :param txAntNum: N
    :return: complex tensor [N, N]
    """
    k = torch.arange(txAntNum).unsqueeze(1)  # [N, 1]
    n = torch.arange(txAntNum).unsqueeze(0)  # [1, N]
    W = torch.exp(-2j * math.pi * k * n / txAntNum)  # [N, N]
    W = W / math.sqrt(txAntNum)
    return W


def complex_mat_to_real_mat(mat):
    # mat: [..., K, N] complex -> [..., 2K, 2N] real block [[Re,-Im],[Im,Re]]
    mat_real = torch.real(mat)
    mat_imag = torch.imag(mat)
    mat_top = torch.cat([mat_real, -mat_imag], -1)
    mat_bottom = torch.cat([mat_imag, mat_real], -1)
    mat_out = torch.cat([mat_top, mat_bottom], -2)
    return mat_out


def get_RMMSE_kkt_Upsilon(channel, txsym, alpha, omega, sigma, model_rslp_a, mu,
                          pt=1, return_psi=False):
    """
    Robust MMSE KKT upsilon from imperfect CSI.
    :param channel: complex [bs, K, N]
    :param txsym: complex [bs, K, L]
    :param alpha: float [bs, L]
    :param omega: float [bs, K, N] — real-valued per-entry variance used in the
      KKT covariance construction
    :param sigma: float [bs, 1] — noise power
    :param model_rslp_a: RSLPN-A mapping to psi
    :param mu: complex [bs, K, N] 
    :param pt: transmit power scalar (default 1)
    :param return_psi: if False, return upsilon [bs, L, 2K, 2K];
      if True, return (upsilon, psi) with psi [bs, K, L]
    """
    [bs, K, N] = list(channel.shape)
    [_, _, L] = list(txsym.shape)
    ch_real = complex_mat_to_real_mat(channel)  # [bs,2K,2N]
    ch_real = ch_real.unsqueeze(1).expand(-1, L, -1, -1)  # [bs,L,2K,2N]
    ch_real = ch_real * alpha[:, :, None, None]
    psi = model_rslp_a(channel, txsym, sigma, alpha, mu)
    psi = torch.transpose(psi, 1, 2)  # [bs,L,K]
    psi_real = torch.cat([psi, psi], -1)
    psi_real = torch.diag_embed(psi_real)
    beta = 1 - alpha ** 2
    beta = torch.sqrt(beta)
    omega_M = omega.unsqueeze(1).expand(-1, L, -1, -1)  # [bs,L,K,N]
    omega_M = torch.sqrt(omega_M)
    omega_M = omega_M * beta[:, :, None, None]
    U_mat = dft_matrix(N).to(txsym.device)  # [N,N]
    Vt_H = torch.conj(torch.transpose(U_mat, -1, -2))  # [N,N]
    omega_M = omega_M.unsqueeze(-1).expand(-1, -1, -1, -1, N)  # [bs,L,K,N,N]
    omega_M = omega_M * Vt_H.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    omega_M = torch.matmul(torch.conj(torch.transpose(omega_M, -1, -2)), omega_M)
    omega_M = complex_mat_to_real_mat(omega_M)  # [bs,L,K,2N,2N]
    psi_sq = psi ** 2  # [bs,L,K]
    omega_M = psi_sq[:, :, :, None, None] * omega_M
    omega_M = torch.sum(omega_M, 2)  # [bs,L,2N,2N]
    noise = torch.sum(psi_sq, -1)  # [bs,L]
    noise = noise * sigma
    noise = noise / pt
    noise = torch.diag_embed(noise.unsqueeze(-1).expand(-1, -1, 2 * N))
    PrecMat = torch.matmul(
        torch.transpose(ch_real, -1, -2),
        torch.matmul(psi_real, torch.matmul(psi_real, ch_real)))
    PrecMat = PrecMat + omega_M + noise
    PrecMat = torch.linalg.inv(PrecMat)
    PrecMat = torch.matmul(
        PrecMat, torch.matmul(torch.transpose(ch_real, -1, -2), psi_real))
    innerMat = torch.matmul(psi_real, torch.matmul(ch_real, PrecMat))
    innerMat = torch.diag_embed(torch.ones(bs, L, 2 * K)).to(innerMat.device) - innerMat

    if return_psi:
        return innerMat, psi.transpose(-1, -2)
    return innerMat
