import torch.nn as nn
import torch
from torch.nn import init
from torch.utils.data import Dataset
import numpy as np
import random

def init_weights(net, init_type='normal', cwg=5e-2, cbg=0.5, fwg=0.04, fbg=0.5, bwg=5e-2):
    def init_func(net):  # define the initialization function
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # Gaussian distribution initialization
                if init_type == 'normal':
                    init.normal_(m.weight.data, mean=0, std=cwg)
                # Xavier initialization, scaling factor applied to the variance
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=cwg)
                # Initialization function proposed for nonlinear activation functions like ReLU and Leaky ReLU
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=cwg)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, cbg)
            elif isinstance(m, nn.Linear):
                # Gaussian distribution initialization
                if init_type == 'normal':
                    init.normal_(m.weight.data, mean=0, std=fwg)
                # Xavier initialization, scaling factor applied to the variance
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=fwg)
                # Initialization function proposed for nonlinear activation functions like ReLU and Leaky ReLU
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=fwg)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, fbg)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, bwg)
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class PrecodingTrainParam_CIZF:
    """
    Training settings for CIZF-DL (perfect CSI).
    No snr_list: upsilon and delta labels are SNR-independent.
    Train loop uses upsilon + symbol + delta only.
    """

    def __init__(
            self,
            model,
            net_name,
            epoch_num,
            lr_step_size,
            in_folder='../datafolder',
            out_folder='../savefolder',
            learn_rate=5e-4,
            init_type='kaiming',
            batch_size=256,
            gpu_id=0,
            lr_gamma=0.1,
            upsilon_file='upsilon_cizf',
            upsilon_key='upsilon_cizf',
            symbol_file='symbol_data',
            symbol_key='symbol_data',
            delta_file='delta_cizf',
            delta_key='delta_cizf',
            constellation='QPSK',
            pskOrder=2,
            n_train=90000,
            n_test=10000):
        self.model = model
        self.net_name = net_name
        self.epoch_num = epoch_num
        self.lr_step_size = lr_step_size
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.learn_rate = learn_rate
        self.init_type = init_type
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.lr_gamma = lr_gamma
        self.upsilon_file = upsilon_file
        self.upsilon_key = upsilon_key
        self.symbol_file = symbol_file
        self.symbol_key = symbol_key
        self.delta_file = delta_file
        self.delta_key = delta_key
        self.constellation = constellation
        self.pskOrder = pskOrder
        self.n_train = n_train
        self.n_test = n_test


class PrecodingTrainParam_CIMMSE:
    """
    Training settings for CIMMSE-DL (perfect CSI, multi-SNR upsilon / delta labels).
    snr_list selects which SNR slices of upsilon / delta to use.
    Train loop uses upsilon + symbol + delta only.
    delta mat expected shape: [bs, n_snr, 2K, L].
    """

    def __init__(
            self,
            model,
            net_name,
            epoch_num,
            lr_step_size,
            in_folder='../datafolder',
            out_folder='../savefolder',
            learn_rate=5e-4,
            init_type='kaiming',
            batch_size=256,
            snr_list=None,
            gpu_id=0,
            lr_gamma=0.1,
            upsilon_file='upsilon_cimmse',
            upsilon_key='upsilon_cimmse',
            symbol_file='symbol_data',
            symbol_key='symbol_data',
            delta_file='delta_cimmse',
            delta_key='delta_cimmse',
            constellation='QPSK',
            pskOrder=2,
            n_train=90000,
            n_test=10000):
        self.model = model
        self.net_name = net_name
        self.epoch_num = epoch_num
        self.lr_step_size = lr_step_size
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.learn_rate = learn_rate
        self.init_type = init_type
        self.batch_size = batch_size
        self.snr_list = [0, 5, 10, 15, 20, 25, 30] if snr_list is None else snr_list
        self.gpu_id = gpu_id
        self.lr_gamma = lr_gamma
        self.upsilon_file = upsilon_file
        self.upsilon_key = upsilon_key
        self.symbol_file = symbol_file
        self.symbol_key = symbol_key
        self.delta_file = delta_file
        self.delta_key = delta_key
        self.constellation = constellation
        self.pskOrder = pskOrder
        self.n_train = n_train
        self.n_test = n_test


class PrecodingTrainParam_RSLPN_A:
    """
    Training settings for RSLPN-A: supervised MSE on psi [bs, K, L]
    (Softplus network output vs psi labels, paper Psi).
    Mat paths/keys: channel, symbol, U, psi (psi_file may include a subfolder).
    """

    def __init__(
            self,
            model_rslp_a,
            net_name,
            epoch_num,
            lr_step_size,
            in_folder='../datafolder',
            out_folder='../savefolder',
            channel_file='channel_data',
            channel_key='channel_data',
            learn_rate=5e-4,
            init_type='kaiming',
            batch_size=256,
            snr_list=None,
            gpu_id=0,
            lr_gamma=0.1,
            txlen=50,
            n_train=55000,
            n_test=5000,
            alpha_sim=0.995,
            symbol_file='symbol_data',
            symbol_key='symbol_data',
            U_file='U_data',
            U_key='U_data',
            psi_file='psi_data',
            psi_key='psi_data'):
        self.model_rslp_a = model_rslp_a
        self.net_name = net_name
        self.epoch_num = epoch_num
        self.lr_step_size = lr_step_size
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.channel_file = channel_file
        self.channel_key = channel_key
        self.learn_rate = learn_rate
        self.init_type = init_type
        self.batch_size = batch_size
        self.snr_list = [10, 15, 20, 25, 30, 35, 40] if snr_list is None else snr_list
        self.gpu_id = gpu_id
        self.lr_gamma = lr_gamma
        self.txlen = txlen
        self.n_train = n_train
        self.n_test = n_test
        self.alpha_sim = alpha_sim
        self.symbol_file = symbol_file
        self.symbol_key = symbol_key
        self.U_file = U_file
        self.U_key = U_key
        self.psi_file = psi_file
        self.psi_key = psi_key


class PrecodingTrainParam_RSLPN_B:
    """
    Training settings for RSLPN-B (SLPN): supervised MSE on delta [bs, K, L, 2]
    with frozen pretrained model_rslp_a used inside get_RMMSE_kkt_Upsilon.
    Mat paths/keys: channel, symbol, omega, U, and delta.
    delta mat expected shape: [bs, 2K, L, n_snr].
    """

    def __init__(
            self,
            model_rslp_b,
            model_rslp_a,
            net_name,
            epoch_num,
            lr_step_size,
            in_folder='../datafolder',
            out_folder='../savefolder',
            channel_file='channel_data',
            channel_key='channel_data',
            learn_rate=5e-4,
            init_type='kaiming',
            batch_size=256,
            snr_list=None,
            gpu_id=0,
            lr_gamma=0.1,
            txlen=50,
            n_train=55000,
            n_test=5000,
            alpha_sim=0.995,
            pskOrder=2,
            symbol_file='symbol_data',
            symbol_key='symbol_data',
            omega_file='omega_data',
            omega_key='omega_data',
            U_file='U_data',
            U_key='U_data',
            delta_file='delta_data',
            delta_key='delta_data'):
        self.model_rslp_b = model_rslp_b
        self.model_rslp_a = model_rslp_a
        self.net_name = net_name
        self.epoch_num = epoch_num
        self.lr_step_size = lr_step_size
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.channel_file = channel_file
        self.channel_key = channel_key
        self.learn_rate = learn_rate
        self.init_type = init_type
        self.batch_size = batch_size
        self.snr_list = [10, 15, 20, 25, 30, 35, 40] if snr_list is None else snr_list
        self.gpu_id = gpu_id
        self.lr_gamma = lr_gamma
        self.txlen = txlen
        self.n_train = n_train
        self.n_test = n_test
        self.alpha_sim = alpha_sim
        self.pskOrder = pskOrder
        self.symbol_file = symbol_file
        self.symbol_key = symbol_key
        self.omega_file = omega_file
        self.omega_key = omega_key
        self.U_file = U_file
        self.U_key = U_key
        self.delta_file = delta_file
        self.delta_key = delta_key


class PrecodingTestParam_CIZF:
    """
    Runtime settings for CIZF-DL MSE testing.
    Includes former CIZFDataParam fields (mat files / modulation).
    No snr_list: upsilon and delta are SNR-independent.
    Expected layouts: upsilon [bs, K, K], symbol [bs, K, L], delta [bs, 2K, L].
    """

    def __init__(
            self,
            model,
            net_name,
            in_folder='../datafolder',
            out_folder='../savefolder',
            batch_size=400,
            gpu_id=0,
            begin_num=90000,
            n_test=10000,
            upsilon_file='upsilon_cizf',
            upsilon_key='upsilon_cizf',
            symbol_file='symbol_data',
            symbol_key='symbol_data',
            delta_file='delta_cizf',
            delta_key='delta_cizf',
            constellation='QPSK',
            pskOrder=2):
        self.model = model
        self.net_name = net_name
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.begin_num = begin_num
        self.n_test = n_test
        self.upsilon_file = upsilon_file
        self.upsilon_key = upsilon_key
        self.symbol_file = symbol_file
        self.symbol_key = symbol_key
        self.delta_file = delta_file
        self.delta_key = delta_key
        self.constellation = constellation
        self.pskOrder = pskOrder


class PrecodingTestParam_CIMMSE:
    """
    Runtime settings for CIMMSE-DL MSE testing.
    Includes former CIMMSEDataParam fields (mat files / modulation).
    snr_list selects upsilon / delta SNR slices.
    Expected layouts: upsilon [bs, K, K, snrNum], symbol [bs, K, L],
    delta [bs, n_snr, 2K, L].
    """

    def __init__(
            self,
            model,
            net_name,
            in_folder='../datafolder',
            out_folder='../savefolder',
            snr_list=None,
            gpu_id=0,
            begin_num=90000,
            n_test=10000,
            upsilon_file='upsilon_cimmse',
            upsilon_key='upsilon_cimmse',
            symbol_file='symbol_data',
            symbol_key='symbol_data',
            delta_file='delta_cimmse',
            delta_key='delta_cimmse',
            constellation='QPSK',
            pskOrder=2):
        self.model = model
        self.net_name = net_name
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.snr_list = [0, 5, 10, 15, 20, 25, 30] if snr_list is None else snr_list
        self.gpu_id = gpu_id
        self.begin_num = begin_num
        self.n_test = n_test
        self.upsilon_file = upsilon_file
        self.upsilon_key = upsilon_key
        self.symbol_file = symbol_file
        self.symbol_key = symbol_key
        self.delta_file = delta_file
        self.delta_key = delta_key
        self.constellation = constellation
        self.pskOrder = pskOrder


class PrecodingTestParam_RSLPN:
    """
    Runtime settings for RSLPN-B MSE testing (with pretrained model_rslp_a).
    Independent of PrecodingTrainParam_RSLPN_*; no training-only fields.
    begin_num / n_test are per-channel counts before SNR tiling (multiplied by
    len(snr_list) inside the test loader), matching the RSLPN train split convention.
    Mat paths/keys mirror PrecodingTrainParam_RSLPN_B (channel, symbol, omega, U, delta).
    """

    def __init__(
            self,
            model_rslp_b,
            model_rslp_a,
            in_folder='../datafolder',
            out_folder='../savefolder',
            channel_file='channel_data',
            channel_key='channel_data',
            batch_size=400,
            snr_list=None,
            gpu_id=0,
            begin_num=55000,
            n_test=5000,
            txlen=50,
            alpha_sim=0.995,
            pskOrder=2,
            symbol_file='symbol_data',
            symbol_key='symbol_data',
            omega_file='omega_data',
            omega_key='omega_data',
            U_file='U_data',
            U_key='U_data',
            delta_file='delta_data',
            delta_key='delta_data'):
        self.model_rslp_b = model_rslp_b
        self.model_rslp_a = model_rslp_a
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.channel_file = channel_file
        self.channel_key = channel_key
        self.batch_size = batch_size
        self.snr_list = [10, 15, 20, 25, 30, 35, 40] if snr_list is None else snr_list
        self.gpu_id = gpu_id
        self.begin_num = begin_num
        self.n_test = n_test
        self.txlen = txlen
        self.alpha_sim = alpha_sim
        self.pskOrder = pskOrder
        self.symbol_file = symbol_file
        self.symbol_key = symbol_key
        self.omega_file = omega_file
        self.omega_key = omega_key
        self.U_file = U_file
        self.U_key = U_key
        self.delta_file = delta_file
        self.delta_key = delta_key


class DatasetFolder_3(Dataset):
    """Zip three equal-length tensors; __getitem__ returns (t1, t2, t3)[index]."""

    def __init__(self, tensor_1, tensor_2, tensor_3):
        self.label1 = tensor_1
        self.label2 = tensor_2
        self.label3 = tensor_3

    def __getitem__(self, index):
        return self.label1[index], self.label2[index], self.label3[index]

    def __len__(self):
        return self.label1.shape[0]


class DatasetFolder_5(Dataset):
    """Zip five equal-length tensors; __getitem__ returns a 5-tuple at index."""

    def __init__(self, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
        self.label1 = tensor_1
        self.label2 = tensor_2
        self.label3 = tensor_3
        self.label4 = tensor_4
        self.label5 = tensor_5

    def __getitem__(self, index):
        return (
            self.label1[index], self.label2[index], self.label3[index],
            self.label4[index], self.label5[index],
        )

    def __len__(self):
        return self.label1.shape[0]


class DatasetFolder_6(Dataset):
    """Zip six equal-length tensors; __getitem__ returns a 6-tuple at index."""

    def __init__(self, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
        self.label1 = tensor_1
        self.label2 = tensor_2
        self.label3 = tensor_3
        self.label4 = tensor_4
        self.label5 = tensor_5
        self.label6 = tensor_6

    def __getitem__(self, index):
        return (
            self.label1[index], self.label2[index], self.label3[index],
            self.label4[index], self.label5[index], self.label6[index],
        )

    def __len__(self):
        return self.label1.shape[0]
