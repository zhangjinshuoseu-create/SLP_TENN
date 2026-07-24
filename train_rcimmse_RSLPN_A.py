# -*-coding:utf-8-*-
from func.train_func import *
from func import init_func
from models.te_models import *
from models.prec_models import *
import os

if __name__ == '__main__':
    init_func.setup_seed(3407)

    out_folder = './save_data/RSLPN_A_UE12TX14_Robust/'
    in_folder = './train_data/RCIMMSE_UE12TX14_Robust/'
    channel_file = 'channel_data'
    channel_key = 'channel_data'
    symbol_file = 'symbol_data'
    symbol_key = 'symbol_data'
    U_file = 'U_data'
    U_key = 'U_data'
    psi_file = 'psi_data'
    psi_key = 'psi_data'
    batch_size, epoch_num, lr_step_size = 400, 300, 150
    learn_rate = 5e-3
    init_type = 'kaiming'
    net_name = 'TE'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    snr_list = [10, 15, 20, 25, 30, 35, 40]
    gpu_id = 3

    amde_dim_list_1 = generate_amde_dim_list(n_amde_layer=2, n_dim=3)
    amde_dim_list_2 = generate_amde_dim_list(n_amde_layer=2, n_dim=2)
    fa_dim_1 = (1, 2)
    ea_dim_1 = generate_ea_dim_list(n_dim=3)
    fa_dim_2 = (1, 2)
    ea_dim_2 = generate_ea_dim_list(n_dim=2)
    with open(out_folder + 'amde_dim_list.txt', 'w') as f:
        f.write(str(amde_dim_list_1))
        f.write('\n\n')
        f.write(str(amde_dim_list_2))

    model_rslp_a = RSLPN_A(
        d_in=8, d_hidden_1=16, n_amde_1=2, amde_dim_list_1=amde_dim_list_1,
        d_hidden_2=16, n_amde_2=2, amde_dim_list_2=amde_dim_list_2,
        fa_dim_1=fa_dim_1, ea_dim_1=ea_dim_1, fa_dim_2=fa_dim_2, ea_dim_2=ea_dim_2)
    precoding_train_param = init_func.PrecodingTrainParam_RSLPN_A(
        model_rslp_a, net_name, epoch_num, lr_step_size, in_folder,
        out_folder, channel_file, channel_key, learn_rate,
        init_type, batch_size, snr_list, gpu_id,
        lr_gamma=0.1, txlen=50, n_train=55000, n_test=5000,
        alpha_sim=0.995,
        symbol_file=symbol_file, symbol_key=symbol_key,
        U_file=U_file, U_key=U_key,
        psi_file=psi_file, psi_key=psi_key)
    RSLPN_A_train_supervised(
        precoding_train_param)
