# -*-coding:utf-8-*-
from func.train_func import *
from func import init_func
from models.te_models import *
from models.prec_models import *
import os

if __name__ == '__main__':
    init_func.setup_seed(3407)

    out_folder = './save_data/CIZF_UE12TX14_QPSK/'
    in_folder = './train_data/CIZF_UE12TX14_QPSK/'
    upsilon_file = 'upsilon_cizf'
    upsilon_key = 'upsilon_cizf'
    symbol_file = 'symbol_data'
    symbol_key = 'symbol_data'
    delta_file = 'delta_cizf'
    delta_key = 'delta_cizf'

    batch_size, epoch_num, lr_step_size = 400, 800, 400
    learn_rate = 5e-3
    init_type = 'kaiming'
    net_name = 'TE'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    gpu_id = 3

    amde_dim_list = generate_amde_dim_list(n_amde_layer=4, n_dim=2)
    mde_dim_list = generate_mde_dim_list(n_dim=2)
    fa_dim = (1, 2)
    ea_dim = generate_ea_dim_list(n_dim=2)
    with open(out_folder + 'amde_dim_list.txt', 'w') as f:
        f.write(str(amde_dim_list))
        f.write('\n\n')
        f.write(str(mde_dim_list))

    model = SLPN(
        d_c=8, d_b=4, d_hidden=4, n_amde_layer=4,
        amde_dim_list=amde_dim_list, mde_dim_list=mde_dim_list,
        fa_dim=fa_dim, ea_dim=ea_dim)

    precoding_train_param = init_func.PrecodingTrainParam_CIZF(
        model, net_name, epoch_num, lr_step_size, in_folder,
        out_folder, learn_rate,
        init_type, batch_size, gpu_id, lr_gamma=0.1,
        upsilon_file=upsilon_file, upsilon_key=upsilon_key,
        symbol_file=symbol_file, symbol_key=symbol_key,
        delta_file=delta_file, delta_key=delta_key,
        constellation='QPSK', pskOrder=2,
        n_train=90000, n_test=10000)
    CIZF_SLP_train_supervised(precoding_train_param)
