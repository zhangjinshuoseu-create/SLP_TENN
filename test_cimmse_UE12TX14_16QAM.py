# -*-coding:utf-8-*-
from func.train_func import *
from func import init_func
from models.te_models import *
from models.prec_models import *

if __name__ == '__main__':
    init_func.setup_seed(3407)

    out_folder = './save_data/CIMMSE_UE12TX14_16QAM/'
    in_folder = './train_data/CIMMSE_UE12TX14_16QAM/'
    upsilon_file = 'upsilon_cimmse'
    upsilon_key = 'upsilon_cimmse'
    symbol_file = 'symbol_data'
    symbol_key = 'symbol_data'
    delta_file = 'delta_cimmse'
    delta_key = 'delta_cimmse'

    net_name = 'TE'
    snr_list = [0, 5, 10, 15, 20, 25, 30]
    gpu_id = 5

    amde_dim_list = generate_amde_dim_list(n_amde_layer=4, n_dim=2)
    mde_dim_list = generate_mde_dim_list(n_dim=2)
    fa_dim = (1, 2)
    ea_dim = generate_ea_dim_list(n_dim=2)
    model = SLPN(
        d_c=8, d_b=4, d_hidden=4, n_amde_layer=4,
        amde_dim_list=amde_dim_list, mde_dim_list=mde_dim_list,
        fa_dim=fa_dim, ea_dim=ea_dim)

    precoding_test_param = init_func.PrecodingTestParam_CIMMSE(
        model, net_name, in_folder, out_folder,
        snr_list=snr_list, gpu_id=gpu_id,
        begin_num=90000, n_test=10000,
        upsilon_file=upsilon_file, upsilon_key=upsilon_key,
        symbol_file=symbol_file, symbol_key=symbol_key,
        delta_file=delta_file, delta_key=delta_key,
        constellation='16QAM', pskOrder=2)
    CIMMSE_SLP_test_supervised(precoding_test_param)
