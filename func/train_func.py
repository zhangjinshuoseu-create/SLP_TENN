import torch
import numpy as np
from scipy.io import loadmat, savemat
from func import init_func
import os
from func.prec_func import *
from func.init_func import DatasetFolder_3, DatasetFolder_5, DatasetFolder_6
import gc
import time
import torch.nn as nn


def RSLPN_test_supervised(
        precoding_test_param):
    """
    Test the full RSLPN pipeline under imperfect CSI (RCIMMSE / channel-aging).

    Runs RSLPN-A and RSLPN-B together as in RCIMMSE-DL: A estimates ``psi``,
    then B predicts perturbation factors ``delta``. Reports Test MSE and
    writes predictions to disk.
    """
    model_rslp_b = precoding_test_param.model_rslp_b
    model_rslp_a = precoding_test_param.model_rslp_a
    in_folder = precoding_test_param.in_folder
    out_folder = precoding_test_param.out_folder
    channel_file = precoding_test_param.channel_file
    channel_key = precoding_test_param.channel_key
    symbol_file = precoding_test_param.symbol_file
    symbol_key = precoding_test_param.symbol_key
    omega_file = precoding_test_param.omega_file
    omega_key = precoding_test_param.omega_key
    U_file = precoding_test_param.U_file
    U_key = precoding_test_param.U_key
    delta_file = precoding_test_param.delta_file
    delta_key = precoding_test_param.delta_key
    batch_size = precoding_test_param.batch_size
    snr_values = precoding_test_param.snr_list
    gpu_id = precoding_test_param.gpu_id
    begin_num = precoding_test_param.begin_num
    n_test = precoding_test_param.n_test
    txlen = precoding_test_param.txlen
    alpha_sim = precoding_test_param.alpha_sim
    pskOrder = precoding_test_param.pskOrder
    run_device = 'cuda:' + str(gpu_id) if gpu_id >= 0 else 'cpu'
    pt = 1

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # channel mat: [K, N, bs] -> transpose to [bs, K, N]
    channel_all = loadmat(in_folder + channel_file + ".mat")[channel_key]
    channel_all = channel_all.transpose(2, 0, 1)

    # symbol mat: [bs, K, L]
    symbol_all = loadmat(in_folder + symbol_file + ".mat")[symbol_key]

    # omega mat: [K, N, bs] -> transpose to [bs, K, N]
    omega_all = loadmat(in_folder + omega_file + ".mat")[omega_key]
    omega_all = omega_all.transpose(2, 0, 1)

    # U mat (paper U): [bs, K, N]
    U_all = loadmat(in_folder + U_file + ".mat")[U_key]

    data_length = U_all.shape[0]
    n_snr = len(snr_values)
    begin_num = begin_num * n_snr
    n_test = n_test * n_snr
    channel_all = np.repeat(channel_all, n_snr, axis=0)
    symbol_all = np.repeat(symbol_all, n_snr, axis=0)
    omega_all = np.repeat(omega_all, n_snr, axis=0)
    U_all = np.repeat(U_all, n_snr, axis=0)

    snr_list = np.tile(np.array(snr_values).reshape(n_snr, 1), (data_length, 1))

    # delta: expected [bs, 2K, L, n_snr]
    delta_label = loadmat(in_folder + delta_file + ".mat")[delta_key]
    delta_label = np.transpose(delta_label, (0, 3, 1, 2))  # bs, n_snr, 2K, L
    label_shape = list(delta_label.shape)
    delta_label = np.reshape(delta_label, (-1, label_shape[2], label_shape[3]))  # bs*n_snr, 2K, L
    delta_ch0 = delta_label[:, :label_shape[2] // 2, :]
    delta_ch1 = delta_label[:, label_shape[2] // 2:, :]
    delta_label = np.stack((delta_ch1, delta_ch0), axis=-1)  # bs*n_snr, K, L, 2

    h_test = torch.from_numpy(
        channel_all[begin_num:begin_num + n_test, :, :].copy()).to(torch.complex64)
    symbol_test = torch.from_numpy(
        symbol_all[begin_num:begin_num + n_test, :, :txlen].copy()).to(torch.complex64)
    omega_test = torch.from_numpy(
        omega_all[begin_num:begin_num + n_test, ...].copy()).to(torch.complex64)
    U_test = torch.from_numpy(
        U_all[begin_num:begin_num + n_test, ...].copy()).to(torch.complex64)
    snr_test = torch.from_numpy(
        snr_list[begin_num:begin_num + n_test, ...].copy()).to(torch.float32)
    delta_test = torch.from_numpy(
        delta_label[begin_num:begin_num + n_test, ...].copy()).to(torch.float32)

    del symbol_all, channel_all, omega_all, U_all, snr_list, delta_label

    test_dataset = DatasetFolder_6(h_test, symbol_test, omega_test, U_test, snr_test, delta_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del test_dataset, symbol_test, h_test, omega_test, U_test, delta_test
    gc.collect()

    model_rslp_b = model_rslp_b.to(run_device)
    model_rslp_b.eval()
    lossfunction = nn.MSELoss()

    delta_pred_all = []
    psi_pred_list = []
    avg_test_mse = 0.0

    with torch.no_grad():
        for h_test, tx_test, omega_test, U_test, snr_test, delta_test in test_loader:
            h_test = h_test.to(run_device)
            tx_test = tx_test.to(run_device)
            omega_test = torch.real(omega_test.to(run_device))
            U_test = U_test.to(run_device)
            snr_test = snr_test.to(run_device)
            delta_test = delta_test.to(run_device)
            cur_bs = h_test.size(0)

            sigma_test = 1 / (10 ** (snr_test / 10))
            alpha_test = (torch.ones(cur_bs, txlen) * alpha_sim).to(run_device)

            upsilon, psi_pred = get_RMMSE_kkt_Upsilon(
                h_test, tx_test, alpha_test, omega_test, sigma_test,
                model_rslp_a, U_test, pt=1, return_psi=True)
            psi_pred_list.append(psi_pred.detach().cpu().numpy())

            mat, vec = rslpn_kkt_features_from_upsilon(upsilon, tx_test, pskOrder=pskOrder)
            delta_pred = model_rslp_b(mat, vec)
            delta_pred_all.append(delta_pred.detach().cpu().numpy())
            mseloss_test = lossfunction(delta_pred, delta_test)
            avg_test_mse = avg_test_mse + float(mseloss_test.detach().cpu().numpy()) * cur_bs

    avg_test_mse = avg_test_mse / n_test
    print('Test MSE:{mse:.6f}'.format(mse=avg_test_mse))

    delta_pred_all = np.concatenate(delta_pred_all, axis=0)
    psi_pred_list = np.concatenate(psi_pred_list, axis=0)

    delta_save_stem = 'delta_pred_all'
    psi_save_stem = 'psi_pred'
    for i in range(len(snr_values)):
        delta_pred_temp = delta_pred_all[i::len(snr_values), ...]
        savemat(
            out_folder + delta_save_stem + '_snr' + str(snr_values[i]) + '.mat',
            {delta_save_stem + '_snr' + str(snr_values[i]): delta_pred_temp})
        psi_pred_temp = psi_pred_list[i::len(snr_values), ...]
        savemat(
            out_folder + psi_save_stem + '_snr' + str(snr_values[i]) + '.mat',
            {psi_save_stem + '_snr' + str(snr_values[i]): psi_pred_temp})


def RSLPN_B_train_supervised(precoding_train_param):
    """
    Train RSLPN-B under imperfect CSI (RCIMMSE / channel-aging setting).

    Supervised learning of perturbation factors ``delta`` for robust SLP.
    Uses a frozen pretrained RSLPN-A to build robust KKT features, then
    trains RSLPN-B (same architecture as SLPN) with MSE against ``delta`` labels.
    """

    # load parameters from PrecodingTrainParam_RSLPN_B
    model_rslp_b = precoding_train_param.model_rslp_b
    model_rslp_a = precoding_train_param.model_rslp_a
    net_name = precoding_train_param.net_name
    epoch_num = precoding_train_param.epoch_num
    lr_step_size = precoding_train_param.lr_step_size
    in_folder = precoding_train_param.in_folder
    out_folder = precoding_train_param.out_folder
    channel_file = precoding_train_param.channel_file
    learn_rate = precoding_train_param.learn_rate
    init_type = precoding_train_param.init_type
    batch_size = precoding_train_param.batch_size
    snr_values = precoding_train_param.snr_list
    gpu_id = precoding_train_param.gpu_id
    channel_key = precoding_train_param.channel_key
    symbol_file = precoding_train_param.symbol_file
    symbol_key = precoding_train_param.symbol_key
    omega_file = precoding_train_param.omega_file
    omega_key = precoding_train_param.omega_key
    U_file = precoding_train_param.U_file
    U_key = precoding_train_param.U_key
    delta_file = precoding_train_param.delta_file
    delta_key = precoding_train_param.delta_key
    lr_gamma = precoding_train_param.lr_gamma
    txlen = precoding_train_param.txlen
    n_train = precoding_train_param.n_train
    n_test = precoding_train_param.n_test
    alpha_sim = precoding_train_param.alpha_sim
    pskOrder = precoding_train_param.pskOrder
    if gpu_id>=0:
        run_device = 'cuda:'+str(gpu_id)
    else:
        run_device = 'cpu'


    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Initialize network parameters
    if init_type != 'noinit':
        init_func.init_weights(model_rslp_b, init_type=init_type)

    # load channel mat: [K, N, bs] -> transpose to [bs, K, N]
    channel_all = loadmat(in_folder + channel_file + ".mat")  
    channel_all = channel_all[channel_key]
    channel_all = channel_all.transpose(2,0,1) # BS K N

    # symbol mat: [bs, K, L]
    symbol_all = loadmat(in_folder + symbol_file + ".mat") 
    symbol_all = symbol_all[symbol_key] # BS K L

    # omega mat: [K, N, bs] -> transpose to [bs, K, N]
    omega_all = loadmat(in_folder + omega_file + ".mat")  
    omega_all = omega_all[omega_key]
    omega_all = omega_all.transpose(2, 0, 1) # BS K N

    # U mat (paper U): [bs, K, N]
    U_all = loadmat(in_folder + U_file + ".mat")  # BS K N
    U_all = U_all[U_key]


    data_length=U_all.shape[0]
    n_snr=len(snr_values)
    n_train=n_train*n_snr
    n_test=n_test*n_snr
    channel_all = np.repeat(channel_all, n_snr, axis=0)
    symbol_all = np.repeat(symbol_all, n_snr, axis=0)
    omega_all = np.repeat(omega_all, n_snr, axis=0)
    U_all = np.repeat(U_all, n_snr, axis=0)

    snr_list=np.array(snr_values).reshape(n_snr,1)
    snr_list=np.tile(snr_list,(data_length,1))

    # delta: expected [bs, 2K, L, n_snr]
    delta_label = loadmat(in_folder + delta_file + ".mat")[delta_key]
    delta_label = np.transpose(delta_label, (0, 3, 1, 2))  # bs, n_snr, 2K, L
    label_shape = list(delta_label.shape)  # bs, n_snr, 2K, L
    delta_label = np.reshape(delta_label, (-1, label_shape[2], label_shape[3]))  # bs*n_snr, 2K, L
    delta_ch0 = delta_label[:, :label_shape[2] // 2, :]
    delta_ch1 = delta_label[:, label_shape[2] // 2:, :]
    delta_label = np.stack((delta_ch1, delta_ch0), axis=-1)  # bs*n_snr, K, L, 2

    start = 0
    h_train = torch.from_numpy(channel_all[start:(start + n_train), :, :].copy()).to(torch.complex64)
    h_test = torch.from_numpy(channel_all[(start + n_train):(start + n_train)+n_test, :, :].copy()).to(torch.complex64)
    symbol_train = torch.from_numpy(symbol_all[start:(start + n_train), :, :txlen].copy()).to(torch.complex64)
    symbol_test = torch.from_numpy(symbol_all[(start + n_train):(start + n_train)+n_test, :, :txlen].copy()).to(torch.complex64)
    omega_train = torch.from_numpy(omega_all[start:(start + n_train), ...].copy()).to(torch.complex64)
    omega_test = torch.from_numpy(omega_all[(start + n_train):(start + n_train) + n_test, ...].copy()).to(torch.complex64)
    U_train = torch.from_numpy(U_all[start:(start + n_train), ...].copy()).to(torch.complex64)
    U_test = torch.from_numpy(U_all[(start + n_train):(start + n_train) + n_test, ...].copy()).to(torch.complex64)
    snr_train = torch.from_numpy(snr_list[start:(start + n_train), ...].copy()).to(torch.float32)
    snr_test = torch.from_numpy(snr_list[(start + n_train):(start + n_train) + n_test, ...].copy()).to(torch.float32)
    delta_train = torch.from_numpy(delta_label[start:(start + n_train), ...].copy()).to(torch.float32)
    delta_test = torch.from_numpy(delta_label[(start + n_train):(start + n_train) + n_test, ...].copy()).to(torch.float32)

    del symbol_all,channel_all,omega_all,U_all,snr_list,delta_label

    train_dataset = DatasetFolder_6(h_train,symbol_train,omega_train,U_train,snr_train,delta_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    del train_dataset,symbol_train,h_train,omega_train,U_train,delta_train
    gc.collect()
    test_dataset = DatasetFolder_6(h_test,symbol_test,omega_test,U_test,snr_test,delta_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del test_dataset,symbol_test,h_test,omega_test,U_test,delta_test
    gc.collect()

    optimizer = torch.optim.Adam([{'params': model_rslp_b.parameters()}], lr=learn_rate, weight_decay=0)
    model_rslp_b = model_rslp_b.to(run_device)
    lr_manager = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma, last_epoch=-1)

    lossfunction=nn.MSELoss()

    for epoch in range(epoch_num):
        start_time = time.time()
        avg_mse_loss = 0
        model_rslp_b.train()
        for h_train, tx_train, omega_train, U_train, snr_train, delta_train in train_loader:
            h_train = h_train.to(run_device)
            tx_train = tx_train.to(run_device)
            omega_train=omega_train.to(run_device)
            U_train=U_train.to(run_device)
            snr_train=snr_train.to(run_device)
            omega_train=torch.real(omega_train)
            delta_train=delta_train.to(run_device)

            batch_size=h_train.size(0)
            sigma_train = (1 / (10 ** (snr_train / 10)))
            alpha_train = torch.ones(batch_size, txlen)*alpha_sim
            alpha_train=alpha_train.to(run_device)

            with torch.no_grad():
                upsilon = get_RMMSE_kkt_Upsilon(h_train, tx_train, alpha_train, omega_train, sigma_train, model_rslp_a, U_train, pt=1)  # bs L 2K 2K
            mat, vec = rslpn_kkt_features_from_upsilon(upsilon, tx_train, pskOrder=pskOrder)

            delta_pred = model_rslp_b(mat, vec)
            mseloss = lossfunction(delta_pred,delta_train)
            avg_mse_loss = avg_mse_loss + float(mseloss.detach().cpu().numpy()) * h_train.size(0)

            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()

        avg_mse_loss = avg_mse_loss / (n_train)
        lr_manager.step()
        time_cost = time.time() - start_time
        current_lr = lr_manager.get_last_lr()[0]
        print(
            'Epoch:[{0}]\t'
            'Train MSE:{train_mse:.6f}\t'
            'Time:{time:.1f}secs\t'
            'lr:{lr:.8f}'.format(
                epoch, train_mse=avg_mse_loss, time=time_cost, lr=current_lr))
        with open(out_folder + 'loss_history.txt', 'a') as file:
            file.write(
                f'{epoch:<5} {avg_mse_loss:.6f} {time_cost:.4f} {current_lr:.8f}\n')
    torch.save(
        {'state_dict': model_rslp_b.state_dict()},
        out_folder + net_name + '.pth.tar')
    print('Training is finished!')

    model_rslp_b.eval()
    avg_test_mse = 0.0
    with torch.no_grad():
        for h_test, tx_test, omega_test, U_test, snr_test, delta_test in test_loader:
            h_test = h_test.to(run_device)
            tx_test = tx_test.to(run_device)
            omega_test = torch.real(omega_test.to(run_device))
            U_test = U_test.to(run_device)
            snr_test = snr_test.to(run_device)
            delta_test = delta_test.to(run_device)
            cur_bs = h_test.size(0)

            sigma_test = (1 / (10 ** (snr_test / 10)))
            alpha_test = (torch.ones(cur_bs, txlen) * alpha_sim).to(run_device)

            upsilon = get_RMMSE_kkt_Upsilon(
                h_test, tx_test, alpha_test, omega_test, sigma_test,
                model_rslp_a, U_test, pt=1)
            mat, vec = rslpn_kkt_features_from_upsilon(upsilon, tx_test, pskOrder=pskOrder)

            delta_pred = model_rslp_b(mat, vec)
            mseloss_test = lossfunction(delta_pred, delta_test)
            avg_test_mse = avg_test_mse + float(mseloss_test.detach().cpu().numpy()) * cur_bs

    avg_test_mse = avg_test_mse / n_test
    print('Test MSE:{mse:.6f}'.format(mse=avg_test_mse))
    with open(out_folder + 'test_mse.txt', 'w') as file:
        file.write(f'{avg_test_mse:.8f}\n')


def RSLPN_A_train_supervised(precoding_train_param):
    """
    Train RSLPN-A under imperfect CSI (RCIMMSE / channel-aging setting).

    Supervised learning of the auxiliary variable ``psi``
    used by the robust closed-form SLP solution. First stage of the two-stage
    RSLP pipeline; the resulting checkpoint is later frozen when training RSLPN-B.
    """

    # load parameters from PrecodingTrainParam_RSLPN_A
    model_rslp_a = precoding_train_param.model_rslp_a
    net_name = precoding_train_param.net_name
    epoch_num = precoding_train_param.epoch_num
    lr_step_size = precoding_train_param.lr_step_size
    in_folder = precoding_train_param.in_folder
    out_folder = precoding_train_param.out_folder
    channel_file = precoding_train_param.channel_file
    learn_rate = precoding_train_param.learn_rate
    init_type = precoding_train_param.init_type
    batch_size = precoding_train_param.batch_size
    snr_values = precoding_train_param.snr_list
    gpu_id = precoding_train_param.gpu_id
    channel_key = precoding_train_param.channel_key
    symbol_file = precoding_train_param.symbol_file
    symbol_key = precoding_train_param.symbol_key
    U_file = precoding_train_param.U_file
    U_key = precoding_train_param.U_key
    psi_file = precoding_train_param.psi_file
    psi_key = precoding_train_param.psi_key
    lr_gamma = precoding_train_param.lr_gamma
    txlen = precoding_train_param.txlen
    n_train = precoding_train_param.n_train
    n_test = precoding_train_param.n_test
    alpha_sim = precoding_train_param.alpha_sim
    if gpu_id>=0:
        run_device = 'cuda:'+str(gpu_id)
    else:
        run_device = 'cpu'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Initialize network parameters
    if init_type != 'noinit':
        init_func.init_weights(model_rslp_a, init_type=init_type)

    # load channel mat: [K, N, bs] -> transpose to [bs, K, N]
    channel_all = loadmat(in_folder + channel_file + ".mat")  
    channel_all = channel_all[channel_key]
    channel_all = channel_all.transpose(2,0,1) # BS K N

    # symbol mat: [bs, K, L]
    symbol_all = loadmat(in_folder + symbol_file + ".mat") 
    symbol_all = symbol_all[symbol_key] # BS K L

    # U mat (paper U): [bs, K, N]
    U_all = loadmat(in_folder + U_file + ".mat")  
    U_all = U_all[U_key] # BS K N

    # psi mat: [bs, K, L, n_snr] -> transpose to [bs, n_snr, K, L], then reshape to [bs*n_snr, K, L]
    psi_all=loadmat(in_folder+psi_file+".mat")
    psi_all=psi_all[psi_key]

    psi_all=psi_all.transpose(0,3,1,2)
    bs,f,k,l=psi_all.shape
    psi_all=psi_all.reshape(bs*f,k,l)


    data_length=bs
    n_snr=len(snr_values)
    n_train=n_train*n_snr
    n_test=n_test*n_snr
    channel_all=channel_all[:bs,...]
    symbol_all = symbol_all[:bs, ...]
    U_all = U_all[:bs, ...]
    channel_all = np.repeat(channel_all, n_snr, axis=0)
    symbol_all = np.repeat(symbol_all, n_snr, axis=0)
    U_all = np.repeat(U_all, n_snr, axis=0)

    snr_list=np.array(snr_values).reshape(n_snr,1)
    snr_list=np.tile(snr_list,(data_length,1))

    start = 0
    h_train = torch.from_numpy(channel_all[start:(start + n_train), :, :].copy()).to(torch.complex64)
    h_test = torch.from_numpy(channel_all[(start + n_train):(start + n_train)+n_test, :, :].copy()).to(torch.complex64)
    symbol_train = torch.from_numpy(symbol_all[start:(start + n_train), :, :txlen].copy()).to(torch.complex64)
    symbol_test = torch.from_numpy(symbol_all[(start + n_train):(start + n_train)+n_test, :, :txlen].copy()).to(torch.complex64)
    U_train = torch.from_numpy(U_all[start:(start + n_train), ...].copy()).to(torch.complex64)
    U_test = torch.from_numpy(U_all[(start + n_train):(start + n_train) + n_test, ...].copy()).to(torch.complex64)
    snr_train = torch.from_numpy(snr_list[start:(start + n_train), ...].copy()).to(torch.float32)
    snr_test = torch.from_numpy(snr_list[(start + n_train):(start + n_train) + n_test, ...].copy()).to(torch.float32)
    psi_train = torch.from_numpy(psi_all[start:(start + n_train), ...].copy()).to(torch.float32)
    psi_test = torch.from_numpy(psi_all[(start + n_train):(start + n_train) + n_test, ...].copy()).to(torch.float32)

    del symbol_all,channel_all,U_all,snr_list,psi_all

    train_dataset = DatasetFolder_5(h_train, symbol_train, U_train, snr_train, psi_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    del train_dataset,symbol_train,h_train,U_train,psi_train
    gc.collect()
    test_dataset = DatasetFolder_5(h_test, symbol_test, U_test, snr_test, psi_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del test_dataset,symbol_test,h_test,U_test,psi_test
    gc.collect()

    optimizer = torch.optim.Adam([{'params': model_rslp_a.parameters()}], lr=learn_rate, weight_decay=0)
    model_rslp_a = model_rslp_a.to(run_device)
    lr_manager = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma, last_epoch=-1)

    lossfunction=nn.MSELoss()

    for epoch in range(epoch_num):
        start_time = time.time()
        avg_mse_loss = 0
        model_rslp_a.train()
        for h_train, tx_train, U_train, snr_train, psi_train in train_loader:
            h_train = h_train.to(run_device)
            tx_train = tx_train.to(run_device)
            U_train = U_train.to(run_device)
            snr_train = snr_train.to(run_device)
            psi_train = psi_train.to(run_device)

            batch_size = h_train.size(0)
            sigma_train = (1 / (10 ** (snr_train / 10)))
            alpha_train = (torch.ones(batch_size, txlen) * alpha_sim).to(run_device)
            psi_pred = model_rslp_a(h_train, tx_train, sigma_train, alpha_train, U_train)
            mseloss = lossfunction(psi_pred, psi_train)
            avg_mse_loss = avg_mse_loss + float(mseloss.detach().cpu().numpy()) * h_train.size(0)

            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()

        avg_mse_loss = avg_mse_loss / (n_train)
        lr_manager.step()
        time_cost = time.time() - start_time
        current_lr = lr_manager.get_last_lr()[0]
        print(
            'Epoch:[{0}]\t'
            'Train MSE:{train_mse:.6f}\t'
            'Time:{time:.1f}secs\t'
            'lr:{lr:.8f}'.format(
                epoch, train_mse=avg_mse_loss, time=time_cost, lr=current_lr))
        with open(out_folder + 'loss_history.txt', 'a') as file:
            file.write(
                f'{epoch:<5} {avg_mse_loss:.6f} {time_cost:.4f} {current_lr:.8f}\n')
    torch.save(
        {'state_dict': model_rslp_a.state_dict()},
        out_folder + net_name + '.pth.tar')
    print('Training is finished!')

    model_rslp_a.eval()
    avg_test_mse = 0.0
    with torch.no_grad():
        for h_test, tx_test, U_test, snr_test, psi_test in test_loader:
            h_test = h_test.to(run_device)
            tx_test = tx_test.to(run_device)
            U_test = U_test.to(run_device)
            snr_test = snr_test.to(run_device)
            psi_test = psi_test.to(run_device)
            cur_bs = h_test.size(0)

            sigma_test = (1 / (10 ** (snr_test / 10)))
            alpha_test = (torch.ones(cur_bs, txlen) * alpha_sim).to(run_device)

            psi_pred = model_rslp_a(h_test, tx_test, sigma_test, alpha_test, U_test)
            batch_mse = lossfunction(psi_pred, psi_test)
            avg_test_mse = avg_test_mse + float(batch_mse.detach().cpu().numpy()) * cur_bs

    avg_test_mse = avg_test_mse / n_test
    print('Test MSE:{mse:.6f}'.format(mse=avg_test_mse))
    with open(out_folder + 'test_mse.txt', 'w') as file:
        file.write(f'{avg_test_mse:.8f}\n')


def CIMMSE_SLP_train_supervised(precoding_train_param):
    """
    Train SLPN for CIMMSE-DL under perfect CSI.

    Supervised learning of perturbation factors ``delta`` for the CIMMSE
    (CI-MMSE) symbol-level precoding problem. Shared by QPSK and 16QAM
    entry scripts; constellation only affects CIR threshold construction.
    """
    model = precoding_train_param.model
    net_name = precoding_train_param.net_name
    epoch_num = precoding_train_param.epoch_num
    lr_step_size = precoding_train_param.lr_step_size
    in_folder = precoding_train_param.in_folder
    out_folder = precoding_train_param.out_folder
    learn_rate = precoding_train_param.learn_rate
    init_type = precoding_train_param.init_type
    batch_size = precoding_train_param.batch_size
    snr_list = precoding_train_param.snr_list
    gpu_id = precoding_train_param.gpu_id
    lr_gamma = precoding_train_param.lr_gamma
    run_device = 'cuda:' + str(gpu_id) if gpu_id >= 0 else 'cpu'

    upsilon_file = precoding_train_param.upsilon_file
    upsilon_key = precoding_train_param.upsilon_key
    symbol_file = precoding_train_param.symbol_file
    symbol_key = precoding_train_param.symbol_key
    delta_file = precoding_train_param.delta_file
    delta_key = precoding_train_param.delta_key
    constellation = precoding_train_param.constellation
    pskOrder = precoding_train_param.pskOrder
    n_train = precoding_train_param.n_train
    n_test = precoding_train_param.n_test

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if init_type != 'noinit':
        init_func.init_weights(model, init_type=init_type)

    snrNum = len(snr_list)

    # upsilon mat: [bs, K, K, n_snr] complex; slice SNR then reshape to [bs*n_snr, K, K]
    upsilon_cimmse = loadmat(in_folder + upsilon_file + ".mat")
    upsilon_cimmse = upsilon_cimmse[upsilon_key]
    upsilon_cimmse = upsilon_cimmse[:, :, :, :snrNum]
    upsilon_cimmse = np.transpose(upsilon_cimmse, (0, 3, 1, 2))  # bs, n_snr, K, K
    upsilon_dim = list(upsilon_cimmse.shape)
    upsilon_cimmse = np.reshape(upsilon_cimmse, (-1, upsilon_dim[2], upsilon_dim[3]))  # bs*n_snr, K, K

    # symbol mat: [bs, K, L]; later repeated along SNR to [bs*n_snr, K, L]
    symbol_all = loadmat(in_folder + symbol_file + ".mat")
    symbol_all = symbol_all[symbol_key]

    # delta mat: [bs, n_snr, 2K, L]; reshape/split to [bs*n_snr, K, L, 2]
    delta_cimmse = loadmat(in_folder + delta_file + ".mat")[delta_key]
    delta_cimmse = delta_cimmse[:, :snrNum, :, :]
    _, _, k2, l_dim = delta_cimmse.shape
    delta_cimmse = delta_cimmse.reshape(-1, k2, l_dim)
    delta_cimmse_1 = delta_cimmse[:, :k2 // 2, :]
    delta_cimmse_2 = delta_cimmse[:, k2 // 2:, :]
    delta_cimmse_split = np.stack((delta_cimmse_1, delta_cimmse_2), axis=-1)  # bs*n_snr, K, L, 2

    n_snr = len(snr_list)
    symbol_all = np.repeat(symbol_all, n_snr, axis=0)

    n_train = n_train * n_snr
    n_test = n_test * n_snr

    _, _, knum = upsilon_cimmse.shape
    square_sum = np.sum(np.abs(upsilon_cimmse) ** 2, axis=(1, 2), keepdims=True)
    upsilon_cimmse = upsilon_cimmse / np.sqrt(square_sum)
    upsilon_cimmse = upsilon_cimmse * knum

    start = 0
    upsilon_train = torch.from_numpy(upsilon_cimmse[start:(start + n_train), :, :].copy()).to(torch.complex64)
    upsilon_test = torch.from_numpy(
        upsilon_cimmse[(start + n_train):(start + n_train) + n_test, :, :].copy()).to(torch.complex64)
    symbol_train = torch.from_numpy(symbol_all[start:(start + n_train), :, :].copy()).to(torch.complex64)
    symbol_test = torch.from_numpy(
        symbol_all[(start + n_train):(start + n_train) + n_test, :, :].copy()).to(torch.complex64)
    delta_train = torch.from_numpy(
        delta_cimmse_split[start:(start + n_train), :, :].copy()).to(torch.complex64)
    delta_test = torch.from_numpy(
        delta_cimmse_split[(start + n_train):(start + n_train) + n_test, :, :].copy()).to(torch.complex64)
    del upsilon_cimmse, symbol_all, delta_cimmse_split

    train_dataset = DatasetFolder_3(upsilon_train, symbol_train, delta_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    del train_dataset, upsilon_train, symbol_train, delta_train
    gc.collect()
    test_dataset = DatasetFolder_3(upsilon_test, symbol_test, delta_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del test_dataset, upsilon_test, symbol_test, delta_test
    gc.collect()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learn_rate, weight_decay=0)
    model = model.to(run_device)
    lr_manager = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma, last_epoch=-1)

    lossfunction = nn.MSELoss()

    for epoch in range(epoch_num):
        start = time.time()
        avg_mse_loss = 0
        model.train()
        for upsilon_train, tx_train, delta_train in train_loader:
            upsilon_train = upsilon_train.to(run_device)
            tx_train = tx_train.to(run_device)
            delta_train = delta_train.to(run_device)

            threshold_upper, threshold_lower = slp_cir_thresholds(
                tx_train, constellation=constellation, pskOrder=pskOrder)
            mat, vec = slp_kkt_features(upsilon_train, tx_train, threshold_upper, threshold_lower)
            delta_pred = model(mat, vec)

            delta_train = torch.real(delta_train)
            mseloss = lossfunction(delta_pred, delta_train)
            avg_mse_loss = avg_mse_loss + float(mseloss.detach().cpu().numpy()) * upsilon_train.size(0)

            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()
        avg_mse_loss = avg_mse_loss / n_train
        lr_manager.step()
        time_cost = time.time() - start
        current_lr = lr_manager.get_last_lr()[0]
        print(
            'Epoch:[{0}]\t'
            'Train MSE:{train_mse:.6f}\t'
            'Time:{time:.1f}secs\t'
            'lr:{lr:.8f}'.format(
                epoch, train_mse=avg_mse_loss, time=time_cost, lr=current_lr))
        with open(out_folder + 'loss_history.txt', 'a') as file:
            file.write(
                f'{epoch:<5} {avg_mse_loss:.6f} {time_cost:.4f} {current_lr:.8f}\n')
    torch.save(
        {'state_dict': model.state_dict()},
        out_folder + net_name + '.pth.tar')
    print('Training is finished!')

    model.eval()
    avg_test_mse = 0.0
    with torch.no_grad():
        for upsilon_test, tx_test, delta_test in test_loader:
            upsilon_test = upsilon_test.to(run_device)
            tx_test = tx_test.to(run_device)
            delta_test = delta_test.to(run_device)
            delta_test = torch.real(delta_test)

            threshold_upper, threshold_lower = slp_cir_thresholds(
                tx_test, constellation=constellation, pskOrder=pskOrder)
            mat, vec = slp_kkt_features(upsilon_test, tx_test, threshold_upper, threshold_lower)
            delta_pred = model(mat, vec)

            mselosstest = lossfunction(delta_pred, delta_test)
            avg_test_mse = avg_test_mse + float(mselosstest.detach().cpu().numpy()) * upsilon_test.size(0)

    avg_test_mse = avg_test_mse / n_test
    print('Test MSE:{mse:.6f}'.format(mse=avg_test_mse))
    with open(out_folder + 'test_mse.txt', 'w') as file:
        file.write(f'{avg_test_mse:.8f}\n')


def CIZF_SLP_train_supervised(precoding_train_param):
    """
    Train SLPN for CIZF-DL under perfect CSI.

    Supervised learning of perturbation factors ``delta`` for the CIZF
    (CI zero-forcing / SINR-balancing) symbol-level precoding problem.
    Shared by QPSK and 16QAM entry scripts; constellation only affects
    CIR threshold construction.
    """
    model = precoding_train_param.model
    net_name = precoding_train_param.net_name
    epoch_num = precoding_train_param.epoch_num
    lr_step_size = precoding_train_param.lr_step_size
    in_folder = precoding_train_param.in_folder
    out_folder = precoding_train_param.out_folder
    learn_rate = precoding_train_param.learn_rate
    init_type = precoding_train_param.init_type
    batch_size = precoding_train_param.batch_size
    gpu_id = precoding_train_param.gpu_id
    lr_gamma = precoding_train_param.lr_gamma
    run_device = 'cuda:' + str(gpu_id) if gpu_id >= 0 else 'cpu'

    upsilon_file = precoding_train_param.upsilon_file
    upsilon_key = precoding_train_param.upsilon_key
    symbol_file = precoding_train_param.symbol_file
    symbol_key = precoding_train_param.symbol_key
    delta_file = precoding_train_param.delta_file
    delta_key = precoding_train_param.delta_key
    constellation = precoding_train_param.constellation
    pskOrder = precoding_train_param.pskOrder
    n_train = precoding_train_param.n_train
    n_test = precoding_train_param.n_test

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if init_type != 'noinit':
        init_func.init_weights(model, init_type=init_type)

    # upsilon mat: [bs, K, K] complex
    upsilon_cizf = loadmat(in_folder + upsilon_file + ".mat")
    upsilon_cizf = upsilon_cizf[upsilon_key]
    # symbol mat: [bs, K, L]
    symbol_all = loadmat(in_folder + symbol_file + ".mat")
    symbol_all = symbol_all[symbol_key]
    # delta mat: [bs, 2K, L]; split to [bs, K, L, 2]
    delta_cizf = loadmat(in_folder + delta_file + ".mat")
    delta_cizf = delta_cizf[delta_key]  # bs, 2K, L
    bs, k2, l = delta_cizf.shape
    delta_cizf2 = delta_cizf[:, :k2 // 2, :]
    delta_cizf1 = delta_cizf[:, k2 // 2:, :]
    delta_cizf_split = np.zeros((bs, k2 // 2, l, 2))
    delta_cizf_split[:, :, :, 0] = delta_cizf1
    delta_cizf_split[:, :, :, 1] = delta_cizf2

    _, _, knum = upsilon_cizf.shape
    square_sum = np.sum(np.abs(upsilon_cizf) ** 2, axis=(1, 2), keepdims=True)
    upsilon_cizf = upsilon_cizf / np.sqrt(square_sum)
    upsilon_cizf = upsilon_cizf * knum

    start = 0
    upsilon_train = torch.from_numpy(upsilon_cizf[start:(start + n_train), :, :].copy()).to(torch.complex64)
    upsilon_test = torch.from_numpy(
        upsilon_cizf[(start + n_train):(start + n_train) + n_test, :, :].copy()).to(torch.complex64)
    symbol_train = torch.from_numpy(symbol_all[start:(start + n_train), :, :].copy()).to(torch.complex64)
    symbol_test = torch.from_numpy(
        symbol_all[(start + n_train):(start + n_train) + n_test, :, :].copy()).to(torch.complex64)
    delta_train = torch.from_numpy(
        delta_cizf_split[start:(start + n_train), :, :].copy()).to(torch.complex64)
    delta_test = torch.from_numpy(
        delta_cizf_split[(start + n_train):(start + n_train) + n_test, :, :].copy()).to(torch.complex64)
    del upsilon_cizf, symbol_all, delta_cizf_split

    train_dataset = DatasetFolder_3(upsilon_train, symbol_train, delta_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    del train_dataset, upsilon_train, symbol_train, delta_train
    gc.collect()
    test_dataset = DatasetFolder_3(upsilon_test, symbol_test, delta_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del test_dataset, upsilon_test, symbol_test, delta_test
    gc.collect()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learn_rate, weight_decay=0)
    model = model.to(run_device)
    lr_manager = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma, last_epoch=-1)

    lossfunction = nn.MSELoss()

    for epoch in range(epoch_num):
        start = time.time()
        avg_mse_loss = 0

        model.train()
        for upsilon_train, tx_train, delta_train in train_loader:
            upsilon_train = upsilon_train.to(run_device)
            tx_train = tx_train.to(run_device)
            delta_train = delta_train.to(run_device)

            threshold_upper, threshold_lower = slp_cir_thresholds(
                tx_train, constellation=constellation, pskOrder=pskOrder)
            mat, vec = slp_kkt_features(upsilon_train, tx_train, threshold_upper, threshold_lower)
            delta_pred = model(mat, vec)

            delta_train = torch.real(delta_train)
            mseloss = lossfunction(delta_pred, delta_train)
            avg_mse_loss = avg_mse_loss + float(mseloss.detach().cpu().numpy()) * upsilon_train.size(0)

            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()

        avg_mse_loss = avg_mse_loss / n_train
        lr_manager.step()
        time_cost = time.time() - start
        current_lr = lr_manager.get_last_lr()[0]
        print(
            'Epoch:[{0}]\t'
            'Train MSE:{train_mse:.6f}\t'
            'Time:{time:.1f}secs\t'
            'lr:{lr:.8f}'.format(
                epoch, train_mse=avg_mse_loss, time=time_cost, lr=current_lr))
        with open(out_folder + 'loss_history.txt', 'a') as file:
            file.write(
                f'{epoch:<5} {avg_mse_loss:.6f} {time_cost:.4f} {current_lr:.8f}\n')

    torch.save(
        {'state_dict': model.state_dict()},
        out_folder + net_name + '.pth.tar')
    print('Training is finished!')

    model.eval()
    avg_test_mse = 0.0
    with torch.no_grad():
        for upsilon_test, tx_test, delta_test in test_loader:
            upsilon_test = upsilon_test.to(run_device)
            tx_test = tx_test.to(run_device)
            delta_test = delta_test.to(run_device)
            delta_test = torch.real(delta_test)

            threshold_upper, threshold_lower = slp_cir_thresholds(
                tx_test, constellation=constellation, pskOrder=pskOrder)
            mat, vec = slp_kkt_features(upsilon_test, tx_test, threshold_upper, threshold_lower)
            delta_pred = model(mat, vec)

            mselosstest = lossfunction(delta_pred, delta_test)
            avg_test_mse = avg_test_mse + float(mselosstest.detach().cpu().numpy()) * upsilon_test.size(0)

    avg_test_mse = avg_test_mse / n_test
    print('Test MSE:{mse:.6f}'.format(mse=avg_test_mse))
    with open(out_folder + 'test_mse.txt', 'w') as file:
        file.write(f'{avg_test_mse:.8f}\n')


def CIZF_SLP_test_supervised(precoding_test_param):
    """
    Test SLPN for CIZF-DL under perfect CSI.

    Loads a trained checkpoint and evaluates perturbation-factor prediction
    ``delta`` on the held-out split. Reports Test MSE and writes predictions
    to disk. Shared by QPSK and 16QAM entry scripts.
    """
    model = precoding_test_param.model
    net_name = precoding_test_param.net_name
    in_folder = precoding_test_param.in_folder
    out_folder = precoding_test_param.out_folder
    batch_size = precoding_test_param.batch_size
    gpu_id = precoding_test_param.gpu_id
    begin_num = precoding_test_param.begin_num
    n_test = precoding_test_param.n_test
    run_device = 'cuda:' + str(gpu_id) if gpu_id >= 0 else 'cpu'

    upsilon_file = precoding_test_param.upsilon_file
    upsilon_key = precoding_test_param.upsilon_key
    symbol_file = precoding_test_param.symbol_file
    symbol_key = precoding_test_param.symbol_key
    delta_file = precoding_test_param.delta_file
    delta_key = precoding_test_param.delta_key
    constellation = precoding_test_param.constellation
    pskOrder = precoding_test_param.pskOrder

    # symbol mat: [bs, K, L]
    symbol_all = loadmat(in_folder + symbol_file + ".mat")[symbol_key]
    # upsilon mat: [bs, K, K] complex
    upsilon_cizf = loadmat(in_folder + upsilon_file + ".mat")[upsilon_key]
    # delta mat: [bs, 2K, L]; split to [bs, K, L, 2]
    delta_cizf = loadmat(in_folder + delta_file + ".mat")[delta_key]  # bs, 2K, L
    bs, k2, l = delta_cizf.shape
    delta_cizf2 = delta_cizf[:, :k2 // 2, :]
    delta_cizf1 = delta_cizf[:, k2 // 2:, :]
    delta_cizf_split = np.zeros((bs, k2 // 2, l, 2))
    delta_cizf_split[:, :, :, 0] = delta_cizf1
    delta_cizf_split[:, :, :, 1] = delta_cizf2

    _, _, knum = upsilon_cizf.shape
    square_sum = np.sum(np.abs(upsilon_cizf) ** 2, axis=(1, 2), keepdims=True)
    upsilon_cizf = upsilon_cizf / np.sqrt(square_sum)
    upsilon_cizf = upsilon_cizf * knum

    upsilon_test = torch.from_numpy(
        upsilon_cizf[begin_num:begin_num + n_test, ...].copy()).to(torch.complex64).to(run_device)
    tx_test = torch.from_numpy(
        symbol_all[begin_num:begin_num + n_test, ...].copy()).to(torch.complex64).to(run_device)
    delta_label = torch.from_numpy(
        delta_cizf_split[begin_num:begin_num + n_test, ...].copy()).to(torch.complex64).to(run_device)

    model.load_state_dict(
        (torch.load(out_folder + net_name + '.pth.tar', map_location='cpu'))['state_dict'])
    model.to(run_device)
    model.eval()

    with torch.no_grad():
        threshold_upper, threshold_lower = slp_cir_thresholds(
            tx_test, constellation=constellation, pskOrder=pskOrder)
        mat, vec = slp_kkt_features(upsilon_test, tx_test, threshold_upper, threshold_lower)

        outputs = []
        for start in range(0, n_test, batch_size):
            end = start + batch_size
            outputs.append(model(mat[start:end], vec[start:end]))
        delta_pred = torch.cat(outputs, dim=0)

        delta_label = torch.real(delta_label)
        mse = torch.mean((delta_pred - delta_label) ** 2)
        print('Test MSE:{mse:.6f}'.format(mse=float(mse.detach().cpu().numpy())))

    delta_save_stem = 'delta_pred_cizf'
    savemat(
        out_folder + delta_save_stem + '.mat',
        {delta_save_stem: delta_pred.detach().cpu().numpy()})


def CIMMSE_SLP_test_supervised(precoding_test_param):
    """
    Test SLPN for CIMMSE-DL under perfect CSI.

    Loads a trained checkpoint and evaluates perturbation-factor prediction
    ``delta`` on the held-out split across SNR points. Reports Test MSE and
    writes predictions to disk. Shared by QPSK and 16QAM entry scripts.
    """
    model = precoding_test_param.model
    net_name = precoding_test_param.net_name
    in_folder = precoding_test_param.in_folder
    out_folder = precoding_test_param.out_folder
    snr_list = precoding_test_param.snr_list
    gpu_id = precoding_test_param.gpu_id
    begin_num = precoding_test_param.begin_num
    n_test = precoding_test_param.n_test
    run_device = 'cuda:' + str(gpu_id) if gpu_id >= 0 else 'cpu'

    upsilon_file = precoding_test_param.upsilon_file
    upsilon_key = precoding_test_param.upsilon_key
    symbol_file = precoding_test_param.symbol_file
    symbol_key = precoding_test_param.symbol_key
    delta_file = precoding_test_param.delta_file
    delta_key = precoding_test_param.delta_key
    constellation = precoding_test_param.constellation
    pskOrder = precoding_test_param.pskOrder

    # symbol mat: [bs, K, L]
    symbol_all = loadmat(in_folder + symbol_file + ".mat")[symbol_key]
    # upsilon mat: [bs, K, K, n_snr] complex; later indexed as [:, :, :, snr_idx]
    upsilon_cimmse = loadmat(in_folder + upsilon_file + ".mat")[upsilon_key]

    # delta mat: [bs, n_snr, 2K, L]; split/transpose to [bs, K, L, 2, n_snr]
    delta_cimmse = loadmat(in_folder + delta_file + ".mat")[delta_key]
    delta_cimmse = delta_cimmse[:, :len(snr_list), :, :]
    k2 = delta_cimmse.shape[2]
    delta_cimmse_2 = delta_cimmse[:, :, :k2 // 2, :]
    delta_cimmse_1 = delta_cimmse[:, :, k2 // 2:, :]
    delta_cimmse_split = np.stack((delta_cimmse_1, delta_cimmse_2), axis=-1)
    delta_cimmse_split = np.transpose(delta_cimmse_split, (0, 2, 3, 4, 1))  # bs, K, L, 2, n_snr

    upsilon_cimmse = upsilon_cimmse[begin_num:begin_num + n_test, ...]
    symbol_all = symbol_all[begin_num:begin_num + n_test, ...]
    delta_cimmse_split = delta_cimmse_split[begin_num:begin_num + n_test, ...]
    upsilon_cimmse_all = torch.from_numpy(upsilon_cimmse.copy()).to(torch.complex64).to(run_device)
    tx_test = torch.from_numpy(symbol_all.copy()).to(torch.complex64).to(run_device)
    delta_cimmse_split = torch.from_numpy(delta_cimmse_split.copy()).to(torch.float32).to(run_device)

    model.load_state_dict(
        (torch.load(out_folder + net_name + '.pth.tar', map_location='cpu'))['state_dict'])
    model.to(run_device)
    model.eval()

    delta_save_stem = 'delta_pred_cimmse'
    delta_test_save = {snr: [] for snr in snr_list}

    with torch.no_grad():
        for i, snr in enumerate(snr_list):
            upsilon_cimmse_db = upsilon_cimmse_all[:, :, :, i]
            delta_label = delta_cimmse_split[:, :, :, :, i]

            _, _, knum = upsilon_cimmse_db.size()
            square_sum = torch.sum(torch.abs(upsilon_cimmse_db) ** 2, dim=(1, 2), keepdim=True)
            upsilon_cimmse_db = upsilon_cimmse_db / torch.sqrt(square_sum)
            upsilon_cimmse_db = upsilon_cimmse_db * knum

            threshold_upper, threshold_lower = slp_cir_thresholds(
                tx_test, constellation=constellation, pskOrder=pskOrder)
            mat, vec = slp_kkt_features(
                upsilon_cimmse_db, tx_test, threshold_upper, threshold_lower)

            delta_pred = model(mat, vec)
            delta_pred = torch.flip(delta_pred, dims=[-1])

            mse = torch.mean((delta_pred - delta_label) ** 2)
            print('SNR:{snr}\tTest MSE:{mse:.6f}'.format(
                snr=snr, mse=float(mse.detach().cpu().numpy())))
            delta_test_save[snr].append(delta_pred.detach().cpu().numpy())

    for snr in snr_list:
        final_output = np.concatenate(delta_test_save[snr], axis=0)
        savemat(
            out_folder + delta_save_stem + '_snr' + str(snr) + '.mat',
            {delta_save_stem + '_snr' + str(snr): final_output})