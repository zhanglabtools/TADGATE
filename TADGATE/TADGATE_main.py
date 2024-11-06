
import time
import copy
import numpy as np
import pandas as pd
import torch
from . import TADGATE_utils as TL




def combine_key_to_build_full_matrix(res_all_record, range_l, mat_hic):
    """
    Combine the imputed Hi-C contact maps for each split window to build the full Hi-C contact map.
    :param res_all_record: dict, TADGATE results for each window.
    :param range_l: list, range of each split window.
    :param mat_hic: numpy array, Hi-C contact map.
    :return: mat_imputed_sym: numpy array, symmetrized imputed Hi-C contact map.
    """
    mat_rec = np.zeros([mat_hic.shape[0], mat_hic.shape[-1]])
    for m in range(len(res_all_record.keys())):
        key = list(res_all_record.keys())[m]
        rg_y1 = range_l[m][1]
        rg_y2 = range_l[m][2]
        rg_x1 = range_l[m][0]
        rg_x2 = range_l[m][3]
        mat_key = res_all_record[key]['mat_imputed_sym']
        mat_len = mat_key.shape[0]
        if key == 0:
            mat_rec[rg_y1:rg_y2, rg_x1:rg_x2] = mat_key[0: rg_y2, :]
        elif key != list(res_all_record.keys())[-1]:
            mat_rec[rg_y1:rg_y2, rg_x1:rg_x2] = mat_key[rg_y1 - rg_x1 : rg_y2 - rg_x1, :]
        elif key == list(res_all_record.keys())[-1]:
            mat_rec[rg_y1:rg_y2, rg_x1:rg_x2] = mat_key[len(mat_key) - (rg_y2 - rg_y1) : len(mat_key), :]
    mat_imputed_sym = mat_rec + mat_rec.T - np.diag(np.diag(mat_rec))
    return mat_imputed_sym


def TADGATE_specific_chr(mat_hic, length, bin_name_use, resolution, split_size, graph_radius, device,
                         layer_node1, layer_node2, lr, weight_decay, num_epoch, embed_attention,
                         weight_use = 'No', weight_range = 0, weight_rate = 0, batch_use = False, num_batch_x = 1,
                         auto_expand = False, impute_func = False, impute_range = 5, scale_f = 1,
                         diag_cut = 0, wd = 'all', save_model = False,
                         seed = 666, verbose=True, split_stop = ''):
    """
    Run TADGATE on a specific chromosome.
    :param mat_hic: numpy array, Hi-C contact map.
    :param length: int, Length of the chromosome, bin number.
    :param bin_name_use: list, List of bin names.
    :param resolution:  int, Resolution of Hi-C contact map.
    :param split_size:  int, Size of sub-matrices to split Hi-C contact maps into.
    :param graph_radius: int, Radius used to construct spatial network.
    :param device:  str, Device to run TADGATE on.
    :param layer_node1: int, number of nodes in first layer of autoencoder.
    :param layer_node2: int, number of nodes in second layer of autoencoder.
    :param lr:  float, Learning rate.
    :param weight_decay: float, Weight decay.
    :param num_epoch: int, Number of epochs to train TADGATE.
    :param embed_attention: bool, whether to use attention mechanism in the embedding layer (second layer).
    :param weight_use: str, Whether to use weight matrix or use different kind of weight. Choose from 'No', 'Fix', 'Auto'.
    'No' for no weight matrix, 'Fix' for fixed weight according to weight_rate, 'Auto' for auto-learned weight matrix.
    :param weight_range: int, the range near the diagonal to apply the weight matrix.
    :param weight_rate:  float, the weight rate for the fixed weight matrix.
    :param batch_use: bool, whether to use batch training.
    :param num_batch_x: int, number of batch for batch training.
    :param auto_expand:  bool, whether to expand the auto-learned weight rate.
    :param impute_func: bool, whether to impute region near the diagnal of Hi-C contact map.
    :param impute_range: int, the range near the diagonal to apply the impute function.
    :param scale_f: int, scale factor for node embedding scaling with dimension.
    :param diag_cut: int, number of diagonals to cut inner. If not zero, diag_cut to wd will be zero. If zero, will use wd to cut out
    :param wd:  int, number of diagonals to cut out, wd to window will be zero. window is determined by split_size and resolution.
                If 'all', no cut out will be used.
    :param seed: int, random seed.
    :param verbose: bool, whether to print the training process.
    :return: range_l: list, range of each split window;
             mat_split_all: dict, Hi-C contact map for each window;
             row_bad_all: dict, bad rows for each window;
             spatial_net_all: dict, spatial network for each window;
             mat_mask0: numpy array, mask matrix or weight matrix;
             res_all_record: dict, TADGATE results for each window. For each window, ‘loss’ for loss record,
             ‘model’ for trained model, ‘map_imputed’ for imputed Hi-C contact map,
             ‘mat_imputed_sym’ for symmetrized imputed Hi-C contact map,
             ‘bin_rep’ for bin representation, ‘attention_map’ for attention map.
    """
    res_all_record = {}
    window = split_size // resolution
    diag_cut_split = wd
    range_l, mat_split_all, row_bad_all= TL.get_matrix_split(mat_hic, length, window, bin_name_use, resolution,
                                                             diag_cut_split, cut = False)

    print('Hi-C map is splited into ' + str(len(range_l)) + ' sub-map')

    print('Build spatial network...')
    mat_split_all, spatial_net_all = TL.get_split_mat_spatial_network(mat_split_all, row_bad_all, graph_radius)

    print('Get mask-matrix or weight-matrix...')
    if weight_use == 'No':
        if wd == 'all':
            wd = window
        else:
            wd = wd
        mat_mask0 = TL.get_diagnal_near_mask(window, wd, diag_cut = diag_cut)
        mat_mask0 = TL.mat_mask_batch_to_device(mat_mask0, device, batch = 1)
    elif weight_use == 'Fix':
        weight = weight_rate
        mat_weight = np.ones([window, window])
        diag_indices = np.diag_indices(window)
        for k in range(np.max([0, diag_cut]), weight_range):
            diag_indices_k = (diag_indices[0][:window - k], diag_indices[1][k:])
            mat_weight[diag_indices_k] += weight
        mat_upper_indices = np.triu_indices(window, k=0)
        mat_weight_new = np.zeros([window, window])
        mat_weight_new[mat_upper_indices] = mat_weight[mat_upper_indices]
        mat_weight_new += mat_weight_new.T - np.diag(np.diag(mat_weight_new))
        mat_mask0 = mat_weight_new
    else:
        # Auto learn the weight matrix based on each diagonal mean of the Hi-C map
        diagonal_means = np.array([np.mean(np.diag(mat_hic, k)) for k in range(0, mat_hic.shape[1])])
        weights = diagonal_means / np.sum(diagonal_means)
        if auto_expand == True:
            while np.max(weights) < 1:
                weights = weights * 10
        mat_weight = np.ones([window, window])
        diag_indices = np.diag_indices(window)
        for k in range(np.max([0, diag_cut]), weight_range):
            diag_indices_k = (diag_indices[0][:window - k], diag_indices[1][k:])
            mat_weight[diag_indices_k] += (weights[k] + 1)
        mat_upper_indices = np.triu_indices(window, k=0)
        mat_weight_new = np.zeros([window, window])
        mat_weight_new[mat_upper_indices] = mat_weight[mat_upper_indices]
        mat_weight_new += mat_weight_new.T - np.diag(np.diag(mat_weight_new))
        mat_mask0 = mat_weight_new
    mat_mask0_copy = copy.deepcopy(mat_mask0)
    for i in range(len(range_l)):
        if split_stop != '':
            if i != split_stop:
                continue
        mat_mask0 = mat_mask0_copy
        res_all_record[i] = {}
        print('For No.' + str(i) + ' sub-map')
        ind = range_l[i]
        adata = mat_split_all[i]

        if impute_func == True:
            mat_part = copy.deepcopy(adata.X)
            mat_mask0 = TL.ignal_zero_pos_in_mask_mat(mat_part, mat_mask0, impute_range)

        batch_list = []
        mask_list = []
        if batch_use == False:
            pass
        else:
            adata.obs['X'] = adata.obsm['genome_order']
            Batch_list = TL.Batch_Data_SingleDim(adata, num_batch_x, spatial_key='X')
            for batch_adata in Batch_list:
                row_bad = TL.get_zero_row(batch_adata.X, method='row', resolution=resolution)
                if len(row_bad) >= batch_adata.X.shape[0] * 0.8:
                    continue
                Spatial_Net, Net_sparse = TL.build_spatial_net(batch_adata, row_bad, expand_num = graph_radius)
                batch_adata.uns['Spatial_Net'] = Spatial_Net
                data = TL.Transfer_pytorch_Data(batch_adata)
                batch_list.append(data)
                indices = np.where(np.isin(np.array(adata.obs_names), np.array(batch_adata.obs_names)))[0]
                mask_mat_batch = mat_mask0[indices, :]
                mask_list.append(mask_mat_batch)

        if weight_use == 'No':
            model, loss_record = TL.train_TADGATE(adata, mat_mask0, scale_f, layer_node1, layer_node2, lr, weight_decay,
                                                  num_epoch, device, embed_attention = embed_attention, batch_use = batch_use,
                                                  batch_list = batch_list, mask_list = mask_list,seed = seed, verbose=verbose)
        else:
            model, loss_record = TL.train_TADGATE_weight(adata, mat_mask0, scale_f, layer_node1, layer_node2, lr, weight_decay,
                                                         num_epoch, device, embed_attention = embed_attention, batch_use = batch_use,
                                                         batch_list = batch_list, mask_list = mask_list, seed = seed, verbose=verbose)

        mat_imputed, mat_imputed_sym, mat_rep, mat_att1 = TL.TADGATE_use(adata, model, device, scale_f,
                                                                         embed_attention, return_att = True)
        res_all_record[i]['loss'] = loss_record
        if save_model == True:
            res_all_record[i]['model'] = model
        #res_all_record[i]['mat_imputed'] = mat_imputed
        res_all_record[i]['mat_imputed_sym'] = mat_imputed_sym
        res_all_record[i]['bin_rep'] = mat_rep
        res_all_record[i]['attention_map'] = mat_att1
        torch.cuda.empty_cache()
    if len(res_all_record.keys()) > 1:
        mat_imputed_sym_full = combine_key_to_build_full_matrix(res_all_record, range_l, mat_hic)
        res_all_record['full_mat_imputed_sym'] = mat_imputed_sym_full
    return range_l, mat_split_all, row_bad_all, spatial_net_all, mat_mask0, res_all_record


def TADGATE_for_embedding(hic_all, chr_size, resolution, graph_radius, split_size,  device, layer_node1,
                          layer_node2, lr, weight_decay, num_epoch, embed_attention=False, weight_use = 'No', weight_range = 0,
                          weight_rate = 0, batch_use = False, num_batch_x= 1, auto_expand = False,
                          impute_func = False, impute_range = 5, CNN_impute = False,
                          RWR_impute = False,  target_chr_l = [], scale_f = 1, diag_cut = 0, wd = 'all',
                          save_model = False, seed = 666, verbose=True, split_stop = ''):
    """
    Run TADGATE on all chromosomes.
    :param hic_all:  dict, Hi-C contact maps for all chromosomes.
    :param chr_size:  dict, chromosome sizes.
    :param resolution:  int, Resolution of Hi-C contact map.
    :param graph_radius:  int, Radius used to construct spatial network.
    :param split_size:  int, Size of sub-matrices to split Hi-C contact maps into.
    :param device:  str, Device to run TADGATE on.
    :param layer_node1: int, number of nodes in first layer of autoencoder.
    :param layer_node2: int, number of nodes in second layer of autoencoder.
    :param lr:  float, Learning rate.
    :param weight_decay: float, Weight decay.
    :param num_epoch: int, Number of epochs to train TADGATE.
    :param embed_attention: bool, whether to use attention mechanism in the embedding layer (second layer).
    :param weight_use: str, Whether to use weight matrix or use different kind of weight. Choose from 'No', 'Fix', 'Auto'.
    :param weight_range: int, the range near the diagonal to apply the weight matrix.
    :param weight_rate:  float, the weight rate for the fixed weight matrix.
    :param batch_use: bool, whether to use batch training.
    :param num_batch_x: int, number of batch for batch training.
    :param auto_expand:  bool, whether to expand the auto-learned weight rate.
    :param impute_func: bool, whether to impute region near the diagnal of Hi-C contact map.
    :param impute_range: int, the range near the diagonal to apply the impute function.
    :param CNN_impute: bool or int, whether to use CNN to impute Hi-C contact map, if not False, use CNN_impute as padding size.
    :param RWR_impute: bool or float, whether to use Restart random walk to impute Hi-C contact map. if not False, use RWR_impute as restart rate.
    :param target_chr_l: list, List of chromosomes to run TADGATE on. If empty, run TADGATE on all chromosomes in hic_all.
    :param scale_f: int, scale factor for node embedding scaling with dimension.
    :param diag_cut: int, number of diagonals to cut inner. If not zero, diag_cut to wd will be zero. If zero, will use wd to cut out
    :param wd:  int, number of diagonals to cut out, wd to window will be zero. window is determined by split_size and resolution.
    :param seed: int, random seed.
    :param verbose: bool, whether to print the training process.
    :return: TADGATE_res_all: dict, TADGATE results for all chromosomes. For each chromosome,
                ‘range’ for range of each split window, ‘mat_split’ for Hi-C contact map for each window,
                ‘row_bad’ for bad rows for each window, ‘spatial_net’ for spatial network for each window,
                ‘mat_mask’ for mask matrix or weight matrix,
                ‘result’ for TADGATE results for each window.
                Detials of ‘result’ are the same as res_all_record of TADGATE_specific_chr.
    """
    TADGATE_res_all = {}
    #gpu_mem_before = torch.cuda.memory_allocated(device)
    st_time = time.time()
    for Chr in list(hic_all.keys()):
        if len(target_chr_l) != 0:
            if Chr not in target_chr_l:
                continue
        print('For ' + Chr)
        chr_length = chr_size[Chr]
        bin_name_use = TL.chr_cut(chr_length, Chr, resolution)
        length = len(bin_name_use)
        mat_hic = copy.deepcopy(hic_all[Chr])

        if CNN_impute != False and RWR_impute == False:
            mat_hic = TL.neighbor_ave_gpu(mat_hic, CNN_impute)
        elif CNN_impute == False and RWR_impute != False:
            mat_hic = TL.random_walk_gpu(mat_hic, RWR_impute)
        elif CNN_impute != False and RWR_impute != False:
            mat_hic = TL.impute_gpu(mat_hic, CNN_impute, RWR_impute)

        mat_hic2 = mat_hic + mat_hic.T - np.diag(np.diag(mat_hic))
        mat_hic = mat_hic2
        while np.max(mat_hic) <= 1000:
            mat_hic = mat_hic * 10
        while np.max(mat_hic) > 10000:
            mat_hic = mat_hic * 0.1

        if split_size == 'all':
            split_size_use = len(mat_hic) * resolution
        else:
            split_size_use = split_size

        range_l, mat_split_all, row_bad_all, spatial_net_all, mat_mask0, res_all_record = TADGATE_specific_chr(
            mat_hic, length, bin_name_use, resolution, split_size_use, graph_radius, device,
            layer_node1, layer_node2, lr, weight_decay, num_epoch, embed_attention,
            weight_use, weight_range, weight_rate, batch_use, num_batch_x,
            auto_expand, impute_func, impute_range, scale_f, diag_cut, wd, save_model, seed, verbose, split_stop)

        TADGATE_temp = {}
        TADGATE_temp['range'] = range_l
        TADGATE_temp['mat_split'] = mat_split_all
        TADGATE_temp['row_bad'] = row_bad_all
        TADGATE_temp['spatial_net'] = spatial_net_all
        TADGATE_temp['mat_mask'] = mat_mask0
        TADGATE_temp['result'] = res_all_record
        TADGATE_res_all[Chr] = TADGATE_temp

        import gc
        gc.collect()
        torch.cuda.empty_cache()
    end_time = time.time()
    #gpu_mem_after = torch.cuda.memory_allocated(device)
    print('Total time ' + str(end_time - st_time) + 's')
    #print('Total GPU memory ' + str((gpu_mem_after - gpu_mem_before) / (1024*1024)) + ' Mb')
    return TADGATE_res_all













