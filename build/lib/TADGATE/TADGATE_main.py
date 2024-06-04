
import time
import numpy as np
from . import TADGATE_utils as TL


def TADGATE_specific_chr(mat_hic, length, bin_name_use, resolution, split_size, graph_radius, device,
                         layer_node1, layer_node2, lr, weight_decay, num_epoch, embed_attention,
                         weight_use = 'No', weight_range = 0, weight_rate = 0, auto_expand = False,
                         scale_f = 1, diag_cut = 0, wd = 'all', seed = 666, verbose=True):
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
    :param auto_expand:  bool, whether to expand the auto-learned weight rate.
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
        mat_mask = TL.mat_mask_batch_to_device(mat_mask0, device, batch = 1)
    elif weight_use == 'Fix':
        weight = weight_rate
        mat_weight = np.ones([window, window])
        diag_indices = np.diag_indices(window)
        for k in range(0, weight_range):
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
        for k in range(0, weight_range):
            diag_indices_k = (diag_indices[0][:window - k], diag_indices[1][k:])
            mat_weight[diag_indices_k] += (weights[k] + 1)
        mat_upper_indices = np.triu_indices(window, k=0)
        mat_weight_new = np.zeros([window, window])
        mat_weight_new[mat_upper_indices] = mat_weight[mat_upper_indices]
        mat_weight_new += mat_weight_new.T - np.diag(np.diag(mat_weight_new))
        mat_mask0 = mat_weight_new
    for i in range(len(range_l)):
        res_all_record[i] = {}
        print('For No.' + str(i) + ' sub-map')
        ind = range_l[i]
        adata = mat_split_all[i]
        if weight_use == 'No':
            model, loss_record = TL.train_TADGATE(adata, mat_mask, scale_f, layer_node1, layer_node2, lr, weight_decay,
                                                  num_epoch, device, embed_attention = embed_attention, seed = seed, verbose=verbose)
        else:
            model, loss_record = TL.train_TADGATE_weight(adata, mat_weight_new, scale_f, layer_node1, layer_node2, lr, weight_decay,
                                                         num_epoch, device, embed_attention = embed_attention, seed = seed, verbose=verbose)

        mat_imputed, mat_imputed_sym, mat_rep, mat_att1 = TL.TADGATE_use(adata, model, device, scale_f,
                                                                         embed_attention, return_att = True)
        res_all_record[i]['loss'] = loss_record
        res_all_record[i]['model'] = model
        res_all_record[i]['map_imputed'] = mat_imputed
        res_all_record[i]['mat_imputed_sym'] = mat_imputed_sym
        res_all_record[i]['bin_rep'] = mat_rep
        res_all_record[i]['attention_map'] = mat_att1
    return range_l, mat_split_all, row_bad_all, spatial_net_all, mat_mask0, res_all_record



def TADGATE_for_embedding(hic_all, chr_size, resolution, graph_radius, split_size,  device, layer_node1,
                          layer_node2, lr, weight_decay, num_epoch, embed_attention=False, weight_use = 'No', weight_range = 0,
                          weight_rate = 0, auto_expand = False, target_chr_l = [], scale_f = 1, diag_cut = 0, wd = 'all',
                          seed = 666, verbose=True):
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
    :param auto_expand:  bool, whether to expand the auto-learned weight rate.
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
    st_time = time.time()
    for Chr in list(hic_all.keys()):
        if len(target_chr_l) != 0:
            if Chr not in target_chr_l:
                continue
        print('For ' + Chr)
        chr_length = chr_size[Chr]
        bin_name_use = TL.chr_cut(chr_length, Chr, resolution)
        length = len(bin_name_use)
        mat_hic = hic_all[Chr]
        if split_size == 'all':
            split_size_use = len(mat_hic) * resolution
        else:
            split_size_use = split_size

        range_l, mat_split_all, row_bad_all, spatial_net_all, mat_mask0, res_all_record = TADGATE_specific_chr(
            mat_hic, length, bin_name_use, resolution, split_size_use, graph_radius, device,
            layer_node1, layer_node2, lr, weight_decay, num_epoch, embed_attention,
            weight_use, weight_range, weight_rate, auto_expand, scale_f, diag_cut, wd, seed, verbose)

        TADGATE_temp = {}
        TADGATE_temp['range'] = range_l
        TADGATE_temp['mat_split'] = mat_split_all
        TADGATE_temp['row_bad'] = row_bad_all
        TADGATE_temp['spatial_net'] = spatial_net_all
        TADGATE_temp['mat_mask'] = mat_mask0
        TADGATE_temp['result'] = res_all_record
        TADGATE_res_all[Chr] = TADGATE_temp
    end_time = time.time()
    print('Total time ' + str(end_time - st_time) + 's')
    return TADGATE_res_all




