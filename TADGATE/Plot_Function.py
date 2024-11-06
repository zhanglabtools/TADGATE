
import copy
import random
import umap
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

from . import TADGATE_utils as TL

def highlight_cell(x,y, ax=None, **kwargs):
    """
    Frame visualization for plt.imshow heatmap
    :param x: position x
    :param y: position x
    :param ax: None
    :param kwargs: other
    :return: Frame visualization
    """
    rect = plt.Rectangle((x-0.5, y-0.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def get_dense_network(Net_sparse, length):
    """
    Build dense spatial matrix from sparse spatial matrix according to bin symbol along chromosome
    :param Net_sparse: pandas dataframe, sparse spatial matrix
    :param length: int, number of rows or columns for matrix
    :return: mat_dense: numpy array, dense spatial network
    """
    row = list(Net_sparse['bin1'])
    col = list(Net_sparse['bin2'])
    val = list(Net_sparse['edge'])
    mat_hic_sparse = scipy.sparse.csr_matrix((val, (row, col)), shape = (length, length))
    mat_dense_up = mat_hic_sparse.toarray()
    if (mat_dense_up == mat_dense_up.T).all():
        return mat_dense_up
    mat_dense_low = mat_dense_up.T
    mat_dense_diag = np.diag(np.diag(mat_dense_up))
    mat_dense = mat_dense_up + mat_dense_low - mat_dense_diag
    return mat_dense


def draw_graph_used_in_TADGATE_split(st, ed, mat_raw, Net_sparse, bin_name_use, Chr, resolution, bin_size,
                                     save_name=''):
    """
    Draw spatial network and Hi-C map comparison
    :param st: int, start bin index
    :param ed: int, end bin index
    :param mat_raw: np.array, Hi-C contact map
    :param Net_sparse: pd.DataFrame, sparse spatial network
    :param bin_name_use: list, bin name for chrosome used
    :param bin_size: int, bin number for draw interval
    :param save_name: str, save name of picture
    """
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'
    Net_dense = get_dense_network(Net_sparse, len(bin_name_use))
    plt.figure(figsize=(12, 4))
    ### raw mat compare
    plt.subplot(121)
    contact_map = mat_raw[st: ed, st: ed]
    plt.imshow(contact_map, cmap='coolwarm', vmax=np.percentile(contact_map, 95), vmin=np.percentile(contact_map, 5))
    plt.colorbar(fraction=0.05, pad=0.05)
    plt.xticks(cord_list, x_ticks_l, fontsize=10)
    plt.yticks(cord_list, y_ticks_l, fontsize=10)
    plt.title(region_name)

    plt.subplot(122)
    contact_map = Net_dense[st: ed, st: ed]
    plt.imshow(contact_map, cmap='coolwarm', vmax=1, vmin=0)
    plt.colorbar(fraction=0.05, pad=0.05)
    plt.xticks(cord_list, x_ticks_l, fontsize=10)
    plt.yticks(cord_list, y_ticks_l, fontsize=10)
    for i in range(0, ed - st):
        for j in range(0, ed - st):
            highlight_cell(i, j)
    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()

def draw_loss_record(loss_record, x_lim = '', y_lim = '', label = '', fgsize = (4, 3)):
    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    if label != '':
        plt.plot(loss_record, label = label)
    else:
        plt.plot(loss_record, label=label)
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    if x_lim != '':
        plt.xlim([x_lim[0], x_lim[-1]])
    if y_lim != '':
        plt.ylim([y_lim[0], y_lim[-1]])
    if label != '':
        plt.legend()

def draw_multi_mat_compare(st, ed, mat_list, mat_para_list, bin_name_use, Chr, resolution, title_l = [],
                           fgsize = (12,6), bin_size = 50, save_name = '', ori = 'h'):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'
    Num = len(mat_list)
    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    for i in range(len(mat_list)):
        mat_draw = mat_list[i]
        mat_para = mat_para_list[i]
        if len(mat_para) == 0:
            color = 'Reds'
            v_min = 10
            v_max = 90
            diag = False
            net = False
            value_type = 'no-real'
        else:
            color = mat_para['color']
            v_min = mat_para['range'][0]
            v_max = mat_para['range'][-1]
            diag = mat_para['diag']
            net = mat_para['net']
            value_type = mat_para['value_type']
        if ori == 'h':
            plt.subplot2grid((1, Num), (0, i))
            if mat_draw.shape[0] != mat_draw.shape[-1]:
                contact_map_1 = mat_draw[st: ed, 0: ed - st]
            else:
                contact_map_1 = mat_draw[st: ed, st: ed]
            if value_type == 'real':
                plt.imshow(contact_map_1, cmap=color, vmax=v_max,
                           vmin=v_min)
            else:
                plt.imshow(contact_map_1, cmap = color, vmax=np.percentile(contact_map_1, v_max), vmin = np.percentile(contact_map_1, v_min))
            plt.colorbar(fraction=0.05, pad=0.05)
            plt.xticks(cord_list, x_ticks_l, fontsize = 10)
            if i == 0:
                plt.title(region_name)
                plt.yticks(cord_list, y_ticks_l, fontsize = 10)
            else:
                if len(title_l) != 0:
                    plt.title(title_l[i])
                plt.yticks(cord_list, '')
        elif ori == 'v':
            plt.subplot2grid((Num, 1), (i, 0))
            if mat_draw.shape[0] != mat_draw.shape[-1]:
                contact_map_1 = mat_draw[st: ed, 0: ed-st]
            else:
                contact_map_1 = mat_draw[st : ed, st: ed]
            if value_type == 'real':
                plt.imshow(contact_map_1, cmap=color, vmax=v_max,
                           vmin=v_min)
            else:
                plt.imshow(contact_map_1, cmap=color, vmax=np.percentile(contact_map_1, v_max),
                           vmin=np.percentile(contact_map_1, v_min))
            plt.colorbar(fraction=0.05, pad=0.05)
            plt.yticks(cord_list, y_ticks_l, fontsize = 10)
            if i == 0:
                plt.title(region_name)
                plt.xticks(cord_list, '')
            elif i == Num-1:
                plt.xticks(cord_list, x_ticks_l, fontsize = 10)
            else:
                plt.xticks(cord_list, '')
        if diag == True:
            plt.axline((0, 0), slope = 1, color='k', linestyle='--')
        if net == True and (ed - st) <= 30:
            for p in range(0, ed - st):
                for q in range(0, ed - st):
                    highlight_cell(p, q)
        ax=plt.gca()
        ax.spines['bottom'].set_linewidth(1.6)
        ax.spines['left'].set_linewidth(1.6)
        ax.spines['right'].set_linewidth(1.6)
        ax.spines['top'].set_linewidth(1.6)
        ax.tick_params(axis = 'y', length=5, width = 1.6)
        ax.tick_params(axis = 'x', length=5, width = 1.6)
    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()


def draw_multi_mat_compare_multi_row(st, ed, mat_list, mat_para_list, bin_name_use, Chr, resolution, col_num,
                                     title_name_l='', fgsize=(12, 6), bin_size=50, save_name='', ori='h', bar_label = True):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'
    Num = len(mat_list)

    pos_l = []
    row_num = 1
    if len(mat_list) > col_num:
        row_num = int(np.ceil(len(mat_list) / col_num))
        for i in range(row_num):
            for j in range(col_num):
                pos_l.append((i, j))
    else:
        col_num = len(mat_list)
        for i in range(row_num):
            for j in range(col_num):
                pos_l.append((i, j))

    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    for i in range(len(mat_list)):
        mat_draw = mat_list[i]
        mat_para = mat_para_list[i]
        if len(mat_para) == 0:
            color = 'Reds'
            v_min = 10
            v_max = 90
            diag = False
            net = False
            value_type = 'no-real'
        else:
            color = mat_para['color']
            v_min = mat_para['range'][0]
            v_max = mat_para['range'][-1]
            diag = mat_para['diag']
            net = mat_para['net']
            value_type = mat_para['value_type']

        plt.subplot2grid((row_num, col_num), pos_l[i])
        if mat_draw.shape[0] != mat_draw.shape[-1]:
            contact_map_1 = mat_draw[st: ed, 0: ed - st]
        else:
            contact_map_1 = mat_draw[st: ed, st: ed]
        if value_type == 'real':
            plt.imshow(contact_map_1, cmap=color, vmax=v_max,
                       vmin=np.max([v_min, 0]))
        else:
            plt.imshow(contact_map_1, cmap=color, vmax=np.percentile(contact_map_1, v_max),
                       vmin=np.max([np.percentile(contact_map_1, v_min), 0]))
        cbar = plt.colorbar(fraction=0.05, pad=0.05)
        if bar_label == False:
            cbar.set_ticklabels([])
        if i % col_num == 0 :
            plt.yticks(cord_list, y_ticks_l, fontsize=10)
        else:
            plt.yticks(cord_list, ['' for x in range(len(cord_list))], fontsize=10)
        if i + col_num >= len(mat_list):
            plt.xticks(cord_list, x_ticks_l, fontsize=10)
        else:
            plt.xticks(cord_list, ['' for x in range(len(cord_list))], fontsize=10)

        if i == 0:
            plt.title(region_name)
        else:
            if len(title_name_l) != 0:
                plt.title(title_name_l[i])
        if diag == True:
            plt.axline((0, 0), slope=1, color='k', linestyle='--')
        if net == True and (ed - st) <= 30:
            for p in range(0, ed - st):
                for q in range(0, ed - st):
                    highlight_cell(p, q)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.6)
        ax.spines['left'].set_linewidth(1.6)
        ax.spines['right'].set_linewidth(1.6)
        ax.spines['top'].set_linewidth(1.6)
        ax.tick_params(axis='y', length=5, width=1.6)
        ax.tick_params(axis='x', length=5, width=1.6)
    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()


def matrix_part_max_norm(mat_region):
    vec_diag = np.diag(mat_region)
    mat_diag = np.diag(vec_diag)
    mat_region -= mat_diag
    mat_region = mat_region / np.max(mat_region)
    return mat_region

def draw_square_region(st, ed, color, size_v, size_h):
    ## 画竖线
    plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    plt.vlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    ## 画横线
    plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
    plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)

def draw_tad_region(st, ed, color, size_v, size_h):
    ## 画竖线
    plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    plt.vlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    ## 画横线
    plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
    plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)

def draw_tad_region_upper_half(st, ed, range_t, color, size_v, size_h):
    if st < 0:
        #plt.vlines(ed, 0, ed, colors=color, linestyles='solid', linewidths=size_v)
        plt.plot([ed,ed], [0, ed], color=color, linestyle='solid', linewidth=size_v)
    elif ed > range_t:
        #plt.hlines(st, st, range_t, colors=color, linestyles='solid', linewidths=size_v)
        plt.plot([st, range_t], [st, st], color=color, linestyle='solid', linewidth=size_v)
    else:  ## 画竖线
        # plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
        plt.plot([ed, ed], [st, ed], color=color, linestyle='solid', linewidth=size_v)
        ## 画横线
        #plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
        plt.plot([st, ed], [st, st], color=color, linestyle='solid', linewidth=size_v)

def draw_tad_region_lower_half(st, ed, range_t, color, size_v, size_h):
    if st < 0:
        #plt.hlines(ed, 0, ed, colors=color, linestyles='solid', linewidths=size_h)
        plt.plot([0, ed], [ed, ed], color=color, linestyle='solid', linewidth=size_v)
    elif ed > range_t:
        #plt.vlines(st, st, range_t, colors=color, linestyles='solid', linewidths=size_v)
        plt.plot([st, st], [st, range_t], color=color, linestyle='solid', linewidth=size_v)
    else:
        ## 画竖线
        #plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
        plt.plot([st, st], [st, ed], color=color, linestyle='solid', linewidth=size_v)
        ## 画横线
        #plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)
        plt.plot([st, ed], [ed, ed], color=color, linestyle='solid', linewidth=size_v)

def draw_pair_wise_map_compare_TADs(Chr, st, ed, bin_name_use, mat_dense1, mat_dense2, resolution,
                                      m1='', m2='', TAD_list_1=[], TAD_list_2=[],
                                      tad_color_1='black', tad_color_2='black',
                                      fgsize = (9, 9), value_range = (90, 90), save_name='', bin_size=20):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'

    dense_matrix_part1 = copy.deepcopy(mat_dense1[st:ed, st:ed])
    dense_matrix_part2 = copy.deepcopy(mat_dense2[st:ed, st:ed])
    dense_matrix_norm1 = dense_matrix_part1
    dense_matrix_norm2 = dense_matrix_part2
    #dense_matrix_norm1 = matrix_part_max_norm(dense_matrix_part1)
    #dense_matrix_norm2 = matrix_part_max_norm(dense_matrix_part2)
    dense_matrix_combine = np.triu(dense_matrix_norm1) + np.tril(-dense_matrix_norm2)

    #vmax = np.percentile(dense_matrix_combine, 90)
    #vmin = -vmax
    vmax = np.percentile(dense_matrix_norm1, value_range[0])
    vmin = -np.percentile(dense_matrix_norm2, value_range[-1])
    #vmin = -np.percentile(dense_matrix_norm1, 90)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    # plt.imshow(dense_matrix_combine, cmap = 'seismic', vmin = vmin, vmax =vmax, norm=norm)
    range_t = ed - st - 1
    plt.imshow(dense_matrix_combine, cmap='coolwarm', norm=norm)
    if len(TAD_list_1) != 0:
        for i in range(len(TAD_list_1)):
            TAD = TAD_list_1[i]
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st + 1
            # print(st_tad, ed_tad)
            # draw_tad_region(st_tad, ed_tad, TAD_color_1, size_v=5, size_h=5)
            draw_tad_region_upper_half(st_tad, ed_tad, range_t, tad_color_1, size_v=5, size_h=5)
    if len(TAD_list_2) != 0:
        for i in range(len(TAD_list_2)):
            TAD = TAD_list_2[i]
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st + 1
            # print(st_tad, ed_tad)
            # draw_tad_region(st_tad, ed_tad, TAD_color, size_v=5, size_h=5)
            draw_tad_region_lower_half(st_tad, ed_tad, range_t, tad_color_2, size_v=5, size_h=5)
    plt.colorbar(fraction=0.05, pad=0.05)
    plt.xticks(cord_list, x_ticks_l, fontsize=16)
    plt.yticks(cord_list, y_ticks_l, fontsize=16)
    if m1 == '' and m2 == '':
        plt.title(region_name, fontsize=16, pad=15.0)
    elif m1 == m2:
        plt.title(m1 + ' TADs in ' + region_name, fontsize=16, pad=15.0)
    elif m1 != m2:
        plt.title(m1 + ' vs ' + m2 + ' in ' + region_name, fontsize=16, pad=15.0)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis='y', length=5, width=1.6)
    ax.tick_params(axis='x', length=5, width=1.6)
    if save_name != '':
        plt.savefig(save_name, format='svg')
    plt.show()
    # fig = plt.gcf() #获取当前figure
    # plt.close(fig)

def draw_pair_wise_map_compare_TADs_same_matrix(Chr, st, ed, bin_name_use, mat_dense, resolution,
                                      m1='', m2='', TAD_list_1=[], TAD_list_2=[], map_color = 'Reds',
                                      tad_color_1='black', tad_color_2='black',
                                      fgsize = (9, 9), value_range = (10, 90), save_name='', bin_size=20):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'

    dense_matrix_combine = copy.deepcopy(mat_dense[st:ed, st:ed])

    vmax = np.percentile(dense_matrix_combine, value_range[-1])
    vmin = np.percentile(dense_matrix_combine, value_range[0])

    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    range_t = ed - st - 1
    plt.imshow(dense_matrix_combine, cmap=map_color, vmin = vmin, vmax = vmax)
    if len(TAD_list_1) != 0:
        for i in range(len(TAD_list_1)):
            TAD = TAD_list_1[i]
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st + 1
            # print(st_tad, ed_tad)
            # draw_tad_region(st_tad, ed_tad, TAD_color_1, size_v=5, size_h=5)
            draw_tad_region_upper_half(st_tad, ed_tad, range_t, tad_color_1, size_v=5, size_h=5)
    if len(TAD_list_2) != 0:
        for i in range(len(TAD_list_2)):
            TAD = TAD_list_2[i]
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st + 1
            # print(st_tad, ed_tad)
            # draw_tad_region(st_tad, ed_tad, TAD_color, size_v=5, size_h=5)
            draw_tad_region_lower_half(st_tad, ed_tad, range_t, tad_color_2, size_v=5, size_h=5)
    plt.colorbar(fraction=0.05, pad=0.05)
    plt.xticks(cord_list, x_ticks_l, fontsize=16)
    plt.yticks(cord_list, y_ticks_l, fontsize=16)
    if m1 == '' and m2 == '':
        plt.title(region_name, fontsize=16, pad=15.0)
    elif m1 == m2:
        plt.title(m1 + ' TADs in ' + region_name, fontsize=16, pad=15.0)
    elif m1 != m2:
        plt.title(m1 + ' vs ' + m2 + ' in ' + region_name, fontsize=16, pad=15.0)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis='y', length=5, width=1.6)
    ax.tick_params(axis='x', length=5, width=1.6)
    if save_name != '':
        plt.savefig(save_name, format='svg')
    plt.show()
    # fig = plt.gcf() #获取当前figure
    # plt.close(fig)

def draw_multi_mat_compare_multi_row_with_TADs(st, ed, mat_list, mat_para_list, TAD_list, bin_name_use, Chr, resolution, col_num,
                                               title_name_l='', fgsize=(12, 6), bin_size=50, save_name='', ori='h', bar_label = True,
                                               ticks_draw = True, frame_w = 5):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'
    Num = len(mat_list)

    pos_l = []
    row_num = 1
    if len(mat_list) > col_num:
        row_num = int(np.ceil(len(mat_list) / col_num))
        for i in range(row_num):
            for j in range(col_num):
                pos_l.append((i, j))
    else:
        col_num = len(mat_list)
        for i in range(row_num):
            for j in range(col_num):
                pos_l.append((i, j))

    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    for i in range(len(mat_list)):
        mat_draw = mat_list[i]
        mat_para = mat_para_list[i]
        if len(TAD_list) == 0:
            TAD_l = []
        else:
            TAD_l = TAD_list[i]
        if len(mat_para) == 0:
            color = 'Reds'
            v_min = 10
            v_max = 90
            diag = False
            net = False
            value_type = 'no-real'
            tad_color = 'black'
            upper = True
            lower = True
            norm_mat = False
        else:
            color = mat_para['color']
            v_min = mat_para['range'][0]
            v_max = mat_para['range'][-1]
            diag = mat_para['diag']
            net = mat_para['net']
            value_type = mat_para['value_type']
            tad_color = mat_para['tad_color']
            upper = mat_para['tad_upper']
            lower = mat_para['tad_lower']
            norm_mat = mat_para['norm_mat']
        range_t = ed - st - 1
        plt.subplot2grid((row_num, col_num), pos_l[i])
        if mat_draw.shape[0] != mat_draw.shape[-1]:
            contact_map_1 = mat_draw[st: ed, 0: ed - st]
        else:
            contact_map_1 = mat_draw[st: ed, st: ed]
        if norm_mat == True:
            contact_map_1 = copy.deepcopy(contact_map_1 / np.max(contact_map_1))
        if value_type == 'real':
            plt.imshow(contact_map_1, cmap=color, vmax=v_max,
                       vmin=v_min)
        else:
            plt.imshow(contact_map_1, cmap=color, vmax=np.percentile(contact_map_1, v_max),
                       vmin=np.max([np.percentile(contact_map_1, v_min), 0]))
        if len(TAD_l) != 0:
            for p in range(len(TAD_l)):
                TAD = TAD_l[p]
                st_tad = TAD[0] - st
                ed_tad = TAD[1] - st + 1
                # draw_tad_region(st_tad, ed_tad, TAD_color_1, size_v=5, size_h=5)
                if upper == True:
                    draw_tad_region_upper_half(st_tad, ed_tad, range_t, tad_color, size_v=frame_w, size_h=frame_w)
                if lower == True:
                    draw_tad_region_lower_half(st_tad, ed_tad, range_t, tad_color, size_v=frame_w, size_h=frame_w)
        cbar = plt.colorbar(fraction=0.05, pad=0.05)
        if bar_label == False:
            cbar.set_ticklabels([])
        if i % col_num == 0:
            plt.yticks(cord_list, y_ticks_l, fontsize=10)
        else:
            plt.yticks(cord_list, ['' for x in range(len(cord_list))], fontsize=10)
        if i + col_num >= len(mat_list):
            plt.xticks(cord_list, x_ticks_l, fontsize=10)
        else:
            plt.xticks(cord_list, ['' for x in range(len(cord_list))], fontsize=10)

        if i == 0:
            plt.title(region_name)
        else:
            if len(title_name_l) != 0:
                plt.title(title_name_l[i])
        if diag == True:
            plt.axline((0, 0), slope=1, color='k', linestyle='--')
        if net == True and (ed - st) <= 30:
            for p in range(0, ed - st):
                for q in range(0, ed - st):
                    highlight_cell(p, q)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.6)
        ax.spines['left'].set_linewidth(1.6)
        ax.spines['right'].set_linewidth(1.6)
        ax.spines['top'].set_linewidth(1.6)
        if ticks_draw == True:
            ax.tick_params(axis='y', length=5, width=1.6)
            ax.tick_params(axis='x', length=5, width=1.6)
        else:
            ax.tick_params(axis='y', length=0, width=0)
            ax.tick_params(axis='x', length=0, width=0)
    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()


def compare_label_new_old(st, ed, bin_label_l, bin_label_l_new, y_lim = ''):
    plt.figure(figsize = (20, 5))
    plt.scatter(list(range(len(bin_label_l))), bin_label_l)
    plt.xlim([st, ed])
    if y_lim != '':
        plt.ylim(y_lim)
    plt.ylabel('Bin label old')
    plt.xlabel('Bin order')

    plt.figure(figsize = (20, 5))
    plt.scatter(list(range(len(bin_label_l_new))), bin_label_l_new)
    plt.xlim([st, ed])
    if y_lim != '':
        plt.ylim(y_lim)
    plt.ylabel('Bin label new')
    plt.xlabel('Bin order')


def RandColor():
    color_num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color_code = ""
    for i in range(6):
        color_code += color_num[random.randint(0, 14)]
    return "#" + color_code


def compare_tads_and_bin_rep(st, ed, mat_list, mat_para_list, TAD_list, mat_rep, bin_lb, rd_method, bin_name_use, Chr,
                             resolution, col_num, rd_dim = True, title_name_l='', fgsize = (10,10), bin_size=20,
                             save_name='', ori='h', add_tor=False, arrow_draw = True, legend = 'Yes'):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'
    Num = len(mat_list)

    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    i = 0
    mat_draw = mat_list[i]
    mat_para = mat_para_list[i]
    TAD_l = TAD_list[i]
    if len(mat_para) == 0:
        color = 'Reds'
        v_min = 10
        v_max = 90
        diag = False
        net = False
        value_type = 'no-real'
        tad_color = 'black'
        upper = True
        lower = True
    else:
        color = mat_para['color']
        v_min = mat_para['range'][0]
        v_max = mat_para['range'][-1]
        diag = mat_para['diag']
        net = mat_para['net']
        value_type = mat_para['value_type']
        tad_color = mat_para['tad_color']
        upper = mat_para['tad_upper']
        lower = mat_para['tad_lower']
    range_t = ed - st - 1
    if ori == 'h':
        pos_1 = [(1, 2), (0, 0)]
        pos_2 = [(1, 2), (0, 1)]
    elif ori == 'v':
        pos_1 = [(2, 1), (0, 0)]
        pos_2 = [(2, 1), (1, 0)]

    plt.subplot2grid(pos_1[0], pos_1[1])
    if mat_draw.shape[0] != mat_draw.shape[-1]:
        contact_map_1 = mat_draw[st: ed, 0: ed - st]
    else:
        contact_map_1 = mat_draw[st: ed, st: ed]
    if value_type == 'real':
        plt.imshow(contact_map_1, cmap=color, vmax=v_max,
                   vmin=v_min)
    else:
        plt.imshow(contact_map_1, cmap=color, vmax=np.percentile(contact_map_1, v_max),
                   vmin=np.percentile(contact_map_1, v_min))

    #color_list = []
    #for i in range(50):
        #color_list.append(RandColor())
    color_list = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                  '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#8dd3c7', '#ffffb3',
                  '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9',
                  '#bc80bd', '#ccebc5', '#ffed6f']
    if len(TAD_l) != 0:
        for p in range(len(TAD_l)):
            TAD = TAD_l[p]
            if tad_color == 'self_define':
                tad_color_use = color_list[p]
            else:
                tad_color_use = tad_color
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st + 1
            # draw_tad_region(st_tad, ed_tad, TAD_color_1, size_v=5, size_h=5)
            if upper == True:
                draw_tad_region_upper_half(st_tad, ed_tad, range_t, tad_color_use, size_v=5, size_h=5)
            if lower == True:
                draw_tad_region_lower_half(st_tad, ed_tad, range_t, tad_color_use, size_v=5, size_h=5)

    plt.colorbar(fraction=0.05, pad=0.05)
    plt.xticks(cord_list, x_ticks_l, fontsize=10)
    plt.yticks(cord_list, y_ticks_l, fontsize=10)
    if i == 0:
        plt.title(region_name)
    else:
        if len(title_name_l) != 0:
            plt.title(title_name_l[i])
    if diag == True:
        plt.axline((0, 0), slope=1, color='k', linestyle='--')
    if net == True and (ed - st) <= 30:
        for p in range(0, ed - st):
            for q in range(0, ed - st):
                highlight_cell(p, q)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis='y', length=5, width=1.6)
    ax.tick_params(axis='x', length=5, width=1.6)
    print('work')
    if rd_dim == True:
        if rd_method == 'UMAP':
            trans_t = umap.UMAP(n_components=2, random_state=0)
        elif rd_method == 'TSNE':
            trans_t = TSNE(n_components=2, random_state=0)
        elif rd_method == 'PCA':
            trans_t = PCA(n_components=2, random_state=0)
        elif rd_method == 'MDS':
            trans_t = MDS(n_components=2, random_state=0)
        mat_part_rd = trans_t.fit_transform(mat_rep[st:ed, :])
    else:
        mat_part_rd = mat_rep[st:ed, :]

    color_use = {}
    lb_use = sorted(list(np.unique(bin_lb[st:ed])))
    lb_count = 0
    for lb in lb_use:
        if lb == -1:
            color_use[lb] = 'grey'
        else:
            color_use[lb] = color_list[lb_count]
            lb_count += 1
    color_l = []
    for lb in bin_lb[st:ed]:
        color_l.append(color_use[lb])
    df_bin_scatter = pd.DataFrame(mat_part_rd)
    df_bin_scatter.columns = ['Dim_1', 'Dim_2']
    df_bin_scatter['label'] = bin_lb[st:ed]
    df_bin_scatter['color'] = color_l

    if add_tor == True:
        tor_d1 = [random.uniform(-0.2, 0.2) for i in range(len(df_bin_scatter))]
        tor_d2 = [random.uniform(-0.2, 0.2) for i in range(len(df_bin_scatter))]
        df_bin_scatter['Dim_1'] = df_bin_scatter['Dim_1'] + np.array(tor_d1)
        df_bin_scatter['Dim_2'] = df_bin_scatter['Dim_2'] + np.array(tor_d2)

    plt.subplot2grid(pos_2[0], pos_2[1])
    sns.scatterplot(x='Dim_1', y='Dim_2', data=df_bin_scatter, s=100, palette=color_use, hue=df_bin_scatter['label'],
                    alpha=0.8, edgecolor = 'black', linewidth = 1.5)
    dist_hold = 0
    x_spand = np.max(df_bin_scatter['Dim_1']) - np.min(df_bin_scatter['Dim_1'])
    for i in range(len(df_bin_scatter) - 1):
        ax1 = [df_bin_scatter['Dim_1'][i], df_bin_scatter['Dim_2'][i]]
        ax2 = [df_bin_scatter['Dim_1'][i + 1], df_bin_scatter['Dim_2'][i + 1]]
        plt.plot([ax1[0], ax2[0]], [ax1[1], ax2[1]], color='black', linewidth=1.5, linestyle='dashed')
        dist = pdist(np.array([ax1, ax2]), 'euclidean')
        if arrow_draw == True:
            if dist > dist_hold and dist_hold != 0:
                slope = (ax2[1] - ax1[1]) / (ax2[0] - ax1[0])
                inter1 = ax2[0] - ax1[0]
                plt.arrow(ax1[0] + inter1 / 2, ax1[1] + inter1 / 2 * slope, inter1 / 10, inter1 / 10 * slope,
                          head_width = x_spand * 0.01, linestyle='solid', color='black')
                dist_hold = 0
                continue
            if i % 5 == 0:
                dist_hold = 0
            else:
                dist_hold += dist

    plt.xlabel('Dim 1', fontsize=10)
    plt.ylabel('Dim 2', fontsize=10)
    #plt.gca().set_aspect('equal', 'datalim')
    plt.title(rd_method + ' for bin visualization', fontsize=10)
    plt.legend()
    ax = plt.gca()
    if legend == 'No':
        ax.legend_.remove()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis='y', length=5, width=1.6)
    ax.tick_params(axis='x', length=5, width=1.6)

    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()

def compare_tads_and_bin_rep2(random_state, st, ed, mat_list, mat_para_list, TAD_list, mat_rep, bin_lb, rd_method, bin_name_use, Chr,
                             resolution, col_num, rd_dim = True, title_name_l='', fgsize = (10,10), bin_size=20,
                             save_name='', ori='h', add_tor=False, arrow_draw = True, legend = 'Yes'):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'
    Num = len(mat_list)

    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    i = 0
    mat_draw = mat_list[i]
    mat_para = mat_para_list[i]
    TAD_l = TAD_list[i]
    if len(mat_para) == 0:
        color = 'Reds'
        v_min = 10
        v_max = 90
        diag = False
        net = False
        value_type = 'no-real'
        tad_color = 'black'
        upper = True
        lower = True
    else:
        color = mat_para['color']
        v_min = mat_para['range'][0]
        v_max = mat_para['range'][-1]
        diag = mat_para['diag']
        net = mat_para['net']
        value_type = mat_para['value_type']
        tad_color = mat_para['tad_color']
        upper = mat_para['tad_upper']
        lower = mat_para['tad_lower']
    range_t = ed - st - 1
    if ori == 'h':
        pos_1 = [(1, 2), (0, 0)]
        pos_2 = [(1, 2), (0, 1)]
    elif ori == 'v':
        pos_1 = [(2, 1), (0, 0)]
        pos_2 = [(2, 1), (1, 0)]

    plt.subplot2grid(pos_1[0], pos_1[1])
    if mat_draw.shape[0] != mat_draw.shape[-1]:
        contact_map_1 = mat_draw[st: ed, 0: ed - st]
    else:
        contact_map_1 = mat_draw[st: ed, st: ed]
    if value_type == 'real':
        plt.imshow(contact_map_1, cmap=color, vmax=v_max,
                   vmin=v_min)
    else:
        plt.imshow(contact_map_1, cmap=color, vmax=np.percentile(contact_map_1, v_max),
                   vmin=np.percentile(contact_map_1, v_min))

    #color_list = []
    #for i in range(50):
        #color_list.append(RandColor())
    color_list = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                  '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#8dd3c7', '#ffffb3',
                  '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9',
                  '#bc80bd', '#ccebc5', '#ffed6f']
    if len(TAD_l) != 0:
        for p in range(len(TAD_l)):
            TAD = TAD_l[p]
            if tad_color == 'self_define':
                tad_color_use = color_list[p]
            else:
                tad_color_use = tad_color
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st + 1
            # draw_tad_region(st_tad, ed_tad, TAD_color_1, size_v=5, size_h=5)
            if upper == True:
                draw_tad_region_upper_half(st_tad, ed_tad, range_t, tad_color_use, size_v=5, size_h=5)
            if lower == True:
                draw_tad_region_lower_half(st_tad, ed_tad, range_t, tad_color_use, size_v=5, size_h=5)

    plt.colorbar(fraction=0.05, pad=0.05)
    plt.xticks(cord_list, x_ticks_l, fontsize=10)
    plt.yticks(cord_list, y_ticks_l, fontsize=10)
    if i == 0:
        plt.title(region_name)
    else:
        if len(title_name_l) != 0:
            plt.title(title_name_l[i])
    if diag == True:
        plt.axline((0, 0), slope=1, color='k', linestyle='--')
    if net == True and (ed - st) <= 30:
        for p in range(0, ed - st):
            for q in range(0, ed - st):
                highlight_cell(p, q)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis='y', length=5, width=1.6)
    ax.tick_params(axis='x', length=5, width=1.6)

    if rd_dim == True:
        if rd_method == 'UMAP':
            #trans_t = umap.UMAP(n_components=2, random_state=0)
            trans_t = umap.UMAP(n_components=2, random_state=random_state)
            print(random_state)
        elif rd_method == 'TSNE':
            trans_t = TSNE(n_components=2, random_state=0)
        elif rd_method == 'PCA':
            trans_t = PCA(n_components=2, random_state=0)
        elif rd_method == 'MDS':
            trans_t = MDS(n_components=2, random_state=0)
        mat_part_rd = trans_t.fit_transform(mat_rep[st:ed, :])
    else:
        mat_part_rd = mat_rep[st:ed, :]

    color_use = {}
    lb_use = sorted(list(np.unique(bin_lb[st:ed])))
    lb_count = 0
    for lb in lb_use:
        if lb == -1:
            color_use[lb] = 'grey'
        else:
            color_use[lb] = color_list[lb_count]
            lb_count += 1
    color_l = []
    for lb in bin_lb[st:ed]:
        color_l.append(color_use[lb])
    df_bin_scatter = pd.DataFrame(mat_part_rd)
    df_bin_scatter.columns = ['Dim_1', 'Dim_2']
    df_bin_scatter['label'] = bin_lb[st:ed]
    df_bin_scatter['color'] = color_l

    if add_tor == True:
        tor_d1 = [random.uniform(-0.2, 0.2) for i in range(len(df_bin_scatter))]
        tor_d2 = [random.uniform(-0.2, 0.2) for i in range(len(df_bin_scatter))]
        df_bin_scatter['Dim_1'] = df_bin_scatter['Dim_1'] + np.array(tor_d1)
        df_bin_scatter['Dim_2'] = df_bin_scatter['Dim_2'] + np.array(tor_d2)

    plt.subplot2grid(pos_2[0], pos_2[1])
    sns.scatterplot(x='Dim_1', y='Dim_2', data=df_bin_scatter, s=100, palette=color_use, hue=df_bin_scatter['label'],
                    alpha=0.8, edgecolor = 'black', linewidth = 1.5)
    dist_hold = 0
    x_spand = np.max(df_bin_scatter['Dim_1']) - np.min(df_bin_scatter['Dim_1'])
    for i in range(len(df_bin_scatter) - 1):
        ax1 = [df_bin_scatter['Dim_1'][i], df_bin_scatter['Dim_2'][i]]
        ax2 = [df_bin_scatter['Dim_1'][i + 1], df_bin_scatter['Dim_2'][i + 1]]
        plt.plot([ax1[0], ax2[0]], [ax1[1], ax2[1]], color='black', linewidth=1.5, linestyle='dashed')
        dist = pdist(np.array([ax1, ax2]), 'euclidean')
        if arrow_draw == True:
            if dist > dist_hold and dist_hold != 0:
                slope = (ax2[1] - ax1[1]) / (ax2[0] - ax1[0])
                inter1 = ax2[0] - ax1[0]
                plt.arrow(ax1[0] + inter1 / 2, ax1[1] + inter1 / 2 * slope, inter1 / 10, inter1 / 10 * slope,
                          head_width = x_spand * 0.01, linestyle='solid', color='black')
                dist_hold = 0
                continue
            if i % 5 == 0:
                dist_hold = 0
            else:
                dist_hold += dist

    plt.xlabel('Dim 1', fontsize=10)
    plt.ylabel('Dim 2', fontsize=10)
    #plt.gca().set_aspect('equal', 'datalim')
    plt.title(rd_method + ' for bin visualization', fontsize=10)
    plt.legend()
    ax = plt.gca()
    if legend == 'No':
        ax.legend_.remove()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis='y', length=5, width=1.6)
    ax.tick_params(axis='x', length=5, width=1.6)

    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()
    return df_bin_scatter, color_use





from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def draw_map_multi_CI_old(mat_dense, Chr, st, ed, bin_name_use, df_pvalue_multi, window_l, ci_peak_multi, resolution,
                      p_cut, target_site=[], bin_size=10, save_name = ''):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    plt.figure(figsize=(6, 10))
    x_axis_range = range(0, ed - st)
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'

    ax1 = plt.subplot2grid((11, 7), (0, 0), rowspan=6, colspan=6)
    start = int(start_ / resolution)
    end = int(end_ / resolution)
    dense_matrix_part = mat_dense[start:end, start:end]
    img = ax1.imshow(dense_matrix_part, cmap='Reds', vmin=np.percentile(dense_matrix_part, 10),
                     vmax=np.percentile(dense_matrix_part, 90))

    ax1.set_xticks([])
    ax1.spines['bottom'].set_linewidth(0)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)
    ax1.tick_params(axis='y', length=5, width=1.6)
    ax1.tick_params(axis='x', length=5, width=1.6)
    plt.xticks(cord_list, x_ticks_l, fontsize=0, rotation=90)
    plt.yticks(cord_list, y_ticks_l, fontsize=10)
    ax1.set_title('TAD landscape of region:' + region_name, fontsize=12, pad=15.0)

    cax = plt.subplot2grid((11, 7), (0, 6), rowspan=6, colspan=1)
    cbaxes = inset_axes(cax, width="30%", height="100%", loc=3)
    plt.colorbar(img, cax=cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis='y', length=0, width=0)
    cax.tick_params(axis='x', length=0, width=0)
    cax.set_xticks([])
    cax.set_yticks([])

    for i in range(1, len(window_l) + 1):
        wd = window_l[-i]
        ax1_5 = plt.subplot2grid((11, 7), (6 + i - 1, 0), rowspan=1, colspan=6, sharex=ax1)
        ax1_5.plot(list(df_pvalue_multi[wd][st:ed]), color='black')
        ax1_5.bar(x_axis_range, list(df_pvalue_multi[wd][st:ed]), label=wd, color='#00A2E8')
        if p_cut != 0:
            ax1_5.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color='black', linestyles='--')
        ax1_5.spines['bottom'].set_linewidth(1.6)
        ax1_5.spines['left'].set_linewidth(1.6)
        ax1_5.spines['right'].set_linewidth(1.6)
        ax1_5.spines['top'].set_linewidth(1.6)
        ax1_5.tick_params(axis='y', length=5, width=1.6)
        ax1_5.tick_params(axis='x', length=5, width=1.6)
        ax1_5.set_ylabel(wd, fontsize=10)
        ax1_5.set_xticks(cord_list, x_ticks_l, fontsize=0, rotation=90)
        # ax1_5.set_yticks([0, 0.1, 0.2], ['', 0.1, 0.2], fontsize = 8,)
        # plt.ylim([0, 0.2])
        if ci_peak_multi != '':
            target_site = ci_peak_multi[wd]
            if len(target_site) != 0:
                site_use = []
                for i in range(len(target_site)):
                    if target_site[i] < st:
                        pass
                    elif target_site[i] < end:
                        site_use.append(target_site[i])
                    else:
                        break
                plt.vlines(np.array(site_use) - st, 0, 1, color='black')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis='y', length=4, width=1.8)
    ax.tick_params(axis='x', length=4, width=1.8)
    ax.set_xticks(cord_list, x_ticks_l, fontsize=10, rotation=-30)

    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()



def draw_map_multi_CI(mat_dense, Chr, st, ed, bin_name_use, df_pvalue_multi, window_l, ci_peak_multi,
                      resolution, save_name, track_cut, bin_size=10, TAD_l=[],
                      TAD_dict = {'upper' : True, 'lower': False, 'color': 'black', 'linewidth': 5, 'linestyle': 'solid'},
                      color_supply = [], color_multi=False, score_type='No', h_range=(10, 90), track_range = [],
                      fgsize = (6, 10), subfig = (11, 6)):
    st_split = int(bin_name_use[0].split(':')[-1].split('-')[0]) / resolution
    start_ = (st + st_split) * resolution
    end_ = (ed + st_split) * resolution
    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    x_axis_range = range(0, ed - st)
    cord_list = []
    pos_list = []
    pos_start = start_
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i * resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                # pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000)
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'

    subfig_r1 = subfig[0]
    subfig_r2 = subfig[1]

    ax1 = plt.subplot2grid((subfig_r1, subfig_r2+1), (0, 0), rowspan=subfig_r2, colspan=subfig_r2)
    start = int(start_ / resolution)
    end = int(end_ / resolution)
    dense_matrix_part = mat_dense[start:end, start:end]
    if score_type == 'attention':
        img = ax1.imshow(dense_matrix_part, cmap='coolwarm', vmin=0, vmax=0.2)

    else:
        h_vmin = h_range[0]
        h_vmax = h_range[-1]
        img = ax1.imshow(dense_matrix_part, cmap='Reds', vmin=np.percentile(dense_matrix_part, h_vmin),
                         vmax=np.percentile(dense_matrix_part, h_vmax))
    # img = ax1.imshow(dense_matrix_part, cmap='coolwarm', vmin = 0, vmax = 0.25)
    upper = TAD_dict['upper']
    lower = TAD_dict['lower']
    range_t = ed - st - 1
    tad_color = TAD_dict['color']
    lw = TAD_dict['linewidth']
    if len(TAD_l) != 0:
        for p in range(len(TAD_l)):
            TAD = TAD_l[p]
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st + 1
            # draw_tad_region(st_tad, ed_tad, TAD_color_1, size_v=5, size_h=5)
            if upper == True:
                draw_tad_region_upper_half(st_tad, ed_tad, range_t, tad_color, size_v= lw , size_h = lw)
            if lower == True:
                draw_tad_region_lower_half(st_tad, ed_tad, range_t, tad_color, size_v= lw, size_h = lw)
    ax1.set_xticks([])
    ax1.spines['bottom'].set_linewidth(1.6)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(1.6)
    ax1.spines['top'].set_linewidth(1.6)
    ax1.tick_params(axis='y', length=5, width=1.6)
    ax1.tick_params(axis='x', length=5, width=1.6)
    plt.xticks(cord_list, ['' for k in range(len(cord_list))], fontsize=0, rotation=90)
    plt.yticks(cord_list, y_ticks_l, fontsize=10)
    ax1.set_title(region_name, fontsize=12, pad=15.0)

    cax = plt.subplot2grid((subfig_r1, subfig_r2+1), (0, subfig_r2), rowspan=6, colspan=1)
    cbaxes = inset_axes(cax, width="30%", height="50%", loc=1)
    plt.colorbar(img, cax=cbaxes, shrink=1, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis='y', length=0, width=0)
    cax.tick_params(axis='x', length=0, width=0)
    cax.set_xticks([])
    cax.set_yticks([])

    if len(color_supply) == 0:
        color_l = ['#ED1C24', '#00A2E8', '#3F48CC', '#22B14C', '#FF7F27', '#A349A4',
                   '#ED1C24', '#00A2E8', '#3F48CC', '#22B14C', '#FF7F27', '#A349A4',
                   '#ED1C24', '#00A2E8', '#3F48CC', '#22B14C', '#FF7F27', '#A349A4',]
    else:
        color_l = color_supply

    # for i in range(1, len(window_l)+1):
    # wd = window_l[-i]
    for i in range(len(window_l)):
        wd = window_l[i]
        if len(color_supply) == []:
            color_use = color_l[i]
        else:
            if score_type == 'attention':
                color_use = '#22B14C'
            elif score_type == 'clustering':
                color_use = '#FF7F27'
            elif score_type == 'Combine':
                color_use = '#BFB500'
            else:
                if color_multi == True:
                    color_use = color_l[i]
                else:
                    color_use = '#00A2E8'
        # ax1_5 = plt.subplot2grid((11, 7), (6 + i - 1, 0), rowspan=1,colspan=6, sharex=ax1)
        ax1_5 = plt.subplot2grid((subfig_r1, subfig_r2+1), (subfig_r2 + i, 0), rowspan=1, colspan=subfig_r2, sharex=ax1)
        ax1_5.plot(list(df_pvalue_multi[wd][st:ed]), color='black')
        ax1_5.bar(x_axis_range, list(df_pvalue_multi[wd][st:ed]), label=wd, color=color_use)
        track_cut_use = track_cut[i]
        if track_cut_use != '':
            ax1_5.hlines(track_cut_use, x_axis_range[0], x_axis_range[-1], color='black', linestyles='--')
        ax1_5.spines['bottom'].set_linewidth(1.6)
        ax1_5.spines['left'].set_linewidth(1.6)
        ax1_5.spines['right'].set_linewidth(1.6)
        ax1_5.spines['top'].set_linewidth(1.6)
        ax1_5.tick_params(axis='y', length=5, width=1.2)
        ax1_5.tick_params(axis='x', length=5, width=1.2)
        ax1_5.set_ylabel(wd, fontsize=10)
        ax1_5.set_xticks(cord_list, ['' for k in range(len(cord_list))], fontsize=0, rotation=90)
        # ax1_5.set_yticks([0, 0.5, 1], ['', 0.5, 1], fontsize = 8,)
        if len(track_range) != 0:
            track_range_use = track_range[i]
            if len(track_range_use) != 0:
                plt.ylim([track_range_use[0], track_range_use[-1]])
        if ci_peak_multi != '':
            if wd not in list(ci_peak_multi.keys()):
                continue
            target_site = ci_peak_multi[wd]
            if len(target_site) != 0:
                site_use = []
                for i in range(len(target_site)):
                    if target_site[i] < st:
                        pass
                    elif target_site[i] < end:
                        site_use.append(target_site[i])
                    else:
                        break
                    plt.vlines(np.array(site_use) - st, 0, np.max(list(df_pvalue_multi[wd][st:ed])), color='black',
                               linestyle='--')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis='y', length=4, width=1.8)
    ax.tick_params(axis='x', length=4, width=1.8)
    ax.set_xticks(cord_list, x_ticks_l, fontsize=10, rotation=-30)

    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True, bbox_inches='tight')
    plt.show()
    # plt.show()
    # fig = plt.gcf() #获取当前figure
    # plt.close(fig)

def get_bin_tad_label(df_tad_mclust_fill, bin_num):
    lb_hold = 0
    bin_lb = np.zeros(bin_num) - 1
    for i in range(len(df_tad_mclust_fill)):
        domain = list(range(df_tad_mclust_fill['start'][i], df_tad_mclust_fill['end'][i] + 1))
        domain = sorted(domain)
        bin_lb[domain] = lb_hold
        lb_hold += 1
    return bin_lb


def hic_tad_and_bin_rep_compare(random_state, Chr, st, ed, hic_all, resolution, tads_res, embed_all, method_rep, method,
                                rd_method='UMAP', rd_dim=True, fgsize=(12, 4), col_num=2, arrow_draw=False, save_name=''):
    mat_hic = hic_all[Chr]
    mat_list = [mat_hic]
    mat_para_list = [{'color': 'Greys', 'range': [10, 90], 'diag': False, 'net': False,
                      'value_type': 'no-real', 'tad_color': 'self_define', 'tad_upper': True, 'tad_lower': True}]
    bin_name_l = embed_all['TADGATE'][Chr]['mat_split'][0].obs_names
    if rd_dim == True:
        embed_all_chr = embed_all[method_rep]
        if method_rep == 'GRiNCH':
            mat_rep = embed_all_chr[Chr][:, :-1]
        else:
            mat_rep = embed_all_chr[Chr]['result'][0]['bin_rep']
    if method == 'TADGATE_attention':
        df_tad_record = tads_res[method][Chr]['TADs_only']
    else:
        df_tad_record = tads_res[method][Chr]['TADs'][0]
    bin_lb = get_bin_tad_label(df_tad_record, bin_num=len(mat_hic))
    start_bin = 0
    TAD_l = TL.get_tad_list_in_target_ranges(st, ed, df_tad_record, resolution, start_bin, pos_type='bin')
    TAD_list = [TAD_l]

    df_bin_scatter, color_use = compare_tads_and_bin_rep2(random_state, st, ed, mat_list, mat_para_list, TAD_list,
                                                             mat_rep, bin_lb, rd_method, bin_name_l,
                                                             Chr, resolution, col_num, rd_dim=rd_dim, title_name_l='',
                                                             fgsize=fgsize, bin_size=20, save_name=save_name,
                                                             ori='h', add_tor=True, arrow_draw=arrow_draw, legend='No')
    return bin_lb[st:ed], TAD_l, df_bin_scatter, color_use


def get_attention_valley_peak_nearby_map(TADGATE_res_all, TADGATE_tads_all, expand, pos_type,
                                         map_type = 'Hi-C', dist=3, target_chr_l=[]):
    """
    Get the nearby Hi-C map or attention map around attention valley or peak
    :param TADGATE_res_all:  dict, TADGATE result
    :param TADGATE_tads_all: dict, TADs from TADGATE
    :param expand: int, expand size
    :param pos_type: str, attention peak or attention valley
    :param map_type: str, 'Hi-C' or attention
    :param dist: int, distance threshold for calling peak or valley
    :param target_chr_l: list, target chromosome list
    :return: mat_combine: numpy array, combined map; vec_combine: numpy array, combined vector; count_combine: int, count
    """
    mat_combine = np.zeros([2 * expand + 1, 2 * expand + 1])
    vec_combine = np.zeros([1, 2 * expand + 1])
    count_combine = 0
    for Chr in list(TADGATE_res_all.keys()):
        if len(target_chr_l) != 0:
            if Chr not in target_chr_l:
                continue
        if map_type == 'Hi-C':
            mat_use = copy.deepcopy(TADGATE_res_all[Chr]['mat_split'][0].X)
        elif map_type == 'attention':
            mat_use = copy.deepcopy(TADGATE_res_all[Chr]['result'][0]['attention_map'])
        att_profile_smooth = TADGATE_tads_all[Chr]['att_profile'][0]

        if pos_type == 'attention peak':
            pos_l = signal.find_peaks(np.array(att_profile_smooth), distance=dist)[0]
        elif pos_type == 'attention valley':
            pos_l = signal.find_peaks(-np.array(att_profile_smooth), distance=dist)[0]
        for ind in pos_l:
            if ind <= expand or ind >= len(mat_use) - expand:
                continue
            mat_add = copy.deepcopy(mat_use[ind - expand: ind + expand + 1, ind - expand: ind + expand + 1])
            if np.max(mat_add) == 0:
                continue
            mat_add_norm = mat_add / np.nanmax(mat_add)
            vec_add = copy.deepcopy(att_profile_smooth[ind - expand: ind + expand + 1])
            if np.max(vec_add) == 0:
                continue
            vec_norm = vec_add / np.max(vec_add)

            mat_combine += mat_add_norm
            vec_combine += vec_norm
            count_combine += 1
    return mat_combine, vec_combine[0], count_combine


def draw_map_profile(map_type, pos_type, mat_combine, vec_combine, resolution, mat_color,
                     fgsize = (5, 6), save_name='', tick_adj=False):
    expand = int((len(mat_combine) - 1) / 2)
    plt.figure(figsize=(fgsize[0], fgsize[-1]))
    ax1 = plt.subplot2grid((8, 7), (0, 0), rowspan=6, colspan=6)
    dense_matrix_part = mat_combine
    img = ax1.imshow(dense_matrix_part, cmap=mat_color, vmin=np.percentile(dense_matrix_part, 20),
                     vmax=np.percentile(dense_matrix_part, 80))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(1.6)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(1.6)
    ax1.spines['top'].set_linewidth(1.6)
    ax1.tick_params(axis='y', length=5, width=1.6)
    ax1.tick_params(axis='x', length=5, width=1.6)
    # plt.xticks(cord_list, x_ticks_l, fontsize = 0,  rotation = 90)
    # plt.yticks(cord_list, y_ticks_l, fontsize = 10)
    ax1.set_xticks([0, int(expand / 2), expand, int(expand * 3 / 2), 2 * expand], ['', '', '', '', ''],
                   fontsize=0, rotation=0)
    ax1.set_title('Aggregrated ' + map_type + ' around ' + pos_type, fontsize=12, pad=15.0)

    cax = plt.subplot2grid((8, 7), (0, 6), rowspan=6, colspan=1)
    cbaxes = inset_axes(cax, width="30%", height="50%", loc=1)
    plt.colorbar(img, cax=cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis='y', length=0, width=0)
    cax.tick_params(axis='x', length=0, width=0)
    cax.set_xticks([])
    cax.set_yticks([])

    if pos_type == 'attention peak':
        vec_color = '#D65F4D'
    elif pos_type == 'attention valley':
        vec_color = '#4392C3'
    ax1_5 = plt.subplot2grid((8, 7), (6, 0), rowspan=1, colspan=6, sharex=ax1)
    ax1_5.plot(vec_combine, color=vec_color, linewidth=5)
    # ax1_5.bar(x_axis_range, list(df_pvalue_multi[wd][st:ed]), label=wd, color='#00A2E8')
    ax1_5.spines['bottom'].set_linewidth(1.6)
    ax1_5.spines['left'].set_linewidth(1.6)
    ax1_5.spines['right'].set_linewidth(1.6)
    ax1_5.spines['top'].set_linewidth(1.6)
    ax1_5.tick_params(axis='y', length=5, width=1.6)
    ax1_5.tick_params(axis='x', length=5, width=1.6)
    plt.ylim([np.min(vec_combine) * 0.8, np.max(vec_combine) * 1.1])
    ax1_5.set_ylabel('attention \n sum', fontsize = 10)
    # ax1_5.set_xticks(cord_list, x_ticks_l, fontsize = 0, rotation = 90)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis='y', length=4, width=1.8)
    ax.tick_params(axis='x', length=4, width=1.8)
    if tick_adj == False:
        ax.set_xticks([0, int(expand / 2), expand, int(expand * 3 / 2), 2 * expand],
                      [0, int(expand / 2), expand, int(expand * 3 / 2), 2 * expand])
    else:
        x1 = int(expand * resolution / 1000)
        x2 = int(expand / 2 * resolution / 1000)
        if pos_type == 'Insulation':
            x3 = 'Insulation \n valley'
        elif pos_type == 'attention valley':
            x3 = 'Attention \n valley'
        elif pos_type == 'attention peak':
            x3 = 'Attention \n peak'
        x_tick_l = ['-' + str(x1) + 'kb', '-' + str(x2) + 'kb', x3, str(x2) + 'kb', str(x1) + 'kb']
        ax.set_xticks([0, int(expand / 2), expand, int(expand * 3 / 2), 2 * expand], x_tick_l);
    if save_name != '':
        plt.savefig(save_name, format='svg', transparent=True)
    plt.show()


def get_hic_map_original_and_imputed_compare(st, ed, mat_hic, mat_imputed, QN=False):
    mat_hic_part = copy.deepcopy(mat_hic[st:ed, st:ed])
    mat_imputed_part = copy.deepcopy(mat_imputed[st:ed, st:ed])
    mat_combine_part = np.zeros([len(mat_hic_part), len(mat_hic_part)])

    mat_hic_part_norm = (mat_hic_part - np.min(mat_hic_part)) / (np.max(mat_hic_part) - np.min(mat_hic_part))
    mat_imputed_part_norm = (mat_imputed_part - np.min(mat_imputed_part)) / (
                np.max(mat_imputed_part) - np.min(mat_imputed_part))

    if QN == True:
        import qnorm
        hic_vec = mat_hic_part_norm.ravel()
        imputed_vec = mat_imputed_part_norm.ravel()
        df_norm = pd.DataFrame()

        df_norm['hic'] = hic_vec
        df_norm['imputed'] = imputed_vec
        df_norm2 = qnorm.quantile_normalize(df_norm, axis=1)

        mat_hic_part_use = np.array(df_norm2['hic']).reshape(len(mat_hic_part_norm), len(mat_hic_part_norm))
        mat_imputed_part_use = np.array(df_norm2['imputed']).reshape(len(mat_imputed_part_norm),
                                                                     len(mat_imputed_part_norm))
    else:
        mat_hic_part_use = mat_hic_part_norm
        mat_imputed_part_use = mat_imputed_part_norm

    mat_combine_part = np.zeros([len(mat_imputed_part_use), len(mat_imputed_part_use)])
    mat_combine_part += np.tril(mat_hic_part_use)
    mat_combine_part += np.triu(mat_imputed_part_use)

    mat_hic_whole = np.zeros([len(mat_hic), len(mat_hic)])
    mat_hic_whole[st:ed, st:ed] = mat_hic_part_use

    mat_imputed_whole = np.zeros([len(mat_hic), len(mat_hic)])
    mat_imputed_whole[st:ed, st:ed] = mat_imputed_part_use

    mat_combine_whole = np.zeros([len(mat_hic), len(mat_hic)])
    mat_combine_whole[st:ed, st:ed] = mat_combine_part
    return mat_hic_whole, mat_imputed_whole, mat_combine_whole




