import copy
import os
import pickle
import scipy
import random
import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from contextlib import suppress
from scipy.stats import zscore
from skimage.metrics import structural_similarity as ssim

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from . import TADGATE_pyG




def save_data(file, objects):
    """
    Save data to file
    :param file: str, Path and file name to save
    :param objects: Any object in python
    :return: None
    """
    save_file = open(file, 'wb')
    pickle.dump(objects, save_file, 2)


def read_save_data(file):
    """
    Read data from file
    :param file: str, Path and file name to load
    :return: Objects to load
    """

    read_file = open(file, 'rb')
    objects = pickle.load(read_file)
    read_file.close()
    return objects


def chr_cut(chr_length, chr_symbol, resolution):
    """
    Cut chromosome into bins
    :param chr_length: int, length of chromosome
    :param chr_symbol: str, chromosome symbol
    :param resolution: int, resolution of Hi-C contact map
    :return: name_list: list, bin symbol along chromosome
    """
    start_pos = 0
    start = []
    end = []
    name_list = []
    while (start_pos + resolution) <= chr_length:
        start.append(start_pos)
        end.append(start_pos + resolution)
        start_pos += resolution
    start.append(start_pos)
    end.append(chr_length)
    for i in range(len(start)):
        name_list.append(chr_symbol + ':' + str(start[i]) + '-' + str(end[i]))
    return name_list


def SparseMatrixToDense(df_mat_sparse, bin_num, mat_half = False):
    """
    Convert sparse matrix to dense matrix
    :param df_mat_sparse: pandas dataframe, sparse matrix
    :param bin_num: int, number of bins
    :param mat_half: bool, whether to use half matrix
    :return: mat_dense: numpy array, dense matrix
    """
    df_mat_sparse.columns = ['bin1', 'bin2', 'value']
    row = np.array(df_mat_sparse['bin1'])
    col = np.array(df_mat_sparse['bin2'])
    val = np.array(df_mat_sparse['value'])
    mat_hic_sparse = scipy.sparse.csr_matrix((val, (row, col)), shape=(bin_num, bin_num))
    mat_dense_up = mat_hic_sparse.toarray()
    if mat_half == True:
        mat_dense_low = mat_dense_up.T
        mat_dense_diag = np.diag(np.diag(mat_dense_up))
        mat_dense = mat_dense_up + mat_dense_low - mat_dense_diag
    else:
        mat_dense = mat_dense_up
    return mat_dense


def LoadHicMat(mat_file, bin_num, mat_type='dense', mat_half = False):
    """
    Load Hi-C contact map
    :param mat_file: str, path of Hi-C matrix file
    :param bin_num: int, total number of bins of the Hi-C matrix
    :param mat_type:  str, type of Hi-C matrix, 'dense' or 'sparse'
    :return: mat_hic: numpy array, Hi-C contact map
    """
    if os.path.exists(mat_file) == False:
        print('Hi-C matrix do not exit!')
        return None
    if mat_type == 'dense':
        df_mat_dense = pd.read_csv(mat_file, sep='\t', header=None)
        mat_hic = np.array(df_mat_dense.values)
    if mat_type == 'sparse':
        df_mat_sparse = pd.read_csv(mat_file, sep='\t', header=None)
        mat_hic = SparseMatrixToDense(df_mat_sparse, bin_num, mat_half =mat_half)
    return mat_hic


def load_chromosome_size(chrom_size_file):
    """
    Load chromosome size
    :param chrom_size_file: str, path of chromosome size file
    :return: chr_size: dict, chromosome size
    """
    df_chr_size = pd.read_csv(chrom_size_file, sep='\t', header=None)
    chr_size = {}
    for i in range(len(df_chr_size[0])):
        Chr = df_chr_size[0][i]
        size = df_chr_size[1][i]
        chr_size[Chr] = size
    return chr_size

def get_zero_row(mat_raw, method, cum_per = 5, cen_cut = 2000000,
                 cen_ratio_per = 95, resolution = 50000):

    """
    Get bad rows in Hi-C contact map according to three methods:
    'row': get rows with accumulated zero counts
    'center': get rows with zero count in main diagonal
    'combines': combine two methods above
    :param mat_raw: numpy array, Hi-C contact map
    :param method: str, method to use
    :param cum_per: int, quantile of accumulated count, used to filter bins with rare counts
    :param cen_cut: int, range for center sparse ratio calculation
    :param cen_ratio_per: int, quantile of sparse ratio around main diagonal
    :param resolution: int, resolution of Hi-C contact map
    :return: numpy array, bad rows with few counts in Hi-C contact map
    """
    if method == 'combine':
        cl_count = np.percentile(np.sum(mat_raw, axis = 0), cum_per)
        cum_row = np.where(np.sum(mat_raw, axis = 0) <= cl_count)[0]
        row_center_zero_rl = []
        cut = int(cen_cut / resolution)
        for i in range(len(mat_raw)):
            if i <= cut:
                row_test = mat_raw[i,0:i+cut]
            elif i >= len(mat_raw) - cut:
                row_test = mat_raw[i,i-cut:]
            else:
                row_test = mat_raw[i,i-cut:i+cut]
            row_center_zero_rl.append(np.sum(row_test == 0) / len(row_test))
        center_cut = np.percentile(np.array(row_center_zero_rl), cen_ratio_per)
        center_row = np.where(np.array(row_center_zero_rl) >= center_cut)[0]
        row_all = set(cum_row).union(set(center_row))
        row_all = np.array(sorted(list(row_all)))
    elif method == 'row':
        row_all = np.where(np.sum(mat_raw, axis = 1) == 0)[0]
    elif method == 'center':
        row_all = np.where(np.diag(mat_raw, k=0) == 0)[0]
    return row_all

def build_spatial_net(adata, row_bad, expand_num):
    """
    Build spatial network for GAT
    :param adata: scanpy data, scanpy data with node feature for each sample
    :param row_bad: numpy array, bad rows with few counts in Hi-C contact map
    :param expand_num: int, number of neighborhoods used in GAT
    :return: Spatial_Net: pandas dataframe, Spatial Network for GAT; Net_sparse: sparse Spatial Network for visualization
    """
    Spatial_Net = pd.DataFrame(columns=['Cell1', 'Cell2', 'Distance'])
    Net_sparse = pd.DataFrame(columns=['bin1', 'bin2', 'edge'])
    mat_spatial = np.zeros([adata.obs_names.shape[0], adata.obs_names.shape[0]])
    for k in range(1, expand_num + 1):
        mat_spatial += np.diag(np.ones([len(mat_spatial) - k]), k = k)
    mat_spatial += mat_spatial.T
    mat_spatial += np.diag(np.ones([len(mat_spatial)]), k =0)
    for ind in row_bad:
        mat_spatial[ind, :] = 0
        mat_spatial[:, ind] = 0
    mat_spatial_sparse = scipy.sparse.coo_matrix(mat_spatial)
    Spatial_Net = pd.DataFrame(columns=['Cell1', 'Cell2', 'Distance'])
    Net_sparse = pd.DataFrame(columns=['bin1', 'bin2', 'edge'])
    Spatial_Net['Cell1'] = adata.obs_names[mat_spatial_sparse.row]
    Spatial_Net['Cell2'] = adata.obs_names[mat_spatial_sparse.col]
    Spatial_Net['Distance'] = mat_spatial_sparse.data
    Net_sparse['bin1'] = mat_spatial_sparse.row
    Net_sparse['bin2'] = mat_spatial_sparse.col
    Net_sparse['edge'] = mat_spatial_sparse.data
    return Spatial_Net, Net_sparse


def Batch_Data_two_dim(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y']):
    """
    Split spatial data into batches
    :param adata: scanpy data, scanpy data with node feature and spatial network
    :param num_batch_x: int, number of batches along x-axis
    :param num_batch_y: int, number of batches along y-axis
    :param spatial_key: list, key for spatial coordinates
    :return: Batch_list: list, list of batches
    """
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x+1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y+1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    return Batch_list


def Batch_Data_SingleDim(adata, num_batch, spatial_key='X'):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_coor = [np.percentile(Sp_df, (1 / num_batch) * x * 100) for x in range(num_batch + 1)]

    Batch_list = []
    for it in range(num_batch):
        min_coor = batch_coor[it]
        max_coor = batch_coor[it + 1]
        temp_adata = adata.copy()
        temp_adata = temp_adata[temp_adata.obs[spatial_key].map(lambda x: min_coor <= x <= max_coor)]
        Batch_list.append(temp_adata)

    return Batch_list



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


def get_dense_hic_mat(Net_sparse, length, resolution = None):
    """
    Build dense Hi-C matrix from sparse Hi-C matrix according to bin symbol along chromosome
    :param Net_sparse:  pandas dataframe, sparse Hi-C matrix
    :param length:  int, number of rows or columns for matrix
    :return:  mat_dense: numpy array, dense Hi-C network
    """
    if resolution == None:
        row = list(Net_sparse['bin1'])
        col = list(Net_sparse['bin2'])
    else:
        row = list(np.array((Net_sparse['bin1']) / resolution).astype('int'))
        col = list(np.array((Net_sparse['bin2']) / resolution).astype('int'))
    val = list(Net_sparse['value'])
    mat_hic_sparse = scipy.sparse.csr_matrix((val, (row, col)), shape = (length, length))
    mat_dense_up = mat_hic_sparse.toarray()
    if (mat_dense_up == mat_dense_up.T).all():
        return mat_dense_up
    mat_dense_low = mat_dense_up.T
    mat_dense_diag = np.diag(np.diag(mat_dense_up))
    mat_dense = mat_dense_up + mat_dense_low - mat_dense_diag
    return mat_dense

def get_matrix_split(mat_hic, length, window, bin_name_list, resolution, diag_cut, cut = False):
    """
    Split Hi-C contact map into windows
    :param mat_hic: numpy array, Hi-C contact map
    :param length: int, length of Hi-C contact map
    :param window: int, size of window
    :param bin_name_list: list, bin symbol along chromosome
    :param resolution: int, resolution of Hi-C contact map
    :param diag_cut:  int, number of diagonals to cut
    :param cut:  bool, whether to cut Hi-C contact map
    :return: range_l: list, range of each window;
             mat_split_all: dict, Hi-C contact map for each window;
             row_bad_all: dict, bad rows for each window
    """
    range_l = []
    mat_split_all = {}
    row_bad_all = {}
    n = length
    if diag_cut == 'all':
        diag_cut = 0
    for i in range(int(np.ceil(2*n/window))):
        st = max(0, window//2*i-window//4)
        mid1 = st + window//2*i-max(0,window//2*i-window//4)
        mid2 = st + window//2*(i+1)-max(0,window//2*i-window//4)
        ed = min(length,window//2*i+window-window//4)
        if st == 0:
            ed = st + window
        if ed == n:
            st = ed - window
            mid2 = n
        range_l.append((st,mid1, mid2, ed))
        mat_part = mat_hic[st:ed, st:ed]
        if cut:
            mat_diff = np.zeros([window, window])
            if diag_cut <= window / 2:
                for k in range(1, diag_cut):
                    mat_diff += np.diag(np.diag(mat_part, k=k), k=k)
                mat_diff += mat_diff.T
                mat_diff += np.diag(np.diag(mat_part, k=0), k=0)
                mat_part = mat_diff
            else:
                for k in range(diag_cut, window):
                    mat_diff += np.diag(np.diag(mat_part, k=k), k=k)
                mat_diff += mat_diff.T
                mat_part = mat_part - mat_diff
        df_mat_part = pd.DataFrame(mat_part)
        df_mat_part.columns = bin_name_list[st:ed]
        df_mat_part.index = bin_name_list[st:ed]
        adata = sc.AnnData(df_mat_part)
        mat_split_all[i] = adata
        row_bad = get_zero_row(mat_part, method = 'row', resolution = resolution)
        row_bad_all[i] = row_bad
        if ed == n:
            break
    return range_l, mat_split_all, row_bad_all


def get_split_mat_spatial_network(mat_split_all, row_bad_all, expand_num):
    """
    Build spatial network for each window
    :param mat_split_all: dict, Hi-C contact map for each window
    :param row_bad_all: dict, bad rows for each window
    :param expand_num: int, number of neighborhoods used in GAT
    :return: mat_split_all: dict, Hi-C contact map for each window with spatial network;
             spatial_net_all: dict, spatial network for each window
    """
    spatial_net_all = {}
    for i in list(mat_split_all.keys()):
        row_bad = row_bad_all[i]
        adata = mat_split_all[i]
        Spatial_Net, Net_sparse = build_spatial_net(adata, row_bad, expand_num)
        adata.uns['Spatial_Net'] = Spatial_Net
        adata.obsm['genome_order'] = np.array(range(len(adata.obs_names)))
        mat_split_all[i] = adata
        spatial_net_all[i] = Net_sparse
    return mat_split_all, spatial_net_all


def get_diagnal_near_mask(window, wd, diag_cut = 0):
    """
    Get mask for diagonals. The mask is used to filter out the pixels near diagonal or far from diagonal.
    if diag_cut > 0, it will be a mask for pixels near diagonal to diag_cut. The region between diag_cut and wd will be enhanced.
    if diag_cut = 0, it will be a mask for pixels far from diagonal from wd to window.
    :param window: int, size of window
    :param wd: int, number of diagonals to cut out, wd to window will be zero
    :param diag_cut:  int, number of diagonals to cut inner, diag_cut to wd will be zero
    :return:  mat_mask: numpy array, mask for diagonals
    """
    mat_mask = np.zeros([window, window])
    wd = np.min([wd, window - 1])
    if diag_cut != 0:
        if wd <= window / 2:
            for i in range(diag_cut, wd):
                mat_mask += np.diag(np.ones(window - i), k = i)
            mat_mask += mat_mask.T
        else:
            for i in range(1, diag_cut):
                mat_mask += np.diag(np.ones(window - i), k=i)
            for i in range(wd, window):
                mat_mask += np.diag(np.ones(window - i), k=i)

            mat_mask += mat_mask.T
            mat_mask += np.diag(np.ones(window - 0), k=0)
            mat_mask = np.ones([window, window]) - mat_mask
    else:
        if wd <= window / 2:
            for i in range(0, wd):
                mat_mask += np.diag(np.ones(window - i), k = i)
            mat_mask += mat_mask.T
            mat_mask -= np.diag(np.ones(window), k = 0)
        else:
            for i in range(wd, window):
                mat_mask += np.diag(np.ones(window - i), k=i)
            mat_mask += mat_mask.T
            mat_mask = np.ones([window, window]) - mat_mask
    return mat_mask


def ignal_zero_pos_in_mask_mat(mat_part, mat_mask0, impute_range):
    """
    Filter out pixels far from diagonal in mask matrix
    :param mat_part:  numpy array, Hi-C contact map
    :param mat_mask0: numpy array, mask matrix
    :param impute_range:  int, number of diagonals to cut out
    :return:  mat_mask0_new: numpy array, mask matrix
    """
    mat_mask0_new = copy.deepcopy(mat_mask0)
    k = impute_range
    n = len(mat_part)
    indices = np.abs(np.arange(n) - np.arange(n)[:, np.newaxis]) > k
    mat_part[indices] = 1
    mat_mask0_new[np.where(mat_part == 0)] = 0.5
    return mat_mask0_new


def mat_mask_batch_to_device(mat_mask0, device, batch = 1):
    """
    Transfer mask to device
    :param mat_mask0: numpy array, mask for diagonals
    :param device: torch device, device to use
    :param batch: int, number of batch
    :return: mat_mask: torch tensor, mask for diagonals
    """
    mat_mask = mat_mask0
    if batch != 1:
        for j in range(1, batch):
            mat_mask = np.concatenate([mat_mask, mat_mask0], axis = 0)
    mat_mask = mat_mask.astype(bool)
    mat_mask = torch.from_numpy(mat_mask)
    mat_mask = mat_mask.to(device)
    return mat_mask

def Transfer_pytorch_Data(adata):
    """
    Transfer scanpy data to pytorch data
    :param adata: scanpy data, scanpy data with node feature and spatial network
    :return: data: pytorch data, pytorch data with node feature and spatial network
    """
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(
        adata.n_obs, adata.n_obs))


    #G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if len((np.unique(adata.uns['Spatial_Net']['Distance']))) == 1:
        if type(adata.X) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    else:
        if type(adata.X) == np.ndarray:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), edge_weight = torch.FloatTensor(np.array(G_df['Distance'])), x=torch.FloatTensor(adata.X))  # .todense()
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), edge_weight = torch.FloatTensor(np.array(G_df['Distance'])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def mat_diag_extract(mat_use, diag_target):
    n = mat_use.shape[0]
    m = 2 * diag_target + 1
    # 对矩阵进行填充，以便处理边界情况
    padded_mat = np.pad(mat_use, ((0, 0), (diag_target, diag_target)), mode='constant', constant_values=-1)
    # 生成列索引
    col_indices = np.arange(n)[:, None] + np.arange(m)
    # 使用高级索引提取对角线附近的元素
    mat_extract = padded_mat[np.arange(n)[:, None], col_indices]
    return mat_extract, col_indices

def reconstruct_original_map(mat_use, mat_ext, col_indices, diag_target):
    # 初始化与 mat_use 相同形状的新矩阵 mat_rec
    mat_rec = np.full_like(mat_use, fill_value=0)
    padded_mat = np.pad(mat_rec, ((0, 0), (diag_target, diag_target)), mode='constant', constant_values=-1)
    for i in range(len(mat_ext)):
        padded_mat[i, col_indices[i]] = mat_ext[i]
    mat_rec = padded_mat[:, diag_target : diag_target + len(mat_use)]
    return mat_rec


def neighbor_ave_gpu(A, pad):
    """
    Average pooling for Hi-C contact map
    :param A: numpy array, Hi-C contact map
    :param pad: int, padding for convolution
    :return: numpy array, Hi-C contact map after average pooling
    """
    if pad == 0:
        return torch.from_numpy(A).float().cuda()
    ll = pad * 2 + 1
    conv_filter = torch.ones(1, 1, ll, ll).cuda()
    B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().cuda(), conv_filter, padding=pad * 2)
    return (B[0, 0, pad:-pad, pad:-pad] / float(ll * ll)).cpu().numpy()


def random_walk_gpu(A, rp):
    """
    Random walk for Hi-C contact map
    :param A: numpy array, Hi-C contact map
    :param rp: float, restart probability of random walk
    :return:  numpy array, Hi-C contact map after random walk
    """
    if rp == 1:
        return A
    ngene, _ = A.shape
    A = torch.from_numpy(A).float().cuda()
    A = A - torch.diag(torch.diag(A))
    A = A + torch.diag(torch.sum(A, 0) == 0).float()
    P = torch.div(A, torch.sum(A, 0))
    Q = torch.eye(ngene).cuda()
    I = torch.eye(ngene).cuda()
    for i in range(30):
        Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
        delta = torch.norm(Q - Q_new, 2)
        Q = Q_new
        if delta < 1e-6:
            break
    return Q.cpu().numpy()


def impute_gpu(A, pad, rp):
    """
    Impute Hi-C contact map with average pooling and random walk
    :param A: numpy array, Hi-C contact map
    :param pad: int, padding for convolution
    :param rp: float, restart probability of random walk
    :return: numpy array, Hi-C contact map after imputation
    """
    ngene, _ = A.shape
    A = neighbor_ave_gpu(A, pad)
    if rp == -1:
        Q = A[:]
    else:
        Q = random_walk_gpu(A, rp)
    return Q.reshape(ngene, ngene)

def create_batch_mask(batch, adata, mat_mask, spatial_key):
    """
    Create mask for batch
    :param batch: scanpy data, scanpy data with node feature and spatial network
    :param adata: scanpy data, scanpy data with node feature and spatial network
    :param mat_mask: numpy array, mask for diagonals
    :param spatial_key: list, key for spatial coordinates
    """
    batch_indices = batch.batch_indices
    batch_mask = mat_mask[batch_indices]
    return batch_mask

def train_TADGATE(adata, mat_mask, scale_f, layer_node1, layer_node2, lr, weight_decay,
                  num_epoch, device, embed_attention = False, batch_use = False,
                  batch_list = [], mask_list = [],
                  seed = 666, verbose=True):
    """
    Train TADGATE model
    :param adata: scanpy data, scanpy data with node feature and spatial network
    :param mat_mask: numpy array, mask for diagonals
    :param scale_f: int, scale factor for node embedding scaling with dimension
    :param layer_node1: int, number of nodes in the first layer
    :param layer_node2: int, number of nodes in the second layer
    :param lr: float, learning rate
    :param weight_decay: float, weight decay
    :param num_epoch: int, number of epochs
    :param device: torch device, device to use
    :param embed_attention: bool, whether to use attention mechanism in embedding layer
    :param batch_use: bool, whether to use batch training
    :param num_batch_x: int, number of batches along x-axis
    :param seed: int, random seed
    :param verbose: bool, whether to print training process
    :return: model: TADGATE model, trained TADGATE model
             loss_record: list, loss record during training
    """
    # seed_everything()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if not batch_use:
        data = Transfer_pytorch_Data(adata)
        if verbose:
            print('Train TADGATE....')
        model = TADGATE_pyG.TADGATE(in_channels = data.x.shape[1], layer1_nodes = layer_node1, embed_nodes = layer_node2).to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        loss_record = []
        for epoch in tqdm(range(1, num_epoch + 1), leave=False):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index, scale_f, embed_attention=embed_attention, return_att = False)
            #if epoch < 0.1*num_epoch:
                #out_new = (out.data + out.data.T) / 2
                #out_new[out_new < 0] = 0
                #out.data = out_new.data
            out_mask = torch.masked_select(out, mat_mask)
            data_mask = torch.masked_select(data.x, mat_mask)

            loss = F.mse_loss(data_mask, out_mask)
            loss_record.append(loss.to('cpu').detach().numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            scheduler.step(loss)
    else:

        for temp in batch_list:
            temp.to(device)

        from torch_geometric.loader import DataLoader
        # batch_size=1
        loader = DataLoader(batch_list, batch_size=1, shuffle = True)

        model = TADGATE_pyG.TADGATE(in_channels=batch_list[0].x.shape[1], layer1_nodes=layer_node1, embed_nodes=layer_node2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=False,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08)
        loss_record = []
        for epoch in tqdm(range(1, num_epoch + 1)):
            total_loss = 0
            for batch in loader:
                model.train()
                optimizer.zero_grad()
                batch_mask = create_batch_mask(batch, mat_mask)
                z, out = model(batch.x, batch.edge_index, scale_f, embed_attention=embed_attention, return_att=False)
                # if epoch < 0.1*num_epoch:
                # out_new = (out.data + out.data.T) / 2
                # out_new[out_new < 0] = 0
                # out.data = out_new.data
                out_mask = torch.masked_select(out, batch_mask)
                data_mask = torch.masked_select(batch.x, batch_mask)

                loss = F.mse_loss(data_mask, out_mask)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            scheduler.step(total_loss.item())
            loss_record.append(total_loss.item())

    return model, loss_record


def train_TADGATE_weight(adata, mat_weight, scale_f, layer_node1, layer_node2, lr, weight_decay,
                         num_epoch, device, embed_attention = False, batch_use = False,
                         batch_list = [], mask_list = [],
                         seed = 666, verbose=True):
    """
    Train TADGATE model with pre-defined weight near diagonal
    :param adata: scanpy data, scanpy data with node feature and spatial network
    :param mat_weight: numpy array, weight for pixels near diagonal of Hi-C contact map
    :param scale_f: int, scale factor for node embedding scaling with dimension
    :param layer_node1: int, number of nodes in the first layer
    :param layer_node2: int, number of nodes in the second layer
    :param lr: float, learning rate
    :param weight_decay: float, weight decay
    :param num_epoch: int, number of epochs
    :param device: torch device, device to use
    :param embed_attention: bool, whether to use attention mechanism in embedding layer
    :param batch_use: bool, whether to use batch training
    :param num_batch_x: int, number of batches along x-axis
    :param seed: int, random seed
    :param verbose: bool, whether to print training process
    :return:  model: TADGATE model, trained TADGATE model
              loss_record: list, loss record during training
    """

    # seed_everything()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if not batch_use:
        data = Transfer_pytorch_Data(adata)
        if verbose:
            print('Train TADGATE....')
        model = TADGATE_pyG.TADGATE(in_channels = data.x.shape[1], layer1_nodes = layer_node1, embed_nodes = layer_node2).to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        loss_record = []
        for epoch in tqdm(range(1, num_epoch + 1)):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index, scale_f, embed_attention=embed_attention, return_att = False)
            #if epoch < 0.1*num_epoch:
                #out_new = (out.data + out.data.T) / 2
                #out_new[out_new < 0] = 0
                #out.data = out_new.data
            data_weight = data.x * (torch.from_numpy(mat_weight).to(device))
            out_weight = out * (torch.from_numpy(mat_weight).to(device))
            loss = F.mse_loss(data_weight, out_weight)

            #loss = F.mse_loss(data.x, out)  # F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss_record.append(loss.to('cpu').detach().numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            scheduler.step(loss)
    else:

        for temp in batch_list:
            temp.to(device)

        from torch_geometric.loader import DataLoader
        loader = DataLoader(batch_list, batch_size=1, shuffle=False)

        model = TADGATE_pyG.TADGATE(in_channels = batch_list[0].x.shape[1], layer1_nodes=layer_node1,
                                    embed_nodes = layer_node2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=0, eps=1e-08)
        loss_record = []
        for epoch in tqdm(range(1, num_epoch + 1), leave=False):
            total_loss = 0.0
            mask_count = 0
            for batch in loader:
                model.train()
                optimizer.zero_grad()

                batch_mask = mask_list[mask_count]
                mask_count += 1
                z, out = model(batch.x, batch.edge_index, scale_f, embed_attention=embed_attention, return_att=False)

                data_weight = batch.x * torch.from_numpy(batch_mask).to(device)
                out_weight = out * torch.from_numpy(batch_mask).to(device)
                loss = F.mse_loss(data_weight, out_weight)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()

            scheduler.step(total_loss)
            loss_record.append(total_loss)
    return model, loss_record


def get_attention_mat(att, length):
    """
    Get attention matrix from attention tensor
    :param att: torch tensor, attention tensor
    :param length: int, number of rows or columns for matrix
    :return:  df_att_mat: pandas dataframe, sparse attention matrix;
              mat_dense: numpy array, dense attention matrix
    """
    att_index = att[0].to('cpu').detach().numpy()
    att_value = att[1].to('cpu').detach().numpy()
    att_value = att_value.reshape(len(att_value))
    df_att_mat = pd.DataFrame(columns = ['bin1', 'bin2', 'value'])
    df_att_mat['bin1'] = att_index[0]
    df_att_mat['bin2'] = att_index[1]
    df_att_mat['value'] = att_value
    row = list(df_att_mat['bin1'])
    col = list(df_att_mat['bin2'])
    val = list(df_att_mat['value'])
    mat_hic_sparse = scipy.sparse.csr_matrix((val, (row, col)), shape = (length, length))
    mat_dense = mat_hic_sparse.toarray()
    mat_dense = mat_dense.T
    return df_att_mat, mat_dense


def TADGATE_use(adata, model, device, scale_f, embed_attention, return_att = True):
    """
    Use TADGATE model to impute Hi-C contact map and get bin embedding
    :param adata: scanpy data, scanpy data with node feature and spatial network
    :param model: TADGATE model, trained TADGATE model
    :param device: torch device, device to use
    :param scale_f:  int, scale factor for node embedding scaling with dimension
    :param embed_attention: bool, whether to use attention mechanism in embedding layer
    :param return_att: bool, whether to return attention matrix
    :return:  mat_imputed: numpy array, imputed Hi-C contact map;
              mat_imputed_sym: numpy array, symmetric imputed Hi-C contact map;
              mat_rep: numpy array, bin embedding;
              mat_att1: numpy array, attention matrix of layer 1, optional;
              mat_att2: numpy array, attention matrix of layer 2, optional.
    """
    att1 = ''
    att2 = ''
    data = Transfer_pytorch_Data(adata)
    data = data.to(device)
    if return_att== True and embed_attention == False:
        z, att1, out = model(data.x, data.edge_index, scale_f, embed_attention=embed_attention, return_att = return_att)
    elif return_att == True and embed_attention == True:
        z, att1, att2, out = model(data.x, data.edge_index, scale_f, embed_attention=embed_attention, return_att = return_att)
    else:
        z, out = model(data.x, data.edge_index, scale_f, embed_attention=embed_attention, return_att = return_att)
    # imputed matrix
    df_imputed_Data = pd.DataFrame(out.to('cpu').detach().numpy(), index=adata.obs_names, columns=adata.var_names)
    mat_imputed  = np.array(df_imputed_Data.values)

    if mat_imputed.shape[0] == mat_imputed.shape[-1]:
        mat_imputed_sym = (mat_imputed + mat_imputed.T) / 2
        mat_imputed_sym[mat_imputed_sym < 0 ] = 0
    else:
        mat_imputed_sym = []
    # bin low-d representation
    df_rep_Data = pd.DataFrame(z.to('cpu').detach().numpy())
    mat_rep = df_rep_Data.values
    # attention mat of layer 1
    length = mat_rep.shape[0]
    if att1 != '':
        df_att_mat1, mat_att1 = get_attention_mat(att1, length)
    if att2 != '':
        df_att_mat2, mat_att2 = get_attention_mat(att2, length)
    if return_att == True and embed_attention == True:
        return mat_imputed, mat_imputed_sym, mat_rep, mat_att1, mat_att2
    elif return_att == True and embed_attention == False:
        return mat_imputed, mat_imputed_sym, mat_rep, mat_att1
    else:
         return mat_imputed, mat_imputed_sym, mat_rep


def get_tad_list_in_target_ranges(st, ed, df_tad_use_1, resolution, start_bin, pos_type='bin'):
    """
    Get TAD list in target ranges
    :param st: int, start position of target range
    :param ed: int, end position of target range
    :param df_tad_use_1: pandas dataframe, TAD list
    :param resolution: int, resolution of Hi-C contact map
    :param start_bin:  int, start bin of Hi-C contact map
    :param pos_type:  str, position type, 'bin' or 'cord'
    :return: TAD_list: list, TAD list in target ranges
    """
    if pos_type == 'cord':
        df_tad_use = copy.deepcopy(df_tad_use_1)
        df_tad_use['start'] = np.array(df_tad_use['start'] / resolution).astype(np.int32)
        df_tad_use['end'] = np.array(df_tad_use['end'] / resolution).astype(np.int32) - 1
    else:
        df_tad_use = copy.deepcopy(df_tad_use_1)
    TAD_list = []
    for i in range(len(df_tad_use)):
        start = df_tad_use['start'][i] - start_bin
        end = df_tad_use['end'][i] - start_bin
        if start >= st and end < ed:
            st_bin = start
            ed_bin = end - 1
            TAD_list.append((st_bin, ed_bin))
        elif start < st and (st < end <= ed):
            st_bin = start
            ed_bin = end - 1
            TAD_list.append((st_bin, ed_bin))
        elif (ed > start >= st) and end >= ed:
            st_bin = start
            ed_bin = end - 1
            TAD_list.append((st_bin, ed_bin))
    return TAD_list


## function to compare matrix

# Modofied from https://github.com/dejunlin/hicrep/blob/master/hicrep/utils.py
def HiC_rep_trimDiags(a: sp.coo_matrix, iDiagMax: int, bKeepMain: bool):
    """
    Remove diagonal elements whose diagonal index is >= iDiagMax or is == 0
    :param a: scipy.sparse.coo_matrix, Hi-C contact map
    :param iDiagMax: int, Diagonal offset cutoff
    :param bKeepMain:  bool, If true, keep the elements in the main diagonal; otherwise remove them
    :return: scipy.sparse.coo_matrix,  coo_matrix with the specified diagonals removed
    """
    gDist = np.abs(a.row - a.col)
    idx = np.where((gDist < iDiagMax) & (bKeepMain | (gDist != 0)))
    return sp.coo_matrix((a.data[idx], (a.row[idx], a.col[idx])),
                         shape=a.shape, dtype=a.dtype)

def upperDiagCsr(m: sp.coo_matrix, nDiags: int):
    """
    Convert an input sp.coo_matrix into a scipy.sparse.csr_matrix where each row in the
    output corresponds to one diagonal of the upper triangle of the input matrix.
    :param m: scipy.sparse.coo_matrix, input matrix
    :param nDiags: int, output diagonals with index in the range [1, nDiags)
    :return: scipy.sparse.csr_matrix, whose rows are the diagonals of the input
    """
    row = m.col - m.row
    idx = np.where((row > 0) & (row < nDiags))
    idxRowp1 = row[idx]
    # the diagonal index becomes the row index
    idxRow = idxRowp1 - 1
    # offset in the original diagonal becomes the column index
    idxCol = m.col[idx] - idxRowp1
    ans = sp.csr_matrix((m.data[idx], (idxRow, idxCol)),
                        shape=(nDiags - 1, m.shape[1]), dtype=m.dtype)
    ans.eliminate_zeros()
    return ans

def varVstran(n):
    """
    Calculate the variance of variance-stabilizing transformed
    (or `vstran()` in the original R implementation) data. The `vstran()` turns
    the input data into ranks, whose variance is only a function of the input
    size:
        ```
        var(1/n, 2/n, ..., n/n) = (1 - 1/(n^2))/12
        ```
    or with Bessel's correction:
        ```
        var(1/n, 2/n, ..., n/n, ddof=1) = (1 + 1.0/n)/12
        ```
    See section "Variance stabilized weights" in reference for more detail:
    https://genome.cshlp.org/content/early/2017/10/06/gr.220640.117
    Args:
        n (Union(int, np.ndarray)): size of the input data
    Returns: `Union(int, np.ndarray)` variance of the ranked input data with Bessel's
    correction
    """
    with suppress(ZeroDivisionError), np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n < 2, np.nan, (1 + 1.0 / n) / 12.0)

def HiC_rep_sccByDiag(m1: sp.coo_matrix, m2: sp.coo_matrix, nDiags: int):
    """
    Compute diagonal-wise hicrep SCC score for the two input matrices up to
    nDiags diagonals
    :param m1: scipy.sparse.coo_matrix, input contact matrix 1
    :param m2: scipy.sparse.coo_matrix, input contact matrix 2
    :param nDiags:  int, compute SCC scores for diagonals whose index is in the range of [1, nDiags)
    :return: float, hicrep SCC scores
    """
    # convert each diagonal to one row of a csr_matrix in order to compute
    # diagonal-wise correlation between m1 and m2
    m1D = upperDiagCsr(m1, nDiags)
    m2D = upperDiagCsr(m2, nDiags)
    nSamplesD = (m1D + m2D).getnnz(axis=1)
    rowSumM1D = m1D.sum(axis=1).A1
    rowSumM2D = m2D.sum(axis=1).A1
    # ignore zero-division warnings because the corresponding elements in the
    # output don't contribute to the SCC scores
    with np.errstate(divide='ignore', invalid='ignore'):
        cov = m1D.multiply(m2D).sum(axis=1).A1 - rowSumM1D * rowSumM2D / nSamplesD
        rhoD = cov / np.sqrt(
            (m1D.power(2).sum(axis=1).A1 - np.square(rowSumM1D) / nSamplesD ) *
            (m2D.power(2).sum(axis=1).A1 - np.square(rowSumM2D) / nSamplesD ))
        wsD = nSamplesD * varVstran(nSamplesD)
        # Convert NaN and Inf resulting from div by 0 to zeros.
        # posinf and neginf added to fix behavior seen in 4DN datasets
        # 4DNFIOQLTI9G and DNFIH7MQHOR at 5kb where inf would be reported
        # as an SCC score
        wsNan2Zero = np.nan_to_num(wsD, copy=True, posinf=0.0, neginf=0.0)
        rhoNan2Zero = np.nan_to_num(rhoD, copy=True, posinf=0.0, neginf=0.0)

    return rhoNan2Zero @ wsNan2Zero / wsNan2Zero.sum()


def get_HiC_rep_formal_for_two_matrix(mat_raw, mat_imputed_sym, diagcut = 40):
    """
    Get HiCRep score for two Hi-C contact maps
    :param mat_raw: numpy array, raw Hi-C contact map
    :param mat_imputed_sym: numpy array, imputed Hi-C contact map
    :param diagcut: int, the range to calculate HiCRep score, [1, diagcut)
    :return: hic_rep: float, HiCRep score
    """
    mat_sp1 = scipy.sparse.coo_matrix(mat_raw)
    mat_sp2 = scipy.sparse.coo_matrix(mat_imputed_sym)

    m1 = HiC_rep_trimDiags(mat_sp1, diagcut, False)
    m2 = HiC_rep_trimDiags(mat_sp2, diagcut, False)

    hic_rep = HiC_rep_sccByDiag(m1, m2, diagcut)
    return hic_rep


def get_HiC_rep_for_two_matrix(mat_raw, mat_imputed_sym, start_stratum = 1, end_stratum = 40):
    """
    Get HiCRep score for two Hi-C contact maps another implementation
    :param mat_raw: numpy array, raw Hi-C contact map
    :param mat_imputed_sym: numpy array, imputed Hi-C contact map
    :param start_stratum: int, start stratum to calculate HiCRep score
    :param end_stratum: int, end stratum to calculate HiCRep score
    :return: hic_rep: float, HiCRep score
    """
    raw_vec = []
    imputed_vec = []
    for stratum in range(start_stratum, end_stratum + 1):
        raw_vec_stratum = zscore(np.diag(mat_raw, k = stratum))
        imputed_vec_stratum = zscore(np.diag(mat_imputed_sym, k = stratum))
        raw_vec += list(raw_vec_stratum)
        imputed_vec += list(imputed_vec_stratum)
    hic_rep = np.dot(np.array(raw_vec), np.array(imputed_vec)) / len(raw_vec)
    if hic_rep >= 0.999:
        hic_rep = 1
    return hic_rep


def get_PCC_for_two_HiC_mat(mat_raw, mat_imputed_sym, K = 40):
    """
    Get Pearson correlation coefficient for two Hi-C contact maps
    :param mat_raw: numpy array, raw Hi-C contact map
    :param mat_imputed_sym: numpy array, imputed Hi-C contact map
    :param K: int, the diagonal offset to calculate PCC
    :return:  PCC: float, Pearson correlation coefficient
    """
    mat_raw_vec = []
    mat_imputed_vec = []
    for i in range(1, K):
        mat_raw_vec += list(np.diag(mat_raw, k = i))
        mat_imputed_vec += list(np.diag(mat_imputed_sym, k = i))
    PCC = scipy.stats.pearsonr(mat_raw_vec, mat_imputed_vec)[0]
    return PCC


def get_SSIM_for_two_HiC_mat(mat_raw_norm, mat_imputed_sym_norm, win_size, K = 40, norm = False):
    """
    Get SSIM for two Hi-C contact maps
    :param mat_raw_norm: numpy array, normalized raw Hi-C contact map
    :param mat_imputed_sym_norm: numpy array, normalized imputed Hi-C contact map
    :param win_size: int, window size to calculate SSIM
    :param K: int, the diagonal offset to calculate SSIM
    :param norm: bool, whether to normalize Hi-C contact map
    :return: ssim_all_mean: float, mean SSIM for all pixels
    """
    if norm == False:
        mat_raw_norm = mat_raw_norm / np.max(mat_raw_norm)
        mat_imputed_sym_norm = mat_imputed_sym_norm / np.max(mat_imputed_sym_norm)
    ssim_all_mean, ssim_full = ssim(mat_raw_norm, mat_imputed_sym_norm, win_size = win_size, full = True)
    v_1 = []
    for l in range(K):
        v_1 += list(np.diag(ssim_full, l))
    ssim_center_mean = np.mean(v_1)
    return ssim_all_mean, ssim_center_mean, ssim_full


def get_SSIM_for_two_HiC_mat_diag(mat_raw_norm, mat_imputed_sym_norm, win_size = 21, norm = False):
    """
    Get SSIM for two Hi-C contact maps
    :param mat_raw_norm: numpy array, normalized raw Hi-C contact map
    :param mat_imputed_sym_norm: numpy array, normalized imputed Hi-C contact map
    :param win_size: int, window size to calculate SSIM certer on pixel in diagonal
    :param norm: bool, whether to normalize Hi-C contact map
    :return: ssim_all_mean: float, mean SSIM for all pixels
    """
    if norm == False:
        mat_raw_norm = mat_raw_norm / np.max(mat_raw_norm)
        mat_imputed_sym_norm = mat_imputed_sym_norm / np.max(mat_imputed_sym_norm)
    ssim_all_mean, ssim_full = ssim(mat_raw_norm, mat_imputed_sym_norm, win_size = win_size, full = True)
    v_1 = list(np.diag(ssim_full, 0))
    ssim_center_mean = np.mean(v_1)
    return ssim_all_mean, ssim_center_mean, ssim_full


def get_PSNR_for_two_HiC_mat_diag(mat_raw_norm, mat_imputed_sym_norm, win_size = 20):
    """
    Get PSNR for two Hi-C contact maps
    :param mat_raw_norm: numpy array, normalized raw Hi-C contact map
    :param mat_imputed_sym_norm: numpy array, normalized imputed Hi-C contact map
    :param win_size: int, window size to calculate PSNR
    :return: psnr_list: list, PSNR for each pixel
    """
    psnr_list = []
    for i in range(len(mat_raw_norm)):
        if i <= win_size or i >= len(mat_raw_norm) - win_size - 1:
            continue
        mat_1 = mat_raw_norm[i - win_size : i + win_size + 1, i - win_size : i + win_size + 1]
        mat_2 = mat_imputed_sym_norm[i - win_size : i + win_size + 1, i - win_size : i + win_size + 1]
        mse = np.mean((mat_1 - mat_2) ** 2)
        if mse <= 0:
            continue
        psnr = 10*np.log10(np.max(mat_1)**2 / mse)
        psnr_list.append(psnr)
    df_psnr = pd.DataFrame(psnr_list)
    df_psnr = df_psnr.replace(float('inf'), np.nan)
    df_psnr = df_psnr.replace(float('-inf'), np.nan)
    psnr_list = list(df_psnr[0])
    psnr = np.nanmean(psnr_list)
    return psnr_list, psnr



def get_bin_tad_label(df_tad_mclust_fill, bin_num):
    lb_hold = 0
    bin_lb = np.zeros(bin_num) - 1
    for i in range(len(df_tad_mclust_fill)):
        domain = list(range(df_tad_mclust_fill['start'][i], df_tad_mclust_fill['end'][i] + 1))
        domain = sorted(domain)
        bin_lb[domain] = lb_hold
        lb_hold += 1
    return bin_lb

def get_tad_fill_gap(Chr, df_tad_record, bin_num):
    chr_l = []
    st_l = []
    ed_l = []
    type_l = []
    if df_tad_record['start'][0] >= 1:
        chr_l.append(Chr)
        st_l.append(0)
        ed_l.append(df_tad_record['start'][0]-1)
        type_l.append('gap')
    for i in range(len(df_tad_record) - 1):
        st = df_tad_record['start'][i]
        ed = df_tad_record['end'][i]
        st_n = df_tad_record['start'][i+1]
        chr_l.append(Chr)
        st_l.append(st)
        ed_l.append(ed)
        type_l.append('domain')
        if st_n > ed + 1:
            chr_l.append(Chr)
            st_l.append(ed+1)
            ed_l.append(st_n-1)
            type_l.append('gap')
    chr_l.append(Chr)
    st = df_tad_record['start'][len(df_tad_record) - 1]
    ed = df_tad_record['end'][len(df_tad_record) - 1]
    st_l.append(st)
    ed_l.append(ed)
    type_l.append('domain')
    if df_tad_record['end'][len(df_tad_record) - 1] < bin_num - 1:
        chr_l.append(Chr)
        st_l.append(ed+1)
        ed_l.append(bin_num - 1)
        type_l.append('gap')
    df_tad_gap = pd.DataFrame(columns = ['chr', 'start', 'end', 'type'])
    df_tad_gap['chr'] = chr_l
    df_tad_gap['start'] = st_l
    df_tad_gap['end'] = ed_l
    df_tad_gap['type'] = type_l
    return df_tad_gap






