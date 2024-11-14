import os
import numpy as np
import pandas as pd
import torch
import scipy
import warnings
warnings.filterwarnings('ignore')
import TADGATE
from TADGATE import Call_TADs as CT
from TADGATE import TADGATE_main
from TADGATE import TADGATE_utils as TL
from TADGATE import Plot_Function as PF
import argparse
import sys
#sys.path.append('/home/dcdang/pycharm_workspace/TADGATE/TADGATE_script_new')
#import TADGATE_utils as TL
#import Plot_Function as PF
#import Call_TADs as CT
#import TADGATE_main


# Function to read parameters from parameters.txt
def read_parameters(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.split('=')  # Split key-value pair
                params[key.strip()] = value.strip()
    return params

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

# Function to load the contact matrix data based on dtype
def load_data(input_path, dtype, chr_num_l, chr_size, resolution):
    if dtype == 'dense':
        # For dense matrices, each file is named as chr_*_dense_matrix.bed
        hic_data = {}
        for Chr in chr_num_l:
            file_path = os.path.join(input_path, f"chr{Chr}_dense_matrix.bed")
            df_mat_dense = pd.read_csv(file_path, sep='\t', header=None)  # Adjust based on actual data format
            hic_data[Chr] = df_mat_dense.values
        return hic_data
    elif dtype == 'sparse':
        hic_data = {}
        for Chr in chr_num_l:
            file_path = os.path.join(input_path, f"chr{Chr}_sparse_matrix.bed")
            df_sparse = pd.read_csv(file_path, sep='\t', header=None)
            df_sparse.columns = ['bin1', 'bin2', 'value']
            if (chr_size['chr' + Chr] / resolution) % resolution != 0:
                bin_num = int(chr_size['chr' + Chr] / resolution) + 1
            else:
                bin_num = int(chr_size['chr' + Chr] / resolution)
            if np.max(df_sparse['bin1']) * resolution > chr_size['chr' + Chr]:
                mat_dense_chr = get_dense_hic_mat(df_sparse, length=bin_num, resolution=resolution)
            else:
                mat_dense_chr = get_dense_hic_mat(df_sparse, length=bin_num)
            hic_data['chr' + Chr] = mat_dense_chr
        return hic_data
    elif dtype == 'hic':
        import hicstraw
        hic_file = hicstraw.HiCFile(input_path)
        hic_data = {}
        for chrom in hic_file.getChromosomes():
            # print(chrom.name, chrom.length)
            if chrom.name not in chr_num_l:
                continue
            result = hicstraw.straw('observed', 'NONE', input_path, chrom.name, chrom.name, 'BP', resolution)
            bin_1_l = []
            bin_2_l = []
            count_l = []
            for i in range(len(result)):
                bin_1_l.append(result[i].binX)
                bin_2_l.append(result[i].binY)
                count_l.append(result[i].counts)
            df_mat_sparse = pd.DataFrame({
                'bin1': bin_1_l,
                'bin2': bin_2_l,
                'value': count_l
            })
            if (chrom.length / resolution) % resolution != 0:
                bin_num = int(chrom.length / resolution) + 1
            else:
                bin_num = int(chrom.length / resolution)
            if np.max(df_mat_sparse['bin1']) * resolution > chr_size['chr' + chrom.name]:
                df_mat_dense_chr = get_dense_hic_mat(df_mat_sparse, length=bin_num, resolution=resolution)
            else:
                df_mat_dense_chr = get_dense_hic_mat(df_mat_sparse, length=bin_num)
            hic_data['chr' + chrom.name] = df_mat_dense_chr
        return hic_data
    elif dtype == 'mcool':
        import cooler
        hic_data = {}
        for Chr in chr_num_l:
            cool_file = cooler.Cooler(input_path + '::/resolutions/' + str(resolution))
            cool_mat = cool_file.matrix(balance=False).fetch(Chr)
            hic_data['chr' + Chr] = cool_mat
        return hic_data
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


# Function to save results in output_path
def save_results(output_path, TADGATE_res, TADGATE_tads, save_att_mat=False):
    for Chr in TADGATE_res.keys():
        chr_folder = os.path.join(output_path, Chr)
        os.makedirs(chr_folder, exist_ok=True)

        # Save mat_imputed_sym
        mat_imputed_sym = TADGATE_res[Chr]['result'][0]['mat_imputed_sym']
        mat_imputed_sym_path = os.path.join(chr_folder, Chr + '_mat_imputed.bed')
        pd.DataFrame(mat_imputed_sym).to_csv(mat_imputed_sym_path, sep='\t', header=None, index=None)

        # Save attention_mat
        if save_att_mat:
            attention_mat = TADGATE_res[Chr]['result'][0]['attention_mat']
            attention_mat_path = os.path.join(chr_folder, Chr + '_attention_mat.bed')
            pd.DataFrame(attention_mat).to_csv(attention_mat_path, sep='\t', header=None, index=None)

        # Save TADs
        df_tads = TADGATE_tads[Chr]['TADs'][0]
        tads_file_path = os.path.join(chr_folder, Chr + '_TADs_result.bed')
        df_tads[['chr', 'start_pos', 'end_pos', 'start', 'end', 'type']].to_csv(tads_file_path,
                                                                                sep='\t', header=True,
                                                                                index=None)

        # Save other result vectors (e.g., bin_vote_bd_score, CI_original, etc.)
        vectors = ['CI_original', 'pvalue_original',
                   'CI_imputed', 'pvalue_imputed', 'att_profile', 'bin_vote_bd_score']
        df_indacor = pd.DataFrame(columns=vectors)
        ind_file = os.path.join(chr_folder, Chr + '_indactor_all.bed')
        for vector in vectors:
            if vector in TADGATE_tads[Chr]:
                df_indacor[vector] = TADGATE_tads[Chr][vector][0]
        df_indacor.to_csv(ind_file, sep='\t', header=True, index=None)


# Main function to execute the TADGATE workflow
def main():
    # Set up argument parser to accept parameters.txt as an argument
    parser = argparse.ArgumentParser(description="Run TADGATE workflow")
    parser.add_argument('parameter_file', help="Path to the parameters.txt file")
    args = parser.parse_args()

    # Read parameters from the parameters file
    params = read_parameters(args.parameter_file)

    # Load contact matrix data based on dtype
    dtype = params['dtype']
    input_path = params['input_path']
    resolution = int(params['resolution'])
    chr_num_l = list(params['chr_num'].split(','))

    # Load chromosome size information
    chr_size_file = params['chr_size_file']
    df_chr_size = pd.read_csv(chr_size_file, sep="\t", header=None, names=["chr", "size"])
    chr_size = {}
    for i in range(len(df_chr_size)):
        chr_size[df_chr_size['chr'][i]] = df_chr_size['size'][i]
    hic_mat = load_data(input_path, dtype, chr_num_l, chr_size, resolution)

    # Extract other parameters for graph attention and TAD detection
    device = params['device']
    device_gpu = torch.device(device if torch.cuda.is_available() else 'cpu')
    graph_radius = int(params['graph_radius'])
    split_size = params['split_size']
    if split_size != 'all':
        split_size = int(split_size)
    layer_node1 = int(params['layer_node1'])
    layer_node2 = int(params['layer_node2'])
    embed_attention = params['embed_attention']
    if embed_attention == 'False':
        embed_attention = False
    elif embed_attention == 'True':
        embed_attention = True
    lr = float(params['lr'])
    weight_decay = float(params['weight_decay'])
    num_epoch = int(params['num_epoch'])
    weight_use = params['weight_use']
    weight_rate = float(params['weight_rate'])
    weight_range = int(params['weight_range'])
    bd_weight_list = list(map(float, params['bd_weight_list'].strip("[]").split(",")))
    impute_func = params['impute_func']
    if impute_func == 'False':
        impute_func = False
    elif impute_func == 'True':
        impute_func = True
    impute_range = int(params['impute_range'])

    # Parameters for TAD detection
    cluster_method = params['cluster_method']
    window_range = int(params['window_range'])
    wd_ci = int(params['wd_ci'])
    wd_p = int(params['wd_p'])
    dist = int(params['dist'])
    pvalue_cut = float(params['pvalue_cut'])
    exp_length = int(params['exp_length'])
    length_cut = int(params['length_cut'])
    filter_method = params['filter_method']
    output_path = params['output_path']
    save_att_mat = params['save_att_mat']
    if save_att_mat == 'False':
        save_att_mat = False
    elif save_att_mat == 'True':
        save_att_mat = True

    # Run TADGATE
    TADGATE_res = TADGATE_main.TADGATE_for_embedding(hic_mat, chr_size, resolution, graph_radius, split_size,
                                                     device=device_gpu, layer_node1=layer_node1,
                                                     layer_node2=layer_node2,
                                                     lr=lr, weight_decay=weight_decay, num_epoch=num_epoch,
                                                     embed_attention=embed_attention,
                                                     weight_use=weight_use, weight_range=weight_range,
                                                     weight_rate=weight_rate, impute_func=impute_func,
                                                     impute_range=impute_range, target_chr_l=[])

    TADGATE_tads = CT.TADGATE_call_TADs(TADGATE_res, chr_size, resolution, bd_weight_list, cluster_method,
                                        window_range=window_range, wd_ci=wd_ci, wd_p=wd_p, dist=dist,
                                        pvalue_cut=pvalue_cut,
                                        exp_length=exp_length,
                                        length_cut=length_cut, contact_fold_cut=2, zero_ratio_cut=0.3,
                                        target_chr_l=[])

    # Save results
    save_results(output_path, TADGATE_res, TADGATE_tads, save_att_mat=save_att_mat)


if __name__ == "__main__":
    main()
