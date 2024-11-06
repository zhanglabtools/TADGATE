
import scipy
import copy
import time
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import mannwhitneyu


def z_scale(y):
    """
    Z-score normalization
    :param y: input data
    :return: z-score normalized data
    """
    x = y.copy()
    x = x -  np.mean(x)
    x = x / np.std(x, axis=0, ddof=1)
    return x

def Get_Diamond_Matrix(data, i, size):
    """
    Get the diamond matrix
    :param data: numpy array, input data
    :param i: int, index of the bin
    :param size: int, size of the diamond matrix
    :return: list, values of diamond matrix
    """
    n_bins = data.shape[0]
    # new_mat = np.ones_like(data)*np.NaN
    new_mat = np.ones(shape=(size, size)) * np.NaN
    for k in range(1, size + 1):
        if i - (k - 1) >= 1 and i < n_bins:
            lower = min(i + 1, n_bins)
            upper = min(i + size, n_bins)
            new_mat[size - (k - 1) - 1, 0:(upper - lower + 1)] = data[i - (k - 1) - 1, lower - 1:upper]
    new_mat = new_mat[np.logical_not(np.isnan(new_mat))]
    return ((new_mat.transpose()).flatten()).tolist()

def Get_Upstream_Triangle(data, i, size):
    """
    Get the upstream triangle
    :param data: numpy array, input data
    :param i: int, index of the bin
    :param size: int, size of the diamond matrix
    :return: list, values of upstream triangle
    """
    lower = max(1, i - size)
    tmp_mat = data[lower - 1:i, lower - 1:i]
    triag = (np.triu(tmp_mat, k=1).flatten())
    return triag[triag != 0].tolist()

def Get_Downstream_Triangle(data, i, size):
    """
    Get the downstream triangle
    :param data: numpy array, input data
    :param i: int, index of the bin
    :param size: int, size of the diamond matrix
    :return: list, values of downstream triangle
    """
    n_bins = data.shape[0]
    if i == n_bins:
        return np.NAN
    upperbound = min(i + size, n_bins)
    tmp_mat = data[i:upperbound, i:upperbound]
    triag = (np.triu(tmp_mat, k=1).flatten())
    return triag[triag != 0].tolist()

def Get_Pvalue(data, size):
    """
    Get the contrast p-value
    :param data: numpy array, input data
    :param size: int, size of the diamond matrix
    :return: numpy array, contrast p-value of all bins
    """
    n_bins = data.shape[0]
    pvalue = np.ones(n_bins - 1)
    for i in range(1, n_bins):
        dia = Get_Diamond_Matrix(data, i, size=size)
        ups = Get_Upstream_Triangle(data, i, size=size)
        downs = Get_Downstream_Triangle(data, i, size=size)
        wil_test = mannwhitneyu(x=dia, y=ups + downs, use_continuity=True, alternative='less')
        pvalue[i - 1] = wil_test.pvalue
    pvalue[np.isnan(pvalue)] = 1
    return pvalue

def Get_CI_value(data, size):
    """
    Get the contrast index
    :param data: numpy array, input data
    :param size: int, size of the diamond matrix 
    :return: numpy array, contrast index of all bins 
    """
    n_bins = data.shape[0]
    CI_value = np.ones(n_bins - 1)
    for i in range(1, n_bins):
        dia = Get_Diamond_Matrix(data, i, size=size)
        ups = Get_Upstream_Triangle(data, i, size=size)
        downs = Get_Downstream_Triangle(data, i, size=size)

        if len(dia) == 0 or np.mean(dia) == 0:
            CI_value[i - 1] = 0
        elif len(ups) == 0 or len(downs) == 0:
            CI_value[i - 1] = 0
        else:
            CI_value[i - 1] = np.mean(ups + downs) / np.mean(dia)
    CI_value[np.isnan(CI_value)] = 0
    return CI_value

def Get_IS_value(data, size):
    """
    Get the insulation score
    :param data: numpy array, input data
    :param size: int, size of the diamond matrix
    :return: numpy array, insulation score of all bins
    """
    n_bins = data.shape[0]
    IS_value = np.ones(n_bins - 1)
    for i in range(1, n_bins):
        dia = Get_Diamond_Matrix(data, i, size=size)
        if len(dia) == 0:
            IS_value[i - 1] = 0
        else:
            IS_value[i - 1] = np.mean(dia)
    IS_value[np.isnan(IS_value)] = 0
    IS_value_l = list(IS_value)
    IS_value_l.append(0)
    IS_value = np.array(IS_value_l)
    return IS_value

def get_norm_hic_matrix(mat_hic, wd_p):
    """
    Get the normalized Hi-C matrix
    :param mat_hic: numpy array, input Hi-C matrix
    :param wd_p: int, size of the diamond matrix
    :return: numpy array, normalized Hi-C matrix
    """
    mat_norm = np.zeros([mat_hic.shape[0], mat_hic.shape[0]])
    for k in range(1, 4 * wd_p):
        mat_diag = np.diag(mat_hic, k=k)
        scale_values = z_scale(mat_diag)
        mat_norm += np.diag(scale_values, k=k)
    mat_norm += mat_norm.T
    mat_diag = np.diag(mat_hic, k=0)
    scale_values = z_scale(mat_diag)
    mat_norm += np.diag(scale_values, k = 0)
    df_mat_norm = pd.DataFrame(mat_norm)
    df_mat_norm.fillna(0, inplace=True)
    mat_norm = df_mat_norm.values
    return mat_norm


def get_CI_value_and_pvalue(mat_hic, wd_ci = 5, wd_p = -1):
    """
    Get the contrast index and contrast p-value
    :param mat_hic: numpy array, input Hi-C matrix
    :param wd_ci: int, size of the diamond matrix for contrast index
    :param wd_p: int, size of the diamond matrix for contrast p-value
    :return: numpy array, contrast index of all bins
             numpy array, contrast p-value of all bins
    """
    if wd_p == -1:
        wd_p = wd_ci
    mat_norm = get_norm_hic_matrix(mat_hic, wd_p)
    pvalue = Get_Pvalue(data=mat_norm, size=wd_p)
    pvalue_l = list(pvalue)
    pvalue_l.append(1)
    pvalue = np.array(pvalue_l)
    CI_value = Get_CI_value(mat_hic, wd_ci)
    CI_value_l = list(CI_value)
    CI_value_l.append(0)
    CI_value = np.array(CI_value_l)
    return CI_value, pvalue


def smooth_mat_att(mat_att, sm_size=1):
    """
    Get smooth attention sum profile
    :param mat_att: numpy array, input attention matrix
    :param sm_size: int, size of the smoothing window
    :return: numpy array, smoothed attention sum profile
    """
    att_pro = np.sum(mat_att, axis=0)
    att_smooth = []
    for i in range(len(att_pro)):
        if i < sm_size:
            att_smooth.append(att_pro[i])
        elif i > len(att_pro) - sm_size - 1:
            att_smooth.append(att_pro[i])
        else:
            att_ave = []
            for ind in range(sm_size + 1):
                if ind == 0:
                    att_ave.append(att_pro[i])
                else:
                    att_ave.append(att_pro[i - ind])
                    att_ave.append(att_pro[i + ind])
            if len(att_ave) > 0:
                att_smooth.append(np.mean(att_ave))
            else:
                att_smooth.append(0)
    return att_smooth


def check_pvalue_valley(pvalue_ori_part, bd, dist):
    """
    Check the p-value valley
    :param pvalue_ori_part: numpy array, input contrast p-value array
    :param bd:  int, bin index of candidate boundary
    :param dist: int, distance cut-off between valleys and boundaries
    :return: str, 'True' or 'False', 'True' for bd near p-value valley, 'False' for not
    """
    if dist == None:
        pvalue_valley = scipy.signal.find_peaks(-pvalue_ori_part)[0]
    else:
        pvalue_valley = scipy.signal.find_peaks(-pvalue_ori_part, distance = dist)[0]
    pv_dist = np.abs(pvalue_valley - bd)
    v_pos = pvalue_valley[np.argmin(pv_dist)]
    if pv_dist[np.argmin(pv_dist)] <= 1 and pvalue_ori_part[v_pos] < 0.5:
        return 'True'
    else:
        return 'False'

# Mclust for bins
def get_label_of_mclust(lb_pro):
    """
    Get the label of each bin based on the mclust result
    :param lb_pro: numpy array, mclust result for each bin in input window
    :return: list, label of each bin
    """
    bin_label_l = []
    for i in range(len(lb_pro)):
        proba = lb_pro[i]
        pro_max = np.argmax(proba)
        bin_label_l.append(pro_max)
    return bin_label_l


def label_adjust(bin_label_l, dist_tro=5):
    """
    Adjust the label of each bin from Mclust, to make the label more continuous
    :param bin_label_l: list, label of each bin direct from Mclust
    :param dist_tro: int, tolerance distance for bin label change
    :return:
    """
    bin_label_l_new = [bin_label_l[0]]
    hold = bin_label_l[0]
    count = 0
    for i in range(1, len(bin_label_l)):
        if bin_label_l[i] == hold:
            bin_label_l_new.append(bin_label_l[i])
        else:
            # current label of i is not hold and is not the same as the next bin
            if (i + 1) < len(bin_label_l) and bin_label_l[i + 1] != bin_label_l[i]:
                bin_label_l_new.append(hold)
                count += 1
            # if hold still exist in downstream bins, keep the hold
            elif (i + dist_tro) < len(bin_label_l) and bin_label_l[i:i + dist_tro].count(hold) >= 2:
                bin_label_l_new.append(hold)
                count += 1
            # if hold never exist in downstream bins
            else:
                bin_label_l_new.append(bin_label_l[i])
                hold = bin_label_l[i]
    if bin_label_l_new[0] != bin_label_l_new[1]:
        bin_label_l_new[0] = bin_label_l_new[1]
    # reindex the bin label to 0,1,2...
    bin_label_l_new2 = [0]
    lb_first = bin_label_l_new[0]
    lb_hold = 0
    for i in range(1, len(bin_label_l_new)):
        if bin_label_l_new[i] == lb_first:
            bin_label_l_new2.append(lb_hold)
        else:
            lb_hold += 1
            bin_label_l_new2.append(lb_hold)
            lb_first = bin_label_l_new[i]
    return bin_label_l_new2, count


def get_domain_label_from_boundary(CI_att_vote_value_part, bd_peak):
    """
    Get the domain label based on the boundary peak, bin between two bd_peaks are in the same domain
    :param CI_att_vote_value_part: numpy array, object used for get zero-array with the same length of target window
    :param bd_peak: list, boundary peak index in target window (0, window length)
    :return: numpy array, domain label of each bin
    """
    label_l = np.zeros(len(CI_att_vote_value_part))
    st = 0
    count = 0
    for i in range(len(bd_peak)):
        ed = bd_peak[i]
        label_l[st:ed] = count
        st = ed
        count += 1
    label_l[st:] = count
    return label_l


def Clustering_of_bins_for_bd(window, mat_rep, mat_hic, CI_att_vote_value, cluster_method, bd_weight_list,
                              pvalue_ori, pvalue_imp, pvalue_cut, exp_num, dist,
                              cl_add_score = 1, expand_check = 1, filter_method = 'strict'):
    """
    Add clustering boundary score to score profile based on clustering method of Mclust or K-means
    :param window: int, size of the window, we cut all bins into several windows to perform clustering
    :param mat_rep: numpy array, the embedding features of all bins, each row is the feature of one bin
    :param mat_hic: numpy array, the Hi-C matrix of the target chromosome
    :param CI_att_vote_value: numpy array, the boundary score profile after boundary voting and adding attention valleys
    :param cluster_method: str, 'Mclust', 'K-means' or 'None', the clustering method used for bin clustering
    :param bd_weight_list: list, the weight of each boundary peak in the boundary score profile
    :param pvalue_ori: numpy array, the contrast p-values for each bin in the original Hi-C contact map
    :param pvalue_imp: numpy array, the contrast p-values for each bin in the imputed Hi-C contact map
    :param pvalue_cut: float, the p-value cut-off for boundary filtering
    :param exp_num: int, the expected number of domains in the target window
    :param dist: int, the distance range used to call boundary peak
    :param cl_add_score: float, the score added to the boundary score profile for clustering boundary
    :param expand_check: int, the expand range used to check whether the boundary peak can pass p-value cut-off
    :param filter_method: str, 'strict' or 'relax', decide whether add p-value valley below 0.5 to candidate boundary
    :return: bin_label_clustering: list, the clustering label of each bin
             bd_vote_final: numpy array, the final boundary score profile after adding clustering boundary score
             bin_tad_label_new: numpy array, the final domain label of each bin
    """
    bin_tad_label = []
    st_pos = 0
    num = int(len(mat_rep) / window)
    process = 0
    bd_vote_final = copy.deepcopy(CI_att_vote_value)
    bin_label_clustering = list(np.zeros(len(CI_att_vote_value)) - 1)
    for i in range(2 * num):
        if int(len(bin_tad_label) / (int(len(mat_rep) / 5))) > process:
            print('20% bins done!')
            process += 1
        if (st_pos + 2 * window) >= len(mat_rep):
            mat_rep_part = mat_rep[st_pos:, :]
            mat_hic_part = mat_hic[st_pos:, st_pos:]
            pvalue_ori_part = pvalue_ori[st_pos:]
            pvalue_imp_part = pvalue_imp[st_pos:]
            CI_att_vote_value_part = copy.deepcopy(CI_att_vote_value[st_pos:])
            # att_prof_part = att_profile_smooth[st_pos:]
            # att_prof_part = np.array(att_prof_part)
        else:
            mat_rep_part = mat_rep[st_pos:st_pos + window, :]
            mat_hic_part = mat_hic[st_pos:st_pos + window, st_pos:st_pos + window]
            pvalue_ori_part = pvalue_ori[st_pos:st_pos + window]
            pvalue_imp_part = pvalue_imp[st_pos:st_pos + window]
            CI_att_vote_value_part = copy.deepcopy(CI_att_vote_value[st_pos:st_pos + window])
            # att_prof_part = att_profile_smooth[st_pos:st_pos + window]
            # att_prof_part = np.array(att_prof_part)
        zero_ratio = np.sum(np.diag(mat_hic_part) == 0) / len(mat_hic_part)
        # Much bad row, no need clustering
        if zero_ratio >= 0.8:
            bin_label_l_modify = [0 for z in range(len(mat_hic_part))]
        else:
            # No cluster method or zero clustering cluster score, no need clustering
            if cluster_method == 'None' or cl_add_score == 0:
                pass
            # choose one method to cluster
            else:
                if cluster_method == 'Mclust' and cl_add_score != 0:
                    from rpy2.robjects import r
                    import rpy2.robjects as robjects
                    import rpy2.robjects.numpy2ri
                    rpy2.robjects.numpy2ri.activate()
                    robjects.r.library('mclust')
                    mclust = robjects.r['Mclust']
                    if zero_ratio >= 0.5:
                        down_b = 2
                        up_b = exp_num
                    else:
                        down_b = np.max([int(exp_num / 2), 2])
                        up_b = exp_num + int(exp_num / 2)
                    res = mclust(mat_rep_part, G=list(range(down_b, up_b)))
                    import rpy2.rinterface
                    if isinstance(res, rpy2.rinterface.NULLType):
                        bin_label_l_new = np.zeros(len(mat_rep_part))
                    else:
                        lb_pro = np.array(res[13])
                        bin_label_l = get_label_of_mclust(lb_pro)
                        bin_label_l_new, adj_count = label_adjust(bin_label_l, dist_tro=5)
                elif cluster_method == 'K-means' and cl_add_score != 0:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=exp_num, random_state=0).fit(mat_rep_part)
                    bin_label_l = list(kmeans.labels_)
                    bin_label_l_new, adj_count = label_adjust(bin_label_l, dist_tro=5)
                if len(np.unique(bin_label_l_new)) > 1:
                    bin_label_l_new_str = []
                    for x in bin_label_l_new:
                        bin_label_l_new_str.append(str(i) + '_' + str(x))

                    if (st_pos + 2 * window) < len(mat_rep):
                        bin_label_clustering[st_pos:st_pos + window] = bin_label_l_new_str
                    else:
                        bin_label_clustering[st_pos:] = bin_label_l_new_str

                    lb_first = bin_label_l_new[0]
                    bd_l = []
                    for x in range(len(bin_label_l_new)):
                        if bin_label_l_new[x] != lb_first:
                            bd_l.append(x)
                            lb_first = bin_label_l_new[x]
                else:
                    bd_l = []
                ## Add clustering boundary score to profile
                for x in bd_l:
                    CI_att_vote_value_part[x] += cl_add_score

                if (st_pos + 2 * window) < len(mat_rep):
                    bd_vote_final[st_pos:st_pos + window] = CI_att_vote_value_part
                else:
                    bd_vote_final[st_pos:] = CI_att_vote_value_part

            from scipy.ndimage import convolve1d
            CI_att_vote_value_part = CI_att_vote_value_part.astype(float)
            kernel = np.array([0.0005, 1, 0.0005])
            CI_att_vote_value_part_new = convolve1d(CI_att_vote_value_part, weights=kernel, mode='constant', cval=0)

            if dist == None:
                vote_peak = signal.find_peaks(CI_att_vote_value_part_new)[0]
            else:
                vote_peak = signal.find_peaks(CI_att_vote_value_part_new, distance=dist)[0]
            bd_fillter = []
            for bd in vote_peak:
                if pvalue_ori_part[bd] <= pvalue_cut or pvalue_imp_part[bd] <= pvalue_cut:
                    bd_fillter.append(bd)
                else:
                    if bd > expand_check and bd < len(mat_rep_part) - expand_check - 1:
                        pos_r = list(range(bd - expand_check, bd + expand_check + 1))
                        if np.min(pvalue_ori_part[pos_r]) <= pvalue_cut:
                            pos_bd = np.argmin(pvalue_ori_part[pos_r])
                            bd_use = pos_r[pos_bd]
                        elif np.min(pvalue_imp_part[pos_r]) <= pvalue_cut:
                            pos_bd = np.argmin(pvalue_imp_part[pos_r])
                            bd_use = pos_r[pos_bd]
                        else:
                            bd_use = -1
                        if len(bd_fillter) == 0 and bd_use != -1:
                            bd_fillter.append(bd_use)
                        elif len(bd_fillter) > 0 and bd_use != -1:
                            bd_last = bd_fillter[-1]
                            if bd_use > bd_last:
                                bd_fillter.append(bd_use)
                    # allow add pvalue valley < 0.5
                    if filter_method == 'relax' or filter_method == 'relax2':
                        # at least three source suggest bd nearby as boundary
                        if np.sum(CI_att_vote_value_part[bd-1: bd+2]) >= (np.sum(bd_weight_list) - np.min(bd_weight_list)):
                            v_ori = check_pvalue_valley(pvalue_ori_part, bd, dist)
                            v_imp = check_pvalue_valley(pvalue_imp_part, bd, dist)
                            if v_ori == 'True' or v_imp == 'True':
                                bd_fillter.append(bd)
                        if filter_method == 'relax2':
                            if np.sum(CI_att_vote_value_part[bd-1: bd+2]) == np.sum(bd_weight_list):
                                bd_fillter.append(bd)

            bin_label_l_modify = get_domain_label_from_boundary(CI_att_vote_value_part, bd_fillter)

        # For each window we only keep start pos to the upstream boundary of the last domain
        if len(bin_label_l_modify) == window:
            if len(np.unique(bin_label_l_modify)) == 1:
                bin_label_l_modify_use = bin_label_l_modify
            else:
                lb_last = bin_label_l_modify[-1]
                lb_last_index = np.min(np.where(np.array(bin_label_l_modify) == lb_last)[0])
                bin_label_l_modify_use = bin_label_l_modify[:lb_last_index]
            bin_tad_label += [str(i) + '_' + str(int(x)) for x in bin_label_l_modify_use]
            st_pos += len(bin_label_l_modify_use)
        else:
            bin_tad_label += [str(i) + '_' + str(int(x)) for x in bin_label_l_modify]
        if len(bin_tad_label) == len(mat_rep):
            break
            # print(i, len(bin_label_l_modify_use), st_pos, len(bin_tad_label))
    bin_tad_label_new = np.zeros(len(bin_tad_label)) - 1
    record = []
    count = 0
    for lb in bin_tad_label:
        if lb not in record:
            bin_tad_label_new[np.array(bin_tad_label) == lb] = count
            record.append(lb)
            count += 1
    lb_first = bin_tad_label_new[0]
    up_dist = np.sum(bin_tad_label_new == lb_first)
    for i in range(1, len(bin_tad_label_new)):
        lb = bin_tad_label_new[i]
        loop_up = False
        loop_down = False
        if lb != lb_first:
            down_dist = np.sum(bin_tad_label_new == lb)
            if up_dist <= 2 or down_dist <= 2:
                lb_first = lb
                up_dist = np.sum(bin_tad_label_new == lb_first)
                continue
            # loop judge
            # form loop upstream, modify current bin label to upstream label
            center_mean = mat_hic[i - up_dist, i]
            hline_mean = np.mean(mat_hic[i - up_dist, i - up_dist + 1:i - 1])
            vline_mean = np.mean(mat_hic[i - up_dist + 1:i - 1, i])
            if center_mean > (hline_mean + vline_mean) / 2:
                bin_tad_label_new[i] = lb_first
                loop_up = True
            # form loop downstream, current bin label is different from that of upstream bins, no need modify
            center_mean = mat_hic[i, i + down_dist - 1]
            hline_mean = np.mean(mat_hic[i, i + 1: i + down_dist - 1])
            vline_mean = np.mean(mat_hic[i + 1:i + down_dist - 1, i + down_dist - 1])
            if center_mean > (hline_mean + vline_mean) / 2:
                loop_down = True

            # no loop form, modify current bin label accoding to the mean contact between upstream and downstream bins
            if loop_up == False and loop_down == False:
                #### Here update mat_hic to mat_att and try
                # up_att = np.mean(mat_hic[i, np.max([0,i - up_dist]) : i])
                # down_att = np.mean(mat_hic[i, i+1: np.min([i+down_dist, len(mat_hic)])])
                dist_mid = np.min([up_dist, down_dist])
                up_att = np.mean(mat_hic[i, np.max([0, i - dist_mid]): i])
                down_att = np.mean(mat_hic[i, i + 1: np.min([i + dist_mid, len(mat_hic)])])

                # up_att = np.mean(mat_att[i, np.max([0,i - np.min([up_dist, 2])]) : i])
                # down_att = np.mean(mat_att[i, i+1: np.min([i+np.min([down_dist, 2]), len(mat_hic)])])

                if down_att > up_att:
                    pass
                else:
                    bin_tad_label_new[i] = lb_first
            lb_first = lb
            up_dist = np.sum(bin_tad_label_new == lb_first)
    return bin_label_clustering, bd_vote_final, bin_tad_label_new



def get_TAD_label_full_based_on_vectors(mat_hic, bd_vote_combine, pvalue_ori_combine, pvalue_imp_combine, dist, pvalue_cut,
                                        bd_weight_list, filter_method = 'strict'):
    if dist == None:
        vote_peak = signal.find_peaks(bd_vote_combine)[0]
    else:
        vote_peak = signal.find_peaks(bd_vote_combine, distance=dist)[0]
    bd_fillter = []
    for bd in vote_peak:
        if pvalue_ori_combine[bd] <= pvalue_cut or pvalue_imp_combine[bd] <= pvalue_cut:
            bd_fillter.append(bd)
        else:
            if bd >= 1 and bd <= len(bd_vote_combine) - 1:
                if pvalue_ori_combine[bd - 1] <= pvalue_cut or pvalue_imp_combine[bd - 1] <= pvalue_cut:
                    bd_fillter.append(bd - 1)
                    continue
                elif pvalue_ori_combine[bd + 1] <= pvalue_cut or pvalue_imp_combine[bd + 1] <= pvalue_cut:
                    bd_fillter.append(bd + 1)
                    continue
            # allow add pvalue valley < 0.5
            if filter_method == 'relax':
                # at least three source suggest bd nearby as boundary
                if np.sum(bd_vote_combine[bd - 1: bd + 2]) >= (np.sum(bd_weight_list) - np.min(bd_weight_list)):
                    v_ori = check_pvalue_valley(pvalue_ori_combine, bd, dist)
                    v_imp = check_pvalue_valley(pvalue_imp_combine, bd, dist)
                    if v_ori == 'True' or v_imp == 'True':
                        bd_fillter.append(bd)

    bin_label_l_modify = get_domain_label_from_boundary(bd_vote_combine, bd_fillter)
    bin_tad_label_new = bin_label_l_modify
    lb_first = bin_tad_label_new[0]
    up_dist = np.sum(bin_tad_label_new == lb_first)
    for i in range(1, len(bin_tad_label_new)):
        lb = bin_tad_label_new[i]
        loop_up = False
        loop_down = False
        if lb != lb_first:
            down_dist = np.sum(bin_tad_label_new == lb)
            if up_dist <= 2 or down_dist <= 2:
                lb_first = lb
                up_dist = np.sum(bin_tad_label_new == lb_first)
                continue
            # loop judge
            # form loop upstream, modify current bin label to upstream label
            center_mean = mat_hic[i - up_dist, i]
            hline_mean = np.mean(mat_hic[i - up_dist, i - up_dist + 1:i - 1])
            vline_mean = np.mean(mat_hic[i - up_dist + 1:i - 1, i])
            if center_mean > (hline_mean + vline_mean) / 2:
                bin_tad_label_new[i] = lb_first
                loop_up = True
            # form loop downstream, current bin label is different from that of upstream bins, no need modify
            center_mean = mat_hic[i, i + down_dist - 1]
            hline_mean = np.mean(mat_hic[i, i + 1: i + down_dist - 1])
            vline_mean = np.mean(mat_hic[i + 1:i + down_dist - 1, i + down_dist - 1])
            if center_mean > (hline_mean + vline_mean) / 2:
                loop_down = True

            # no loop form, modify current bin label accoding to the mean contact between upstream and downstream bins
            if loop_up == False and loop_down == False:
                #### Here update mat_hic to mat_att and try
                # up_att = np.mean(mat_hic[i, np.max([0,i - up_dist]) : i])
                # down_att = np.mean(mat_hic[i, i+1: np.min([i+down_dist, len(mat_hic)])])
                dist_mid = np.min([up_dist, down_dist])
                up_att = np.mean(mat_hic[i, np.max([0, i - dist_mid]): i])
                down_att = np.mean(mat_hic[i, i + 1: np.min([i + dist_mid, len(mat_hic)])])

                # up_att = np.mean(mat_att[i, np.max([0,i - np.min([up_dist, 2])]) : i])
                # down_att = np.mean(mat_att[i, i+1: np.min([i+np.min([down_dist, 2]), len(mat_hic)])])

                if down_att > up_att:
                    pass
                else:
                    bin_tad_label_new[i] = lb_first
            lb_first = lb
            up_dist = np.sum(bin_tad_label_new == lb_first)
    return bin_tad_label_new



def Get_TAD_domain_with_bin_label(bin_tad_label_new, Chr, chr_size, mat_hic, mat_imputed, resolution, length_cut = 3,
                                  contact_fold_cut = 2, zero_ratio_cut = 0.3):
    """
    Get domain range and type based on the bin tad label
    :param bin_tad_label_new: numpy array, the final domain label of each bin
    :param Chr: str, the chromosome name
    :param chr_size: dict, the chromosome size for all chromosomes
    :param mat_hic: numpy array, the Hi-C contact map of the target chromosome
    :param mat_imputed: numpy array, the imputed Hi-C contact map of the target chromosome
    :param resolution: int, the resolution of the Hi-C contact map
    :param length_cut: int, the length cut-off for domain, below this cut-off, it will be considered as boundary
    :param contact_fold_cut: float, the fold cut-off when comparing domain mean contact with the background,
                             above this cut-off, it will be considered as domain
    :param zero_ratio_cut: float, the ratio cut-off of zero bins in the domain, above this cut-off, it will be considered as gap
    :return: df_tad: pandas DataFrame, the domain range and type of each domain
    """
    if resolution == 10000:
        window_use = 3000000
    elif resolution == 40000:
        window_use = 4000000
    elif resolution == 50000:
        window_use = 5000000
    else:
        window_use = 5000000
    # get background mean contact
    contact_l = []
    window = int(window_use / resolution)
    for i in range(window):
        contact_l += list(np.diag(mat_hic, k = i))
    contact_background = np.mean(contact_l)

    for i in range(window):
        contact_l += list(np.diag(mat_imputed, k = i))
    contact_background_imp = np.mean(contact_l)

    chr_l = []
    st_l = []
    ed_l = []
    tad_bin_l = []
    label_l = []
    type_l = []
    for lb in np.unique(bin_tad_label_new):
        chr_l.append(Chr)
        bin_set = np.where(bin_tad_label_new == lb)[0]
        st = np.min(bin_set)
        ed = np.max(bin_set)
        st_l.append(st)
        ed_l.append(ed)
        tad_bin_l.append(bin_set)
        label_l.append(lb)
        if len(bin_set) < length_cut:
            type_l.append('boundary')
            continue
        contact_mean = np.mean(mat_hic[st:ed+1, st:ed+1])
        contact_mean_imp = np.mean(mat_imputed[st:ed + 1, st:ed + 1])
        if contact_mean / contact_background > contact_fold_cut or contact_mean_imp / contact_background_imp > contact_fold_cut:
            type_l.append('domain')
        else:
            if list(np.diag(mat_hic)[st:ed+1]).count(0) > len(bin_set)*zero_ratio_cut and list(np.diag(mat_imputed)[st:ed+1]).count(0) > len(bin_set)*zero_ratio_cut:
                type_l.append('gap')
            else:
                type_l.append('boundary')
    st_pos = (np.array(st_l) * resolution).astype('int')
    ed_pos = (np.array(ed_l) * resolution + resolution).astype('int')
    if ed_pos[-1] > chr_size[Chr]:
        ed_pos[-1] = chr_size[Chr]
    df_tad = pd.DataFrame(columns = ['chr', 'start_pos', 'end_pos', 'start', 'end', 'tad_bin', 'label', 'type'])
    df_tad['chr'] = chr_l
    df_tad['start_pos'] = st_pos
    df_tad['end_pos'] = ed_pos
    df_tad['start'] = st_l
    df_tad['end'] = ed_l
    df_tad['tad_bin'] = tad_bin_l
    df_tad['label'] = label_l
    df_tad['type'] = type_l
    return df_tad


def combine_TADs_split_windows(TADGATE_tads_res, range_l, resolution):
    '''
    Combine TADs from different split windows
    :param TADGATE_tads_res: dict, the TADs result from different windows
    :param range_l: list, the range of each split window
    :param resolution: int, the resolution of the Hi-C contact map
    :return: df_tad_combine: pandas DataFrame, the combined TADs,  bd_vote_combine: list, the combined boundary score profile,
             CI_ori_combine: the combined original CI profile,
             CI_imp_combine: the combined imputed CI profile,
             pvalue_ori_combine: the combined original p-value,
             pvalue_imp_combine: the combine imputed p-value,
             att_profile_combine
    '''
    df_tad_combine = copy.deepcopy(TADGATE_tads_res['TADs'][0])
    r_ed_tar = range_l[0][2]
    df_tad_combine = df_tad_combine[df_tad_combine['end'] < r_ed_tar]
    df_tad_combine = df_tad_combine.reset_index(drop=True)
    end_pos = df_tad_combine['end'][len(df_tad_combine) - 1]
    for i in range(1, len(list(TADGATE_tads_res['TADs'].keys()))):
        key = i
        r_st = range_l[key][0]
        df_tad_add = copy.deepcopy(TADGATE_tads_res['TADs'][key])
        df_tad_add['start'] = df_tad_add['start'] + r_st
        df_tad_add['end'] = df_tad_add['end'] + r_st
        df_tad_add['start_pos'] = df_tad_add['start_pos'] + int(r_st * resolution)
        df_tad_add['end_pos'] = df_tad_add['end_pos'] + int(r_st * resolution)

        df_filter = df_tad_add[df_tad_add['start'] > end_pos]
        df_tad_combine = pd.concat([df_tad_combine, df_filter], axis=0)
        df_tad_combine = df_tad_combine.reset_index(drop=True)

        r_ed_tar = range_l[key][2]
        df_tad_combine = df_tad_combine[df_tad_combine['end'] < r_ed_tar]
        df_tad_combine = df_tad_combine.reset_index(drop=True)
        end_pos = df_tad_combine['end'][len(df_tad_combine) - 1]
    bd_vote_combine = []
    CI_ori_combine = []
    CI_imp_combine = []
    pvalue_ori_combine = []
    pvalue_imp_combine = []
    att_profile_combine = []
    cluster_bin_lb_combine = []

    for i in range(len(list(TADGATE_tads_res['TADs'].keys()))):
        key = i
        r_st = range_l[key][0]
        r_st_tar = range_l[key][1]
        r_ed_tar = range_l[key][2]
        bd_vote_combine += list(np.array(TADGATE_tads_res['bin_vote_bd_score'][key][r_st_tar - r_st: r_ed_tar - r_st]))
        CI_ori_combine += list(np.array(TADGATE_tads_res['CI_original'][key][r_st_tar - r_st: r_ed_tar - r_st]))
        CI_imp_combine += list(np.array(TADGATE_tads_res['CI_imputed'][key][r_st_tar - r_st: r_ed_tar - r_st]))
        pvalue_ori_combine += list(np.array(TADGATE_tads_res['pvalue_original'][key][r_st_tar - r_st: r_ed_tar - r_st]))
        pvalue_imp_combine += list(np.array(TADGATE_tads_res['pvalue_imputed'][key][r_st_tar - r_st: r_ed_tar - r_st]))
        att_profile_combine += list(np.array(TADGATE_tads_res['att_profile'][key][r_st_tar - r_st: r_ed_tar - r_st]))
        cluster_lb = list(np.array(TADGATE_tads_res['cluster_bin_lb'][key][r_st_tar - r_st: r_ed_tar - r_st]))
        cluster_bin_lb_combine += [str(key) + '_' + x for x in cluster_lb]

    return df_tad_combine, bd_vote_combine, CI_ori_combine, CI_imp_combine, pvalue_ori_combine, pvalue_imp_combine, att_profile_combine, cluster_bin_lb_combine



def TADGATE_call_TADs(TADGATE_embed_all, chr_size, resolution, bd_weight_list = [1, 1, 1, 1], cluster_method = 'Mclust',
                      window_range = 5000000, wd_ci = 5, wd_p = -1, dist = 3, pvalue_cut = 0.05, exp_length = 500000,
                      length_cut = 3, contact_fold_cut = 2, zero_ratio_cut = 0.3, expand_check = 1, filter_method = 'strict',
                      target_chr_l = []):
    """
    Get the TADs based on the TADGATE embedding and attention map
    :param TADGATE_embed_all:  dict, the TADGATE embedding and attention map for all chromosomes
    :param chr_size: dict, the chromosome size for all chromosomes
    :param resolution: int, the resolution of the Hi-C contact map
    :param bd_weight_list: list, the weight list for different boundary score, [w_ori_CI, w_imp_CI, w_att_valley, w_clt_bd]
    :param cluster_method: str, 'Mclust', 'K-means' or 'None', the clustering method used for bin clustering
    :param window_range: int, the window size used for clustering, we split the genome into seperated windows and perform clustering in each window
    :param wd_ci: int, the size of the diamond matrix, used in calculate contrast index
    :param wd_p: int, the size of the diamond matrix, used in calculate contrast p-value, if set -1, use wd_ci to calculate p-value
    :param dist: int, the distance range used to call CI peak, attention valley or boundary peak
    :param pvalue_cut: float, the contrast p-value cut-off for boundary filtering
    :param exp_length: int, the expected length of the domain
    :param length_cut: int, the length cut-off for domain, below this cut-off, it will be considered as boundary
    :param contact_fold_cut: float, the fold cut-off when comparing domain mean contact with the background,
    :param zero_ratio_cut: float, the ratio cut-off of zero bins in the domain, above this cut-off, it will be considered as gap
    :param expand_check: int, the expand range used to check whether the boundary peak can pass p-value cut-off
    :param filter_method: str, 'strict' or 'relax', decide whether add p-value valley below 0.5 to candidate boundary
    :param target_chr_l: list, the target chromosome list, if empty, all chromosomes will be used
    :return: TADGATE_tads_mclust: dict, the TADs and boundary score profile for all target chromosomes.
            "TADs" for the TADs range and type, "cluster_bin_lb" for the bin label after clustering and
            it is set to -1 if no clustering is performed, "bin_vote_bd_score" for the final boundary score profile,
            "bin_domain_label" for the domain label of each bin, "CI_original" for the contrast index of the original Hi-C map,
            "CI_imputed" for the contrast index of the imputed Hi-C map, "pvalue_original" for the contrast p-value of the original
            Hi-C map, "pvalue_imputed" for the contrast p-value of the imputed Hi-C map,  "att_profile" for the smoothed attention
            profile, "run_time" for the running time of the whole process

    """
    w_ori_CI = bd_weight_list[0]
    w_imp_CI = bd_weight_list[1]
    w_att_valley = bd_weight_list[2]
    w_cluster_bd = bd_weight_list[3]
    TADGATE_tads_mclust = {}
    for Chr in list(TADGATE_embed_all.keys()):
        #if Chr != 'chr1':
            #continue
        if len(target_chr_l) != 0:
            if Chr not in target_chr_l:
                continue
        st_time = time.time()
        print('For ' + Chr)
        TADGATE_tads_mclust[Chr] = {}
        tad_l = {}
        bin_l = {}
        bin_vote_l = {}
        bin_mclust_l = {}
        CI_ori_l = {}
        CI_imp_l = {}
        pvalue_ori_l = {}
        pvalue_imp_l = {}
        att_profile_l = {}
        for key in list(TADGATE_embed_all[Chr]['result'].keys()):
            if key == 'full_mat_imputed_sym':
                continue
            print('For sub-matrix ' + str(key) + '...')
            mat_rep = TADGATE_embed_all[Chr]['result'][key]['bin_rep']
            mat_att = TADGATE_embed_all[Chr]['result'][key]['attention_map']
            mat_hic = TADGATE_embed_all[Chr]['mat_split'][key].X
            mat_imputed = TADGATE_embed_all[Chr]['result'][key]['mat_imputed_sym']
            att_profile_smooth = smooth_mat_att(mat_att, sm_size=1)
            CI_value_ori, pvalue_ori = get_CI_value_and_pvalue(mat_hic, wd_ci, wd_p)
            CI_value_imp, pvalue_imp = get_CI_value_and_pvalue(mat_imputed, wd_ci, wd_p)
            if dist == None:
                CI_peak_ori = signal.find_peaks(CI_value_ori)[0]
                CI_peak_imp = signal.find_peaks(CI_value_imp)[0]
                Att_valley = signal.find_peaks(-np.array(att_profile_smooth))[0]
            else:
                CI_peak_ori = signal.find_peaks(CI_value_ori, distance=dist)[0]
                CI_peak_imp = signal.find_peaks(CI_value_imp, distance=dist)[0]
                Att_valley = signal.find_peaks(-np.array(att_profile_smooth), distance=dist)[0]
            window = int(window_range / resolution)
            exp_num = int(window_range / exp_length)
            CI_att_vote_value = np.zeros(len(mat_hic))
            for x in CI_peak_ori:
                CI_att_vote_value[x] += w_ori_CI
            for x in CI_peak_imp:
                CI_att_vote_value[x] += w_imp_CI
            for x in Att_valley:
                CI_att_vote_value[x] += w_att_valley

            bin_label_mclust, bd_vote_final, bin_tad_label_new = Clustering_of_bins_for_bd(window, mat_rep, mat_hic,
                                                                                           CI_att_vote_value,
                                                                                           cluster_method, bd_weight_list,
                                                                                           pvalue_ori, pvalue_imp, pvalue_cut,
                                                                                           exp_num, dist, w_cluster_bd,
                                                                                           expand_check, filter_method)

            df_tad = Get_TAD_domain_with_bin_label(bin_tad_label_new, Chr, chr_size, mat_hic, mat_imputed, resolution, length_cut = length_cut,
                                                   contact_fold_cut = contact_fold_cut, zero_ratio_cut = zero_ratio_cut)
            tad_l[key] = df_tad
            bin_l[key] = bin_tad_label_new
            bin_mclust_l[key] = bin_label_mclust
            bin_vote_l[key] = bd_vote_final
            CI_ori_l[key] = CI_value_ori
            CI_imp_l[key] = CI_value_imp
            pvalue_ori_l[key] = pvalue_ori
            pvalue_imp_l[key] = pvalue_imp
            att_profile_l[key] = att_profile_smooth

        TADGATE_tads_mclust[Chr]['TADs'] = tad_l
        TADGATE_tads_mclust[Chr]['cluster_bin_lb'] = bin_mclust_l
        TADGATE_tads_mclust[Chr]['bin_vote_bd_score'] = bin_vote_l
        TADGATE_tads_mclust[Chr]['bin_domain_label'] = bin_l
        TADGATE_tads_mclust[Chr]['CI_original'] = CI_ori_l
        TADGATE_tads_mclust[Chr]['CI_imputed'] = CI_imp_l
        TADGATE_tads_mclust[Chr]['pvalue_original'] = pvalue_ori_l
        TADGATE_tads_mclust[Chr]['pvalue_imputed'] = pvalue_imp_l
        TADGATE_tads_mclust[Chr]['att_profile'] = att_profile_l

        if len(TADGATE_embed_all[Chr]['result'].keys()) > 1:
            print(TADGATE_embed_all[Chr]['result'].keys())
            df_tad_combine, bd_vote_combine, CI_ori_combine, CI_imp_combine, pvalue_ori_combine, pvalue_imp_combine, att_profile_combine, cluster_bin_lb_combine = combine_TADs_split_windows(TADGATE_tads_mclust[Chr], TADGATE_embed_all[Chr]['range'],resolution)


            TADGATE_tads_mclust[Chr]['full'] = {}
            TADGATE_tads_mclust[Chr]['full']['TADs'] = df_tad_combine
            TADGATE_tads_mclust[Chr]['full']['bin_vote_bd_score'] = bd_vote_combine
            TADGATE_tads_mclust[Chr]['full']['CI_original'] = CI_ori_combine
            TADGATE_tads_mclust[Chr]['full']['CI_imputed'] = CI_imp_combine
            TADGATE_tads_mclust[Chr]['full']['pvalue_original'] = pvalue_ori_combine
            TADGATE_tads_mclust[Chr]['full']['pvalue_imputed'] = pvalue_imp_combine
            TADGATE_tads_mclust[Chr]['full']['att_profile'] = att_profile_combine
            TADGATE_tads_mclust[Chr]['full']['cluster_bin_lb'] = cluster_bin_lb_combine
        ed_time = time.time()
        run_time = ed_time - st_time
        print('Running time ' + str(run_time) + 's')
        TADGATE_tads_mclust[Chr]['run_time'] = run_time
    return TADGATE_tads_mclust





