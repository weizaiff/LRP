import torch

from tqdm import tqdm
# 只保留MLP 层

def extract_mlp_data_pre_calc(data):
    new_res = {}
    for ikey in data.keys():
        if 'mlp' in ikey:
            new_res[ikey] = data[ikey]
    return new_res

def extract_mlp_data(data):
    new_res = {}
    for ikey in data.keys():
        if 'mlp' in ikey:
            tmp_list = data[ikey]['kurt_vals'].tolist()
            for index in range(len(tmp_list)):
                new_res[f"{ikey}_index_{index}"] = tmp_list[index]
    return new_res

import numpy as np
from scipy.stats import skew, kurtosis

def compute_skew_kurtosis(matrix):
    """
    对每一列计算偏度、峰度
    matrix shape: (num_samples, num_neurons)

    return:
        skewness: shape (num_neurons,)
        kurt:     shape (num_neurons,)
    """
    skewness = None#skew(matrix, axis=0, bias=False)
    kurt_vals = kurtosis(matrix, axis=0, bias=False)  # Fisher=True 默认，峰度=0为正态分布

    return skewness, kurt_vals
    
def calc_skew_kurtosis(data1):
    '''
    计算神经元的，偏度和峰度

    '''
    result={}

    bar = tqdm(total=len(data1.keys()))

    for ikey in data1.keys():
        if "layers" not in ikey: continue
        # mean & var dict
        result[ikey]={}

        
        result[ikey]['skewness'],  result[ikey]['kurt_vals'] = compute_skew_kurtosis(
            np.array([itmp.float().numpy().tolist() for itmp in data1[ikey] ]),
        )
                


        bar.update(1)
        
    return result


    

    
'''
选取top *%

'''
def get_top_1perc_neuron(data):

    count_len = int(len(data)*0.01)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=True)

    return sorted_data[:count_len]

def get_top_0_5_perc_neuron(data):

    count_len = int(len(data)*0.005)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=True)

    return sorted_data[:count_len]

def get_top_0_1_perc_neuron(data):

    count_len = int(len(data)*0.001)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=True)

    return sorted_data[:count_len]

def get_top_0_0_1perc_neuron(data):

    count_len = int(len(data)*0.0001)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=True)

    return sorted_data[:count_len]

'''
选取bottom *%

'''
def get_bottom_1perc_neuron(data):

    count_len = int(len(data)*0.01)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=False)

    return sorted_data[:count_len]



def get_bottom_0_5_perc_neuron(data):

    count_len = int(len(data)*0.005)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=False)

    return sorted_data[:count_len]

def get_bottom_0_1_perc_neuron(data):

    count_len = int(len(data)*0.001)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=False)

    return sorted_data[:count_len]
def get_bottom_0_0_1_perc_neuron(data):

    count_len = int(len(data)*0.0001)

    sorted_data = sorted(data.items(), key = lambda x: x[1], reverse=False)

    return sorted_data[:count_len]
    
'''
save reuslt


'''
import os
def save_neuron(neuron, save_path):

    tmp_dir = '/'.join(save_path.split('/')[:-1])
    os.makedirs(tmp_dir, exist_ok =True)

    torch.save(neuron, save_path)





'''
选取当前语种和其他语种的gap rate

'''
def calc_gap_rate(data_list):

    res_list = [{} for _ in range(len(data_list))]

    for ikey in data_list[0].keys():
        for index in range(len(data_list)):
            gap_list = list(range(len(data_list)))
            gap_list.remove(index)

            data_list[index][ikey] = ((data_list[index][ikey] - data_list[gap_list[0]][ikey]) + (data_list[index][ikey] - data_list[gap_list[1]][ikey]))/2/ data_list[index][ikey]

    return data_list

def get_quantiles(data, quantiles=None):
    """
    计算给定列表的分位数。

    参数:
        data (list): 数值型列表。
        quantiles (list of float, optional): 要计算的分位数列表，例如 [0.25, 0.5, 0.75]。
                                            默认为 [0.0, 0.25, 0.5, 0.75, 1.0]（五数概括）。

    返回:
        dict: 分位数及其对应的值。
    """
    if not data:
        raise ValueError("输入列表不能为空。")
    
    if quantiles is None:
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # 转换为 NumPy 数组以方便计算
    arr = np.array(data.float())
    
    # 计算分位数
    values = np.quantile(arr, quantiles)
    
    # 构建结果字典
    result = {f"{q*100:.0f}%": v for q, v in zip(quantiles, values)}
    return result

import numpy as np

def compute_zscore(divergence_dict):
    # 取所有数值
    values = np.array(list(divergence_dict.values()), dtype=float)
    
    # 计算均值和标准差
    mean = values.mean()
    std = values.std()

    if std == 0:
        raise ValueError("Standard deviation is zero, cannot compute z-score.")

    # 构造新的 z-score 字典
    z_dict = {}
    for k, v in divergence_dict.items():
        z = (v - mean) / std
        z_dict[k] = z

    return z_dict
# save zscore > num

def get_gt_zscore(divergence_dict, th):
    print('org len:', len(divergence_dict))
    res_cit={}

    for ikey, ival in divergence_dict.items():
        if ival>th:
            res_cit[ikey] = ival


    print('final selected len:', len(res_cit))
    return res_cit

import numpy as np

def select_language_specific_neurons(all_D, abs_threshold=None, margin_threshold=0.0):
    """
    all_D: dict(lang -> dict(neuron_id -> D_value))
    abs_threshold: dict(lang -> threshold) or scalar
    margin_threshold: scalar (global margin threshold)
    
    return: dict(lang -> selected neuron_ids)
    """
    
    langs = list(all_D.keys())
    neurons = list(all_D[langs[0]].keys())  # assume aligned neurons across languages
    
    # 如果 abs_threshold 是 scalar，变成每个语言都一样
    if not isinstance(abs_threshold, dict):
        abs_threshold = {lang: abs_threshold for lang in langs}
    
    # 结果集合
    selected = {lang: {} for lang in langs}

    print('before selected:',)
    for ilang,ival in all_D.items():
        print(ilang, len(ival))
    
    # 遍历每个语言、一颗神经元
    for lang in langs:
        for nid in neurons:
            D_lang = all_D[lang][nid]
            
            # 绝对值条件
            if D_lang < abs_threshold[lang]:
                continue
            
            # 相对优势：max of others
            max_other = max(all_D[other][nid] for other in langs if other != lang)
            
            # margin
            if (D_lang - max_other) >= margin_threshold:
                selected[lang][nid] = D_lang
                
    print('after selected:',)
    for ilang,ival in selected.items():
        print(ilang, len(ival))
    return selected


import numpy as np

def compute_soft_preference(all_D, eps=1e-9):
    """
    all_D: dict(lang -> dict(neuron_id -> D_value))
    return: dict(lang -> dict(neuron_id -> preference_score))
            preference_score in [0,1], sum over langs = 1 (soft preference)
    """
    langs = list(all_D.keys())
    neurons = list(all_D[langs[0]].keys())

    # build soft preference scores
    pref = {lang: {} for lang in langs}

    for nid in neurons:
        # collect absolute contributions for all langs
        vals = []
        for lang in langs:
            vals.append(abs(all_D[lang][nid]))
        vals = np.array(vals, dtype=float)
        denom = vals.sum() + eps

        # normalize
        for i, lang in enumerate(langs):
            pref[lang][nid] = vals[i] / denom

    return pref


def select_by_soft_preference(pref, threshold=0.6):
    """
    pref: output of compute_soft_preference
    threshold: e.g. 0.6 means neuron is strongly preferred to this lang
    return: dict(lang -> list(neuron_ids))
    """
    print('before selected:',)
    for ilang,ival in pref.items():
        print(ilang, len(ival))
        
    langs = list(pref.keys())
    selected = {lang: {} for lang in langs}

    for lang in langs:
        for nid, score in pref[lang].items():
            if score >= threshold:
                selected[lang][nid] = score
    print('after selected:',)
    for ilang,ival in selected.items():
        print(ilang, len(ival))
    return selected





    


