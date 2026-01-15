'''
    20260113:
        设置合适的F1score的参数，获取最合适的方法！
'''


# import lib

import numpy as np
import datasets

import transformers

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
# download checkpoint
from accelerate import load_checkpoint_and_dispatch
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

#from accelerate import load_checkpoint_and_dispatch

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

#import util
#import importlib

#importlib.reload(langspecf1_utils)      # 只能 reload 模块本身
#from util import calc_ppl, get_test_data   # reload 后再重新 import 函数

import pandas as pd
import copy


# reload utils
#import util
import importlib
import langspecf1_utils
importlib.reload(langspecf1_utils)      # 只能 reload 模块本身
#from util import calc_ppl, get_test_


'''
    load judge file
'''
def main(judege_path):
        
    
    
    
    '''
        get LAPE result
    
    '''

    if False:
        # lape neuronFromBase & chat generation
        ds_lape_mask_base = datasets.load_dataset('json', data_files = '/root/autodl-fs/LRP/open_ended_data_generation/20251204_all_exp/1208_all_generation4judge.json_gpt4o.json')
        
        
        
        
        df_lape_mask_base = ds_lape_mask_base['train'].to_pandas()
        
        df_lape_mask_base_en = df_lape_mask_base.loc[(df_lape_mask_base['model_type']=='mask_en|open_ended')].reset_index(drop=True)
        df_lape_mask_base_vi = df_lape_mask_base.loc[(df_lape_mask_base['model_type']=='mask_vi|open_ended')].reset_index(drop=True)
        
        df_lape_mask_base_zh = df_lape_mask_base.loc[(df_lape_mask_base['model_type']=='mask_zh|open_ended')].reset_index(drop=True)
        
        df_lape_mask_lape = pd.concat([df_lape_mask_base_en,df_lape_mask_base_vi, df_lape_mask_base_zh ])
        
        
        df_lape_mask_lape['model_type'].value_counts()
    
    
    
    '''
        load LRP method result
    
    '''
    
    ds_judge_res = datasets.load_dataset('json', data_files = judege_path)
    
    tmp = ds_judge_res['train'].to_pandas()
    print(tmp['model_type'].value_counts())
    
    
    result_dict, df_judge_res = langspecf1_utils.get_analysis_res(ds_judge_res)
    
    df_judge_res['mask_neuron_lang'] = df_judge_res['class_tpye'].apply(langspecf1_utils.get_mask_neuron_lang)
    df_judge_res['method_name'] = df_judge_res['class_tpye'].apply(langspecf1_utils.get_method_name)
    
    
    df_judge_res_no_mask = df_judge_res.loc[df_judge_res['method_name'] != 'mask'].reset_index(drop=True)
    
    res_score, res_acuall_score, lang_score_baseline = langspecf1_utils.calc_metric(df_judge_res, result_dict, beta = 2.5)
    
    # res_score_mean
    final_score={}
    for imethod in res_score:
        final_score[imethod] = sum(res_score[imethod].values())/len(res_score[imethod].values())
        
    print('==='*10)
    print('==='*10)
    print('sorted method...')
    print(sorted(final_score.items(), key = lambda x: x[1], reverse=True ))
    print('==='*10)
    print('==='*10)
    
    print('baseline:', lang_score_baseline)
    
    
    langspecf1_utils.show_result('mask', res_acuall_score, res_score)
    
    
    langspecf1_utils.show_result('th_1_selected_LRP_kur_res_zscore', res_acuall_score, res_score)


if __name__=='__main__':
    
    # llama2 7b chat
    judege_path ='/root/autodl-fs/LRP/open_ended_data_generation/20251204_all_exp/1208_all_generation4judge.json_gpt4o.json'
    
    # llama2 7b base
    #judege_path ='/root/autodl-fs/LRP/open_ended_data_generation/20251210_llama2_7b_base/1210_all_generation4judge.json_gpt4o.json'
    
    # llama2 7b base lang neuron & chat generation
    #judege_path= '/root/autodl-fs/LRP/open_ended_data_generation/20251210_llama2_7b_neuronFromBase_GenerationFromChat/20251210_LRP_generation_all_models.json4judge.json_gpt4o.json'
    
    
    # llama2 7b base LRP random mask
    #judege_path= '/root/autodl-fs/LRP/open_ended_data_generation/20251218_LRP_random_mask/20251218_LRP_generation_all.json4judge.json_gpt4o.json'

    main(judege_path)



