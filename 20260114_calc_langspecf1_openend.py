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
    
    
    langspecf1_utils.show_result('LAPE', res_acuall_score, res_score)


    print('first five:')
    for imt_name, itmp_score in sorted(final_score.items(), key = lambda x: x[1], reverse=True )[:5]:
        langspecf1_utils.show_result(imt_name, res_acuall_score, res_score)
    
    
    #langspecf1_utils.show_result('th_0.25_selected_LRP_kur_res_zscore_margin_selected', res_acuall_score, res_score)

    #langspecf1_utils.show_result('th_0.5_selected_LRP_kur_res_zscore_margin_selected', res_acuall_score, res_score)

    #langspecf1_utils.show_result('th_1_selected_LRP_kur_res_zscore_margin_selected', res_acuall_score, res_score)

    #langspecf1_utils.show_result('th_0_selected_LRP_kur_res_zscore', res_acuall_score, res_score)

    #langspecf1_utils.show_result('th_0.5_selected_LRP_kur_res_zscore', res_acuall_score, res_score)


    # random
    #langspecf1_utils.show_result('th_1_selected_LRP_kur_res_random_zscore', res_acuall_score, res_score)

    # global random
    #langspecf1_utils.show_result('th_1_selected_LRP_kur_res_global_random_zscore', res_acuall_score, res_score)

    # full random mask 'fully_random_mask_zscore'
    #langspecf1_utils.show_result('fully_random_mask_zscore', res_acuall_score, res_score)

if __name__=='__main__':
    
    # llama2 7b chat
    #judege_path ='/root/autodl-fs/LRP/open_ended_data_generation/20251204_all_exp/1208_all_generation4judge.json_gpt4o.json'
    
    # llama2 7b base
    #judege_path ='/root/autodl-fs/LRP/open_ended_data_generation/20251210_llama2_7b_base/1210_all_generation4judge.json_gpt4o.json'
    
    # llama2 7b base lang neuron & chat generation
    #judege_path= '/root/autodl-fs/LRP/open_ended_data_generation/20251210_llama2_7b_neuronFromBase_GenerationFromChat/20251210_LRP_generation_all_models.json4judge.json_gpt4o.json'
    
    
    # llama2 7b base LRP random mask
    #judege_path= '/root/autodl-fs/LRP/open_ended_data_generation/20251218_LRP_random_mask/20251218_LRP_generation_all.json4judge.json_gpt4o.json'

    # 20260119 new random 5000samples
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260115_newrandom5000samples_cal_llama2_7b_base/20260119_generation_all_models.json_gpt4o.json'

    #20260119 new random 5000samples & random mask 3time & judge 3time
    #judege_path ='/root/autodl-fs/LRP/open_ended_data_generation/20260119_newrandom5000samples_cal_llama2_7b_base_random_mask_0_1_2_total/20260119_generation_all_models_random_mask0_1_2.json_gpt4o_judge3times.json'
    ##20260119 new random 5000samples & random mask 3time & global random
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260120_newrandom5000samples_cal_llama2_7b_base_globalrandom_mask_0/20260120_all_method_add_global_random_ans4judge.json_gpt4o_.json'

    # 20260121 new random 5000samples & random mask 3time & global random & full random 3times
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260121_newrandom5000samples_cal_llama2_7b_base_fullrandom_mask_0/20260121_all_method_add_global_random_And_full_random.json_gpt4o_.json'

    # 20260123 new vi zh random 3times
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/0260123_generation_all_models_org_lape_random_all.json_gpt4o_3times.json'

    # 20260127 new vi 
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/20260127all_gene_and_random.json_gpt4o.json'
    # 20260127 new vi 3times
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/20260127_generation_all_modelsAnd_random3times.json_gpt4o.json'
    #20260128 new search all
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260128_newrandom1000samples_llama2_base/20260128_all_and_org_LAPE.json_gpt4o.json'

    # 20260304 arr paper reproduce
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260304_arr_paper_llama2_7b_base_/20260304_generation_all_models.json_gpt4o.json'

    #20260304 arr paper reproduce random
    #judege_path='/root/autodl-fs/LRP/open_ended_data_generation/20260304_arr_paper_llama2_7b_base_/20260304_generation_all_models_add1time_random.json_gpt4o.json'
    #20260304 arr paper reproduce random - add fully random
    judege_path = '/root/autodl-fs/LRP/open_ended_data_generation/20260304_arr_paper_llama2_7b_base_/20260304_generation_all_models_add1time_random_A_fully_random.json_gpt4o.json'
    main(judege_path)





