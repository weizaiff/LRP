#import transformers


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
# download checkpoint
from accelerate import load_checkpoint_and_dispatch
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

#from accelerate import load_checkpoint_and_dispatch

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json



import copy

import datasets

import util
import importlib

importlib.reload(util)      # 只能 reload 模块本身
from util import calc_ppl, get_test_data, get_open_ended_answer_vllm   # reload 后再重新 import 函数



import argparse
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from tqdm import tqdm



import os
import copy
import datasets
import torch
from transformers import AutoConfig
from datetime import datetime
from tqdm import tqdm
from open_ended_utils import *
from types import MethodType

model_path ='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf'
is_llama = True# bool(model_path.find('llama') >= 0)
config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)

# llama2 7b base
save_dir = "/root/autodl-fs/LRP/open_ended_data_generation/20260119_newrandom5000samples_cal_llama2_7b_base_random_mask_2"

#neuron path
#BASE = "/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base"
#random  mask time0 
BASE = "/root/autodl-fs/LRP_kur_res/20260119_newrandom5000samples_cal_llama2_7b_base_random_neuron_time2_"

IS_ADD_LAPE_MODEL_LIST = False #是否加上LAPE的open-ended测试

# ======== Build model list (auto) ========
#BASE = "/root/autodl-fs/LRP_kur_res/20251204_5000samples_cal_llama2_7b_chat"

# llama2 7b chat

#save_dir = "/root/autodl-fs/LRP/open_ended_data_generation/20251210_llama2_7b_neuronFromBase_GenerationFromChat"


#BASE = "/root/autodl-fs/LRP_kur_res/20251210_5000samples_cal_llama2_7b_base"

# llama2 7b base random mask
#save_dir = "/root/autodl-fs/LRP/open_ended_data_generation/20251218_LRP_random_mask"
# ======== Build model list (auto) ========
#BASE = "/root/autodl-fs/LRP_kur_res/20251210_5000samples_cal_llama2_7b_base_random_neuron"



#model_path ='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-chat-hf'


#model, tokenizer = load_model(model_path)

# 初始化 vLLM
llm =LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True) #LLM(model=model_path, tensor_parallel_size=1)


def get_need_exp(model_list):
    

    target_list=[('th_1_selected_LRP_kur_res_random', np.float64(0.19038895836479783))]


    new_model_List = []
    for ipath, iname in model_list:
        for name, _ in target_list:
            if name in iname or iname=='org_model':
                new_model_List.append((ipath, iname))
    return new_model_List
        


# ======== Config ========
config = AutoConfig.from_pretrained(model_path)

# load test data
ds_test = datasets.load_dataset(
    'json',
    data_files='/root/autodl-fs/LRP/open_ended_dataset/all_data.json'
)

os.makedirs(save_dir, exist_ok=True)

# ======== Date stamp ========
dt = datetime.now().strftime("%Y%m%d")



def build_model_list(llm, base=BASE):
    ml = []#[(llm, "org_model")]
    for fname in sorted(os.listdir(base)):
        if fname.endswith(".pt") and not fname.startswith("all_mlp"):
            path = f"{base}/{fname}"
            ml.append((path, fname.replace(".pt","")))
    return ml

model_list = build_model_list(llm)

# filter
#model_list = (get_need_exp(model_list))


# add LAPE exp list
if IS_ADD_LAPE_MODEL_LIST:
    # neuron usage from: 20251127_calc_org_LAPE_ppl.ipynb
    neuron_path = "/root/autodl-fs/Language-Specific-Neurons/LLaMA-2-7B.neuron.pth"
    print('='*10)
    print('='*10)
    print("neuron path:", neuron_path)
    print('='*10)
    print('='*10)
    '''
     format: method_name+ language
    '''
    model_list+=[
        (neuron_path, 'LAPE_en'),
        (neuron_path, 'LAPE_vi'),
        (neuron_path, 'LAPE_zh')
    ]
        
print('model_list:', model_list)

data_list = [
    (ds_test, 'open_ended')
]

# ======== merge buffer ========
ds_list = []

print(f"[INFO] Total models: {len(model_list)}")

model_list_bar = tqdm(total=len(model_list), desc="Model Progress")

# ======== Running ========
for model_path_pt, model_name in model_list:
    print("*" * 40)
    print("Model:", model_name)

    # loop datasets
    for i_data, data_name in data_list:
        
        # ======== Save path ========
        save_name = f"{dt}_LRP_generation_{model_name}_{data_name}.json"
        save_path = os.path.join(save_dir, save_name)

        # ======== Skip if exists ========
        if os.path.exists(save_path):
            print(f"[Skip] Exists: {save_path}")
            # load data
            ds_test_tmp = datasets.load_dataset('json', data_files= save_path)
            ds_list.append(ds_test_tmp['train'])
            continue

        # ======== Load Mask ========
        if model_name != 'org_model':
            
            
            if 'LAPE' in model_name:
                #LRP
                #tmp_neuron = torch.load(model_path_pt, weights_only=False)
                need_lang = model_name.split('_')[-1]
                assert len(need_lang)>0
                i_model = get_mask_neuron_model_vllm_LAPE(llm, model_path_pt, need_lang, is_llama = is_llama)
                
            else:
                #LRP
                tmp_neuron = torch.load(model_path_pt, weights_only=False)
                i_model = get_mask_neuron_model_vllm_LRP(llm, convert_LAPE_format(tmp_neuron, config), is_llama = is_llama)
        else:
            i_model = llm

        # ======== Run ========
        ds_test_tmp = copy.deepcopy(i_data)
        text_list = list(ds_test_tmp['train']['text'])

        ans_list = get_open_ended_answer_vllm(i_model, text_list)

        # ======== Meta ========
        model_col = [model_name + "|" + data_name] * len(ans_list)
        ds_test_tmp['train'] = ds_test_tmp['train'].add_column(name='answer', column=ans_list)
        ds_test_tmp['train'] = ds_test_tmp['train'].add_column(name='model_type', column=model_col)

        # ======== append global ========
        ds_list.append(ds_test_tmp['train'])

        # ======== Save one snapshot ========
        ds_test_tmp['train'].to_json(save_path, force_ascii=False)
        print(f"[Saved] {save_path}")

    model_list_bar.update(1)


'''
生成judge的prompt

'''
'''
生成judge的prompt

'''

prompt_template = (
    "You are a neutral judge. Score the model’s answer from 1 to 10 based on correctness, completeness, clarity, and usefulness.\n"
    "Respond ONLY in this JSON format:\n"
    '{{"score": <1-10>, "reason": "<brief reason>"}}\n\n'
    "Question:\n{question}\n\n"
    "Answer:\n{answer}"
)

def get_judge_promt(x):
    return prompt_template.format(question= x[0], answer= x[1])
    

# ======== Save final merge ========
if len(ds_list) > 0:
    ds_final = datasets.concatenate_datasets(ds_list)

    # add judge prompt
    df_final = ds_final.to_pandas()
    df_final['judge_prompt'] = df_final[['text','answer']].apply(lambda x: prompt_template.format(question= x[0], answer= x[1]), axis=1)

    ds_final = datasets.Dataset.from_pandas(df_final)
    
    final_path = os.path.join(save_dir, f"{dt}_generation_all_models.json")
    ds_final.to_json(final_path, force_ascii=False)
    print(f"[Saved final merged] {final_path}")
else:
    print("[Info] No new results, final merge skipped.")


    
    