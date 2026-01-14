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



import pandas as pd

def pretty_print_total_res(total_res, digits=4, sort_by="mean", show_rank=True):
    """
    total_res: {exp: {lang: {task: score}}}
    输出：
      - exp × lang 的矩阵（每格是该 lang 对应 task 的 score）
      - mean / max 汇总列
      - 可选排名
    """
    rows = []
    for exp, lang_dict in total_res.items():
        row = {"exp": exp}
        for lang, task_dict in lang_dict.items():
            # task_dict 形如 {'MMLU': 0.33}，取唯一 value
            if isinstance(task_dict, dict) and len(task_dict) > 0:
                score = float(next(iter(task_dict.values())))
            else:
                score = float("nan")
            row[lang] = score
        rows.append(row)

    df = pd.DataFrame(rows).set_index("exp")

    # 保证列顺序稳定
    lang_cols = [c for c in ["en", "vi", "zh"] if c in df.columns] + \
               [c for c in df.columns if c not in ["en", "vi", "zh"]]
    df = df[lang_cols]

    df["mean"] = df.mean(axis=1, numeric_only=True)
    df["max"] = df.max(axis=1, numeric_only=True)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    df_show = df.round(digits)

    if show_rank:
        df_show.insert(0, "rank", range(1, len(df_show) + 1))

    print("\n" + "=" * 90)
    print("LangSpec-F1 Summary (exp × lang)")
    print("=" * 90)
    print(df_show.to_string())
    print("=" * 90)

    return df






org_model_task_result = {
    'MMLU':0.457911,
    'C-eval':0.34695393759286774,
    'Belebele_vi':0.3722
}

LAPE_diff_mask={
    'en':{
    'MMLU':0.4576,
    'C-eval':0.3350668647845468,
    'Belebele_vi':0.3711
},
    'vi':{
    'MMLU':0.4553,
    'C-eval':0.33803863298662706,
    'Belebele_vi':0.3833
},
    'zh':{
    'MMLU':0.4589089873237431,
    'C-eval':0.3410104011887073,
    'Belebele_vi':0.3778
}
}
LRP_diff_mask={
    'en':{
    'MMLU':0.3483,
    'C-eval':0.2800891530460624,
    'Belebele_vi':0.2811
},
    'vi':{
    'MMLU':0.3517,
    'C-eval':0.2816,
    'Belebele_vi':0.2233
},
    'zh':{
    'MMLU':0.4366,
    'C-eval':0.3313521545319465,
    'Belebele_vi':0.3344
}
}

'''
random mask


'''
LRP_random_diff_mask={
    'en':{
    'MMLU':0.4249,
    'C-eval':0.3179791976225854,
    'Belebele_vi':0.3678
},
    'vi':{
    'MMLU':0.4486,
    'C-eval':0.32689450222882616,
    'Belebele_vi':0.36
},
    'zh':{
    'MMLU':0.4375,
    'C-eval':0.3165,
    'Belebele_vi':0.36
}
}


'''
base neuron & mask chat model

'''


org_model_task_result_baseneuron_maskchat = {
    'MMLU':0.47201253382709013,
    'C-eval':0.3491827637444279,
    'Belebele_vi':0.4066666666666667
}

LAPE_diff_mask_baseneuron_maskchat={
    'en':{
    'MMLU':0.47336561743341404,
    'C-eval':0.3514115898959881,
    'Belebele_vi':0.4044444444444
},
    'vi':{
    'MMLU':0.4719,
    'C-eval':0.3476968796433878,
    'Belebele_vi':0.4133333
},
    'zh':{
    'MMLU':0.4719,
    'C-eval':0.3447,
    'Belebele_vi':0.4078
}
}


LRP_diff_mask_baseneuron_maskchat={
    'en':{
    'MMLU':0.4415,
    'C-eval':0.3351,
    'Belebele_vi':0.336666666
},
    'vi':{
    'MMLU':0.44274319897450504,
    'C-eval':0.3276374442793462,
    'Belebele_vi':0.2867
},
    'zh':{
    'MMLU':0.4645,
    'C-eval':0.3462109955423477,
    'Belebele_vi':0.3733
}
}





# lang & task map
lang2targettask={
    'en': 'MMLU',
    'vi': 'Belebele_vi',
    'zh': 'C-eval'
}


exp_map=[
    ('baseNeuron_baseGeneration_LAPE', LAPE_diff_mask, org_model_task_result),
    ('baseNeuron_baseGeneration_LRP', LRP_diff_mask, org_model_task_result),
    ('baseNeuron_baseGeneration_randommask', LRP_random_diff_mask, org_model_task_result),
    ('baseNeuron_chatGeneration_LAPE', LAPE_diff_mask_baseneuron_maskchat, org_model_task_result_baseneuron_maskchat),
    ('baseNeuron_chatGeneration_LRP', LRP_diff_mask_baseneuron_maskchat, org_model_task_result_baseneuron_maskchat),
    
]


#total res

total_res = {}

for iexpname, imask_resm, iorg_res in exp_map:

    ires_dict = {}
    for ilang, ires in imask_resm.items():
        res_score = langspecf1_utils.get_nlu_metric_score(iorg_res, ires, iexpname, ['MMLU', 'C-eval', 'Belebele_vi'], lang2targettask[ilang] )
        ires_dict[ilang] = res_score
    
    total_res[iexpname] = ires_dict

#print(total_res)

df = pretty_print_total_res(total_res, digits=4, sort_by="mean")







