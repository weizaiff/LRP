'''
    每一层随机random 挑选同样个数的神经元再mask

'''

#save random neuron 
import gc
import os

from open_ended_utils import *


#save_dir='/root/autodl-tmp/LRP_llama2_7b_base_BaseNeuron_RandomMask_lang_{}'
#save_dir='/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_{}'

import numpy as np



def save_LAPE_model(model, save_dir, activation_mask_path, lang_list = ['en', 'vi', 'zh']):

    '''
    get LAPE model
    
    '''
    
    #save_dir='/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_{}'
    #save_dir='/root/autodl-tmp/llama2_7b_chat_BaseNeuron_ChatMask_lang_{}'
    
    #lang_list = ['en', 'vi', 'zh']
    #activation_mask_path='/root/autodl-fs/Language-Specific-Neurons/LLaMA-2-7B.neuron.pth'
    for ilang in lang_list:
        
        tmp_model = get_mask_neuron_model(copy.deepcopy(model), activation_mask_path, ilang)
        print(tmp_model.state_dict()[f'model.layers.0.mlp.gate_proj.weight'].sum())
        save_model(tmp_model, tokenizer, save_dir.format(ilang))
    
        tmp_model.cpu()
        del tmp_model
        gc.collect()


def save_model(model, tokenizer, save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

def save_LRP_model(model, save_dir, activation_mask_path_map, lang_list = ['en', 'vi', 'zh']):
    '''
    get lrp model
    
    '''
    #save_dir='/root/autodl-tmp/LRP_llama2_7b_base_BaseNeuron_BaseMask_lang_{}'
    #save_dir='/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_{}'
    
    #lang_list = ['en', 'vi', 'zh']
    #activation_mask_path_map = {
    #    'en': '/root/autodl-fs/LRP_kur_res/20251210_5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_en_zscore.pt',
    #    'vi': '/root/autodl-fs/LRP_kur_res/20251210_5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_vi_zscore.pt',
    #    'zh': '/root/autodl-fs/LRP_kur_res/20251210_5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_zh_zscore.pt'
    #}
    for ilang in lang_list:
        # LRP version
        tmp_model = get_mask_neuron_model_LRP(copy.deepcopy(model), convert_LAPE_format(torch.load(activation_mask_path_map[ilang], weights_only=False), config), ilang)
        print(tmp_model.state_dict()[f'model.layers.0.mlp.gate_proj.weight'].sum())
        print('save dir:', save_dir.format(ilang))
        save_model(tmp_model, tokenizer, save_dir.format(ilang))
    
        tmp_model.cpu()
        del tmp_model
        gc.collect()

org_model_path ='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf/' # base model 
is_llama=True

model, tokenizer = load_model(org_model_path)
config = AutoConfig.from_pretrained(org_model_path)

# LRP model
if True:
    if False: #random
        save_dir= '/root/autodl-tmp/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh_random_{}'
        activation_mask_path_map = {
                'en': '/root/autodl-fs/LRP_kur_res/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh_random_neuron_time0_/th_1_selected_LRP_kur_res_random_en_zscore.pt',
                'vi': '/root/autodl-fs/LRP_kur_res/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh_random_neuron_time0_/th_1_selected_LRP_kur_res_random_vi_zscore.pt',
                'zh': '/root/autodl-fs/LRP_kur_res/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh_random_neuron_time0_/th_1_selected_LRP_kur_res_random_zh_zscore.pt'
            }
        lang_list = ['en', 'vi', 'zh']
        #lang_list = ['en']

    else:
        save_dir= '/root/autodl-tmp/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh_{}'
        activation_mask_path_map = {
                'en': '/root/autodl-fs/LRP_kur_res/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/th_1_selected_LRP_kur_res_en_zscore.pt',
                'vi': '/root/autodl-fs/LRP_kur_res/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/th_1_selected_LRP_kur_res_vi_zscore.pt',
                'zh': '/root/autodl-fs/LRP_kur_res/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/th_1_selected_LRP_kur_res_zh_zscore.pt'
            }
        lang_list = ['en', 'vi', 'zh']
        #lang_list = ['en']
    
    save_LRP_model(model, save_dir, activation_mask_path_map, lang_list)
else:

    # LRP random-mask model
    save_dir= '/root/autodl-tmp/20250120_LRP_llama2_7b_base_BaseNeuron_random_BaseMask_lang_{}'
    activation_mask_path_map = {
            'en': '/root/autodl-fs/LRP_kur_res/20260119_newrandom5000samples_cal_llama2_7b_base_random_neuron_time0_/th_1_selected_LRP_kur_res_random_en_zscore.pt',
            'vi': '/root/autodl-fs/LRP_kur_res/20260119_newrandom5000samples_cal_llama2_7b_base_random_neuron_time0_/th_1_selected_LRP_kur_res_random_vi_zscore.pt',
            'zh': '/root/autodl-fs/LRP_kur_res/20260119_newrandom5000samples_cal_llama2_7b_base_random_neuron_time0_/th_1_selected_LRP_kur_res_random_zh_zscore.pt',
        }
    lang_list = ['en', 'vi', 'zh']
    
    save_LRP_model(model, save_dir, activation_mask_path_map, lang_list)






    
