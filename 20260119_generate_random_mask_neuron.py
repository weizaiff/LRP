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

# 20250120
if False:
    base_save_neuron_dir = '/root/autodl-fs/LRP_kur_res/20260119_newrandom5000samples_cal_llama2_7b_base_random_neuron_time{}_/'
    #org_model_path ='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-chat-hf
    
    org_model_path ='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf/' # base model 
    is_llama=True
    
    model, tokenizer = load_model(org_model_path)
    config = AutoConfig.from_pretrained(org_model_path)
    activation_mask_path_map = {
            'en': '/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_en_zscore.pt',
            'vi': '/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_vi_zscore.pt',
            'zh': '/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_zh_zscore.pt'
        }

# 20260123

base_save_neuron_dir = '/root/autodl-fs/LRP_kur_res/20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh_random_neuron_time{}_/'
#org_model_path ='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-chat-hf

org_model_path ='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf/' # base model 
is_llama=True

model, tokenizer = load_model(org_model_path)
config = AutoConfig.from_pretrained(org_model_path)
activation_mask_path_map = {
        'en': '/root/autodl-fs/LRP_kur_res/20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/th_1_selected_LRP_kur_res_en_zscore.pt',
        'vi': '/root/autodl-fs/LRP_kur_res/20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/th_1_selected_LRP_kur_res_vi_zscore.pt',
        'zh': '/root/autodl-fs/LRP_kur_res/20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/th_1_selected_LRP_kur_res_zh_zscore.pt'
    }



np.random.seed(42)

# random
N_COUNT  = 3 

for i_time in range(N_COUNT):

    
    save_neuron_dir = base_save_neuron_dir.format(i_time)
    os.makedirs(save_neuron_dir, exist_ok=True)
    save_neuron_path = os.path.join(save_neuron_dir,'th_1_selected_LRP_kur_res_random_{}_zscore.pt' )
    
    lang_list = ['en', 'vi', 'zh']
    
    
    max_proj_num_neoron_dict = { 
         'gate_proj':model.state_dict()[f'model.layers.0.mlp.gate_proj.weight'].shape[0],
         'up_proj':model.state_dict()[f'model.layers.0.mlp.up_proj.weight'].shape[0],
         'down_proj':model.state_dict()[f'model.layers.0.mlp.down_proj.weight'].shape[0]
     }
    
    for ilang in lang_list:
    
        i_save_neuron_path = save_neuron_path.format(ilang)
        # LRP version
        LAPE_format_data = convert_LAPE_format(torch.load(activation_mask_path_map[ilang], weights_only=False), config)
    
        new_LAPE_random_data =[{} for _ in range(len(LAPE_format_data))]
        for ilayer in range(len(LAPE_format_data)):
            for iprojname in LAPE_format_data[ilayer].keys():
                rn_neuron_count = len(LAPE_format_data[ilayer][iprojname])
                max_neuron_count = max_proj_num_neoron_dict[iprojname]
        
                # generate
                nums = np.random.choice(max_neuron_count, size=rn_neuron_count, replace=False).tolist()
                new_LAPE_random_data[ilayer][iprojname] = nums
        torch.save(reverse_convert_LAPE_format(new_LAPE_random_data), i_save_neuron_path)
        
                                      
        tmp_model = get_mask_neuron_model_LRP(copy.deepcopy(model), new_LAPE_random_data, is_llama = is_llama)
        print(tmp_model.state_dict()[f'model.layers.0.mlp.gate_proj.weight'].sum())
        #save_model(tmp_model, tokenizer, save_dir.format(ilang))
    
        tmp_model.cpu()
        del tmp_model
        gc.collect()

