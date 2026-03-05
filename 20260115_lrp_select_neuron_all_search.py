from tqdm import tqdm

from lrp_select_neuron_utils import *


import os
# 20260120
#input_dir_prefix='/root/autodl-fs/output_grad/20260115_newrandom5000samples_quantied_llama2_base/Llama-2-7b-hf/'
#out_dir_prefix='/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/'

# 20260123
#input_dir_prefix='/root/autodl-fs/output_grad/20260128_newrandom1000samples_llama2_base/Llama-2-7b-hf/' 
#out_dir_prefix='/root/autodl-fs/LRP_kur_res/20260128_newrandom1000samples_llama2_base/'


# 20260305-just kursis val filter
input_dir_prefix='/root/autodl-fs/output_grad/2026304_arr_paper_1000samples_llama2_base/Llama-2-7b-hf/' 
out_dir_prefix='/root/autodl-fs/LRP_kur_res/20260305_arr_paper_llama2_7b_base_kursis_val_filter/'

os.makedirs(out_dir_prefix+'/', exist_ok=True)
print('load lrp.pt...')
data    = torch.load(os.path.join(input_dir_prefix, 'en/lrp.pt'), map_location=torch.device('cpu'))
data_vi = torch.load(os.path.join(input_dir_prefix, 'vi/lrp.pt'), map_location=torch.device('cpu'))
data_zh = torch.load(os.path.join(input_dir_prefix, 'zh/lrp.pt'), map_location=torch.device('cpu'))
    

print('extract_mlp_data_pre_calc...')
data_mlp = extract_mlp_data_pre_calc(data)
data_vi_mlp = extract_mlp_data_pre_calc(data_vi)
data_zh_mlp = extract_mlp_data_pre_calc(data_zh)


print('calc_skew_kurtosis...')
# calc kurtosis
data_neu_result_skew_kur = calc_skew_kurtosis(data_mlp)
data_vi_neu_result_skew_kur = calc_skew_kurtosis(data_vi_mlp)
data_zh_neu_result_skew_kur = calc_skew_kurtosis(data_zh_mlp)

data_en_neu_result_skew_kur_mlp = extract_mlp_data(data_neu_result_skew_kur)
data_vi_neu_result_skew_kur_mlp = extract_mlp_data(data_vi_neu_result_skew_kur)
data_zh_neu_result_skew_kur_mlp = extract_mlp_data(data_zh_neu_result_skew_kur)

print('save_neuron...')
save_neuron(data_en_neu_result_skew_kur_mlp, out_dir_prefix+"all_mlp_LRP_kur_res_en.pt")
save_neuron(data_vi_neu_result_skew_kur_mlp, out_dir_prefix+"all_mlp_LRP_kur_res_vi.pt")
save_neuron(data_zh_neu_result_skew_kur_mlp, out_dir_prefix+"all_mlp_LRP_kur_res_zh.pt")


data_en_neu_result_skew_kur_mlp_zscore=compute_zscore(data_en_neu_result_skew_kur_mlp)
data_vi_neu_result_skew_kur_mlp_zscore=compute_zscore(data_vi_neu_result_skew_kur_mlp)
data_zh_neu_result_skew_kur_mlp_zscore=compute_zscore(data_zh_neu_result_skew_kur_mlp)


# save _zscore

save_neuron(data_en_neu_result_skew_kur_mlp_zscore, out_dir_prefix+"all_mlp_LRP_kur_res_en_zscore.pt")
save_neuron(data_vi_neu_result_skew_kur_mlp_zscore, out_dir_prefix+"all_mlp_LRP_kur_res_vi_zscore.pt")
save_neuron(data_zh_neu_result_skew_kur_mlp_zscore, out_dir_prefix+"all_mlp_LRP_kur_res_zh_zscore.pt")

print('save target neuron...')

if False:
    ### zscore_ selected
    th_list = [0.2, 0.5,0.7, 1,2, 3, 5, 10]
    #th_1_selected_LRP_kur_res_zscore
    for ith in th_list:
        # get gt th neuron
        save_neuron(get_gt_zscore(data_en_neu_result_skew_kur_mlp_zscore, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_en_zscore.pt")
        save_neuron(get_gt_zscore(data_vi_neu_result_skew_kur_mlp_zscore, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_vi_zscore.pt")
        save_neuron(get_gt_zscore(data_zh_neu_result_skew_kur_mlp_zscore, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_zh_zscore.pt")
        
    ### zscore_margein_selected
    
    all_D={
        'en':data_en_neu_result_skew_kur_mlp_zscore,
        'vi':data_vi_neu_result_skew_kur_mlp_zscore,
        'zh':data_zh_neu_result_skew_kur_mlp_zscore
    }
    margin_threshold_list = [0.1, 0.25, 0.5, 1, 2, 3, 5, 10]
    for imar_th in margin_threshold_list:
        print('========')
        print(imar_th)
    
        selected_all_D = select_language_specific_neurons(all_D, abs_threshold=0, margin_threshold=imar_th)
        
        # get gt th neuron
        save_neuron(selected_all_D['en'], out_dir_prefix+f"th_{imar_th}_selected_LRP_kur_res_en_zscore_margin_selected.pt")
        save_neuron(selected_all_D['vi'], out_dir_prefix+f"th_{imar_th}_selected_LRP_kur_res_vi_zscore_margin_selected.pt")
        save_neuron(selected_all_D['zh'], out_dir_prefix+f"th_{imar_th}_selected_LRP_kur_res_zh_zscore_margin_selected.pt")
            
    
    
    ## soft preference
    '''
    zscore
    '''
    
    all_D={
        'en':data_en_neu_result_skew_kur_mlp_zscore,
        'vi':data_vi_neu_result_skew_kur_mlp_zscore,
        'zh':data_zh_neu_result_skew_kur_mlp_zscore
    }
    th_list = [0.1, 0.25, 0.5, 0.9 ,0.95, 0.99]
    for ith in th_list:
        print('==========')
        print('th_list:', ith)
        selected = select_by_soft_preference(compute_soft_preference(all_D), threshold = ith)
    
        
        # get gt th neuron
        save_neuron(selected['en'], out_dir_prefix+f"soft_preference_th_{ith}_selected_LRP_kur_res_en_zscore_margin_selected.pt")
        save_neuron(selected['vi'], out_dir_prefix+f"soft_preference_th_{ith}_selected_LRP_kur_res_vi_zscore_margin_selected.pt")
        save_neuron(selected['zh'], out_dir_prefix+f"soft_preference_th_{ith}_selected_LRP_kur_res_zh_zscore_margin_selected.pt")

'''
# 20260305 not z score because of kursis‘s meanings
'''
#th_list = [3,4,5,6,7,8,9,10,11,12]

th_list = [5, 8, 10, 15, 20, 25, 30, 35, 40] #list(range(13, 30))
for ith in th_list:
    # get gt th neuron
    save_neuron(get_gt_zscore(data_en_neu_result_skew_kur_mlp, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_en.pt")
    save_neuron(get_gt_zscore(data_vi_neu_result_skew_kur_mlp, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_vi.pt")
    save_neuron(get_gt_zscore(data_zh_neu_result_skew_kur_mlp, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_zh.pt")


    

















                                      