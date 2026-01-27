from tqdm import tqdm

from lrp_select_neuron_utils import *


import os
# 20260120
#input_dir_prefix='/root/autodl-fs/output_grad/20260115_newrandom5000samples_quantied_llama2_base/Llama-2-7b-hf/'
#out_dir_prefix='/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/'

# 20260123
input_dir_prefix='/root/autodl-fs/output_grad/20260123_newrandom5000samples_quantied_llama2_base_v3/Llama-2-7b-hf/' 
out_dir_prefix='/root/autodl-fs/LRP_kur_res/20260127_vi_v2_20260123_newrandom5000samples_cal_llama2_7b_base_v3_new_vi_zh/'
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
th_list = [1]
#th_1_selected_LRP_kur_res_zscore
for ith in th_list:
    # get gt th neuron
    save_neuron(get_gt_zscore(data_en_neu_result_skew_kur_mlp_zscore, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_en_zscore.pt")
    save_neuron(get_gt_zscore(data_vi_neu_result_skew_kur_mlp_zscore, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_vi_zscore.pt")
    save_neuron(get_gt_zscore(data_zh_neu_result_skew_kur_mlp_zscore, ith), out_dir_prefix+f"th_{ith}_selected_LRP_kur_res_zh_zscore.pt")
    










                                      