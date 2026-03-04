"""
Fully Random baseline:
- mask 数量 B ~ Uniform([0, total_neurons])
- mask 位置 global-uniform
- 与 CRANE 完全无关
"""

import gc
import os
import copy
import numpy as np
import torch
from transformers import AutoConfig

from open_ended_utils import *

if False:
    base_save_neuron_dir = '/root/autodl-fs/LRP_kur_res/20260121_fully_random_time{}_/'
    org_model_path = '/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf/'

# 20260304 arr paper reproduce
base_save_neuron_dir = '/root/autodl-fs/LRP_kur_res/20260304_arr_paper_llama2_7b_base_fullly_random_neuron_time{}_/'
org_model_path = '/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf/'

is_llama = True

model, tokenizer = load_model(org_model_path)
config = AutoConfig.from_pretrained(org_model_path)

proj_names = ['gate_proj', 'up_proj', 'down_proj']
num_layers = config.num_hidden_layers

# neuron 数（保持与你 mask 语义一致）
proj_neuron_num = {
    'gate_proj': model.state_dict()['model.layers.0.mlp.gate_proj.weight'].shape[0],
    'up_proj': model.state_dict()['model.layers.0.mlp.up_proj.weight'].shape[0],
    'down_proj': model.state_dict()['model.layers.0.mlp.down_proj.weight'].shape[0],
}

per_layer_total = sum(proj_neuron_num[p] for p in proj_names)
total_universe = num_layers * per_layer_total

def flat_to_lpn(flat_idx: int):
    layer = flat_idx // per_layer_total
    r = flat_idx % per_layer_total
    acc = 0
    for p in proj_names:
        n = proj_neuron_num[p]
        if r < acc + n:
            return int(layer), p, int(r - acc)
        acc += n
    raise RuntimeError("flat_to_lpn failed")

N_COUNT = 3
lang_list = ['en', 'vi', 'zh']

for i_time in range(N_COUNT):
    rng = np.random.default_rng(42 + i_time)

    save_neuron_dir = base_save_neuron_dir.format(i_time)
    os.makedirs(save_neuron_dir, exist_ok=True)
    save_neuron_path = os.path.join(
        save_neuron_dir,
        'fully_random_mask_{}_zscore.pt'
    )

    for ilang in lang_list:
        i_save_neuron_path = save_neuron_path.format(ilang)

        # ===== 1) 完全随机的预算 =====
        B = int(rng.integers(low=0, high=total_universe + 1))
        print(f"[FullyRandom][time={i_time}][{ilang}] B = {B}")

        # ===== 2) global-uniform 抽样 =====
        sampled_flat = rng.choice(total_universe, size=B, replace=False)

        new_LAPE_random_data = [{p: [] for p in proj_names} for _ in range(num_layers)]

        for fi in sampled_flat:
            l, p, nid = flat_to_lpn(int(fi))
            new_LAPE_random_data[l][p].append(int(nid))

        for l in range(num_layers):
            for p in proj_names:
                new_LAPE_random_data[l][p].sort()

        torch.save(
            reverse_convert_LAPE_format(new_LAPE_random_data),
            i_save_neuron_path
        )

        # sanity check
        tmp_model = get_mask_neuron_model_LRP(
            copy.deepcopy(model),
            new_LAPE_random_data,
            is_llama=is_llama
        )
        print(tmp_model.state_dict()['model.layers.0.mlp.gate_proj.weight'].sum().item())

        tmp_model.cpu()
        del tmp_model
        gc.collect()
