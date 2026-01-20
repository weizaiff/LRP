"""
Global-uniform random baseline
- budget matched to CRANE
- neuron identity uniformly sampled over all layers / projections
"""

import gc
import os
import copy
import torch
import numpy as np

from open_ended_utils import *
from transformers import AutoConfig


base_save_neuron_dir = '/root/autodl-fs/LRP_kur_res/20260120_global_random5000samples_llama2_7b_base_time{}_/'
org_model_path = '/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf/'
is_llama = True

model, tokenizer = load_model(org_model_path)
config = AutoConfig.from_pretrained(org_model_path)

activation_mask_path_map = {
    'en': '/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_en_zscore.pt',
    'vi': '/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_vi_zscore.pt',
    'zh': '/root/autodl-fs/LRP_kur_res/20260115_newrandom5000samples_cal_llama2_7b_base/th_1_selected_LRP_kur_res_zh_zscore.pt'
}

np.random.seed(42)

# repeat random trials
N_COUNT = 3

# neuron count per projection (assume identical across layers)
proj_neuron_num = {
    'gate_proj': model.state_dict()['model.layers.0.mlp.gate_proj.weight'].shape[0],
    'up_proj': model.state_dict()['model.layers.0.mlp.up_proj.weight'].shape[0],
    'down_proj': model.state_dict()['model.layers.0.mlp.down_proj.weight'].shape[0],
}

num_layers = config.num_hidden_layers
proj_names = ['gate_proj', 'up_proj', 'down_proj']


for i_time in range(N_COUNT):

    save_neuron_dir = base_save_neuron_dir.format(i_time)
    os.makedirs(save_neuron_dir, exist_ok=True)
    save_neuron_path = os.path.join(
        save_neuron_dir,
        'th_1_selected_LRP_kur_res_global_random_{}_zscore.pt'
    )

    for ilang in ['en', 'vi', 'zh']:

        # ===== 1. load CRANE mask to compute total budget =====
        LAPE_data = convert_LAPE_format(
            torch.load(activation_mask_path_map[ilang], weights_only=False),
            config
        )

        total_budget = 0
        for l in range(num_layers):
            for p in proj_names:
                total_budget += len(LAPE_data[l][p])

        print(f'[Global-Random] {ilang}: total budget = {total_budget}')

        # ===== 2. build global neuron universe =====
        universe = []
        for l in range(num_layers):
            for p in proj_names:
                for n in range(proj_neuron_num[p]):
                    universe.append((l, p, n))

        universe_size = len(universe)
        print('total_budget:', total_budget)
        print('universe_size:', universe_size)
        assert total_budget <= universe_size

        # ===== 3. global uniform sampling =====
        sampled_idx = np.random.choice(
            universe_size,
            size=total_budget,
            replace=False
        )

        # ===== 4. map back to layer-wise structure =====
        new_LAPE_random_data = [
            {p: [] for p in proj_names}
            for _ in range(num_layers)
        ]

        for idx in sampled_idx:
            l, p, n = universe[idx]
            new_LAPE_random_data[l][p].append(n)

        # ===== 5. save mask =====
        i_save_neuron_path = save_neuron_path.format(ilang)
        torch.save(
            reverse_convert_LAPE_format(new_LAPE_random_data),
            i_save_neuron_path
        )

        # ===== 6. sanity check (optional) =====
        tmp_model = get_mask_neuron_model_LRP(
            copy.deepcopy(model),
            new_LAPE_random_data,
            is_llama=is_llama
        )
        print(tmp_model.state_dict()['model.layers.0.mlp.gate_proj.weight'].sum())

        tmp_model.cpu()
        del tmp_model
        gc.collect()
