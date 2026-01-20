import argparse
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
#from vllm import LLM, SamplingParams
from tqdm import tqdm
import transformers

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
# download checkpoint
from accelerate import load_checkpoint_and_dispatch
from tqdm import tqdm
import copy
import torch
from types import MethodType
'''
    convert LRP_based neuron to LAPE format

'''
import re


'''
LAPE mask neuron

'''

def get_mask_neuron_model(model, activation_mask_path, need_lang, is_llama= True):

    # get state_dict 
    state_dict = model.state_dict()
    
    if activation_mask_path:
        activation_masks = torch.load(activation_mask_path)
    else:
        activation_masks = [None]
    
    final_output = []
    if is_llama:
        languages = ["en", "zh", "fr", "es", "vi", "id", "ja"]
    else:
        languages = ["en", "zh", "fr", "es", "vi", "id"]
    
    need_lang = [need_lang]#['en', 'zh', 'vi']
    
    for activation_mask, mask_lang in zip(activation_masks, languages):
    
        if mask_lang not in need_lang:continue
            
        print(f'get mask =====lang:{mask_lang}=====')
    
        
        if activation_mask:
            def factory(mask):
                def llama_forward(self, x):
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    activation.index_fill_(2, mask, 0)
                    x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    return x
                def llama_forward_split(self, x):
                    gate_ = self.gate_proj(x)  # b, l, 2i
                    i = gate_.size(-1)
                    activation = F.silu(gate_)
                    #activation.index_fill_(2, mask, 0)

                    # test
                    print(activation.shape)
                    activation.index_fill_(2, torch.tensor(list(range(activation.shape[1]))), 0 )
                    x = activation * self.up_proj(x)
                    x = self.down_proj(x)
                    return x
    
                def bloom_forward(self, x: torch.Tensor):
                    x, _ = self.dense_h_to_4h(x)
                    x = self.gelu_impl(x)
                    x.index_fill_(2, mask, 0)
                    x, _ = self.dense_4h_to_h(x)
                    return x
    
                if is_llama:
                    return llama_forward
                else:
                    return bloom_forward
    
            for i, layer_mask in enumerate(activation_mask): 
                #print('ilayer:',i, layer_mask)
                if is_llama:
                    #obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
                    
                    # just mask neuron
                    # only masj gate proj
                    
                    #model.model.layers[i].mlp.gate_proj = 
                    state_dict[f'model.layers.{i}.mlp.gate_proj.weight'][layer_mask,:] = 0
                    
                    
                else:
                    #obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
                    assert 1==0
                #obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)
    return model



def get_mask_neuron_model_LRP(model, activation_mask_dict, is_llama =True):

        # get state_dict 
        state_dict = model.state_dict()
        if activation_mask_dict:
            def factory(mask):
                def llama_forward_lrp(self, x):
                    '''
                        mask: {'up_proj':[...], 'gate_proj':[...], 'down_proj':[...]}
                        
                    '''
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    if 'gate_proj' in mask:
                        activation.index_fill_(2, mask['gate_proj'], 0)

                    if 'up_proj' in mask:
                        x = activation * gate_up[:, :, i // 2 :].index_fill_(2, mask['up_proj'], 0)
                    else:
                        x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    if 'down_proj' in mask:
                        x.index_fill_(2, mask['down_proj'], 0)
                    return x
                def llama_forward(self, x):
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    activation.index_fill_(2, mask, 0)
                    x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    return x
                def llama_forward_split(self, x):
                    gate_ = self.gate_proj(x)  # b, l, 2i
                    i = gate_.size(-1)
                    activation = F.silu(gate_)
                    #activation.index_fill_(2, mask, 0)

                    # test
                    print(activation.shape)
                    activation.index_fill_(2, torch.tensor(list(range(activation.shape[1]))), 0 )
                    x = activation * self.up_proj(x)
                    x = self.down_proj(x)
                    return x
    
                def bloom_forward(self, x: torch.Tensor):
                    x, _ = self.dense_h_to_4h(x)
                    x = self.gelu_impl(x)
                    x.index_fill_(2, mask, 0)
                    x, _ = self.dense_4h_to_h(x)
                    return x
    
                if is_llama:
                    return llama_forward_lrp
                else:
                    return bloom_forward
    
            for i, ilayer_mask_dict in enumerate(activation_mask_dict): 
                #print('ilayer:',i, layer_mask)
                #if is_llama:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
                #else:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.transformer.h[i].mlp

                if is_llama:
                    #obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp

                    ilayer_mask_dict_cuda = {}
                    for ikey ,ival in ilayer_mask_dict.items():
                        if len(ival)>0:
                            #ilayer_mask_dict_cuda[ikey] = torch.LongTensor(ival).to('cuda')
                            
                            state_dict[f'model.layers.{i}.mlp.{ikey}.weight'][torch.LongTensor(ival),:] = 0       
                else:
                    #obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
                    assert 1 ==0
                
                
                #obj.forward = MethodType(factory(ilayer_mask_dict_cuda), obj)
        return model

def load_model(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint,trust_remote_code=True)
    print('checkpoint:', checkpoint)
    
    
    if True:
    
        device_map='auto'
        model= AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype= torch.bfloat16,device_map=device_map,weights_only=False ) # for download model weight
    
    else:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
        model = load_checkpoint_and_dispatch(
            model, checkpoint, device_map="auto", dtype=torch.bfloat16#, no_split_module_classes=["GPTJBlock"]
        )
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)#AutoTokenizer.from_pretrained("/home/work/lyftri/projects/model_zoo/compass_sea_13b_s4_merge2HF_org_convert_TP_1_PP_2",  trust_remote_code=True)#AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    return model, tokenizer


import re
from typing import Dict, List, Tuple, Union, Optional

LayerList = List[Dict[str, List[int]]]

def reverse_convert_LAPE_format(
    layer_list: LayerList,
    *,
    scores: Optional[Union[Dict[Tuple[int, str, int], float], List[float]]] = None,
    default_score: float = 1.0,
    as_dict: bool = False,
    prefix: str = "model",              # 生成 "model.layers.{i}..."
    use_layers_plural: bool = True,     # 生成 "layers" 还是 "layer"
    sort_indices: bool = True,
) -> Union[Dict[str, float], List[Tuple[str, float]]]:
    """
    将 convert_LAPE_format 的输出 layer_list 反向还原为 LAPE 格式：
      - list[(key, score)] 或 dict[key]=score

    scores 支持两种：
      1) dict[(layer_idx, proj_type, neuron_idx)] = score
      2) list[score]：按遍历顺序依次填充（不太推荐，容易对不上）
    """
    valid_types = ["up_proj", "gate_proj", "down_proj"]
    layers_token = "layers" if use_layers_plural else "layer"

    # 收集三元组，保证遍历顺序可控
    triples: List[Tuple[int, str, int]] = []
    for layer_idx, d in enumerate(layer_list):
        for t in valid_types:
            idxs = d.get(t, [])
            if sort_indices:
                idxs = sorted(idxs)
            for neuron_idx in idxs:
                triples.append((layer_idx, t, int(neuron_idx)))

    # 给每个三元组分配 score
    def get_score(i: int, tri: Tuple[int, str, int]) -> float:
        if scores is None:
            return float(default_score)
        if isinstance(scores, dict):
            return float(scores.get(tri, default_score))
        # list/sequence
        if i < len(scores):
            return float(scores[i])
        return float(default_score)

    # 生成 key
    out_list: List[Tuple[str, float]] = []
    for i, (layer_idx, proj_type, neuron_idx) in enumerate(triples):
        key = f"{prefix}.{layers_token}.{layer_idx}.mlp.{proj_type}.weight_index_{neuron_idx}"
        out_list.append((key, get_score(i, (layer_idx, proj_type, neuron_idx))))

    if as_dict:
        return {k: v for k, v in out_list}
    return out_list
    
def convert_LAPE_format(neuron_en, config):

    layer_list = [{'up_proj':[], 'gate_proj':[], 'down_proj':[]} for _ in range(config.num_hidden_layers)]

    if isinstance(neuron_en, dict):
        new_tmp = []
        for ineuron, iscore in neuron_en.items():
            new_tmp.append((ineuron, iscore))

        neuron_en = new_tmp
        
    
    for ineuron, iscore in neuron_en:
    
        
        m = re.search(r'layers?\.(\d+).mlp.(.+).weight_index_([0-9]+)', ineuron)
        
        layer_index = int(m.group(1))
        layer_type = m.group(2)
        layer_neuron_index = int(m.group(3))
        assert layer_type in ['up_proj', 'gate_proj', 'down_proj']
    
        layer_list[layer_index][layer_type].append(int(layer_neuron_index))

    return layer_list
def get_mask_neuron_model_vllm_LAPE(model, activation_mask_path, need_lang, is_llama = True):

    if activation_mask_path:
        activation_masks = torch.load(activation_mask_path)
    else:
        activation_masks = [None]
    
    final_output = []
    if is_llama:
        languages = ["en", "zh", "fr", "es", "vi", "id", "ja"]
    else:
        languages = ["en", "zh", "fr", "es", "vi", "id"]
    
    need_lang = [need_lang]#['en', 'zh', 'vi']
    
    for activation_mask, mask_lang in zip(activation_masks, languages):
    
        if mask_lang not in need_lang:continue
            
        print(f'get mask =====lang:{mask_lang}=====')
    
        
        if activation_mask:
            def factory(mask):
                def llama_forward(self, x):
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    activation.index_fill_(2, mask, 0)
                    x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    return x
                def llama_forward_split(self, x):
                    gate_ = self.gate_proj(x)  # b, l, 2i
                    i = gate_.size(-1)
                    activation = F.silu(gate_)
                    #activation.index_fill_(2, mask, 0)

                    # test
                    print(activation.shape)
                    activation.index_fill_(2, torch.tensor(list(range(activation.shape[1]))), 0 )
                    x = activation * self.up_proj(x)
                    x = self.down_proj(x)
                    return x
    
                def bloom_forward(self, x: torch.Tensor):
                    x, _ = self.dense_h_to_4h(x)
                    x = self.gelu_impl(x)
                    x.index_fill_(2, mask, 0)
                    x, _ = self.dense_4h_to_h(x)
                    return x
    
                if is_llama:
                    return llama_forward
                else:
                    return bloom_forward
    
            for i, layer_mask in enumerate(activation_mask): 
                #print('ilayer:',i, layer_mask)
                #if is_llama:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
                #else:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.transformer.h[i].mlp

                if is_llama:
                    obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
                else:
                    obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
                obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)
    return model

def get_mask_neuron_model_vllm_LAPE_org(model, activation_mask, is_llama = True):

    if activation_mask:
            def factory(mask):
                def llama_forward(self, x):
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    activation.index_fill_(2, mask, 0)
                    x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    return x
    
                def bloom_forward(self, x: torch.Tensor):
                    x, _ = self.dense_h_to_4h(x)
                    x = self.gelu_impl(x)
                    x.index_fill_(2, mask, 0)
                    x, _ = self.dense_4h_to_h(x)
                    return x
    
                if is_llama:
                    return llama_forward
                else:
                    return bloom_forward
    
            for i, layer_mask in enumerate(activation_mask):
                if is_llama:
                    obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
                else:
                    obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
                obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)
    return model

def get_mask_neuron_model_vllm_LRP(model, activation_mask_dict, is_llama = True):


        
        if activation_mask_dict:
            def factory(mask):
                def llama_forward_lrp(self, x):
                    '''
                        mask: {'up_proj':[...], 'gate_proj':[...], 'down_proj':[...]}
                        
                    '''
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    if 'gate_proj' in mask:
                        activation.index_fill_(2, mask['gate_proj'], 0)

                    if 'up_proj' in mask:
                        x = activation * gate_up[:, :, i // 2 :].index_fill_(2, mask['up_proj'], 0)
                    else:
                        x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    if 'down_proj' in mask:
                        x.index_fill_(2, mask['down_proj'], 0)
                    return x
                def llama_forward(self, x):
                    gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                    i = gate_up.size(-1)
                    activation = F.silu(gate_up[:, :, : i // 2])
                    activation.index_fill_(2, mask, 0)
                    x = activation * gate_up[:, :, i // 2 :]
                    x, _ = self.down_proj(x)
                    return x
                def llama_forward_split(self, x):
                    gate_ = self.gate_proj(x)  # b, l, 2i
                    i = gate_.size(-1)
                    activation = F.silu(gate_)
                    #activation.index_fill_(2, mask, 0)

                    # test
                    print(activation.shape)
                    activation.index_fill_(2, torch.tensor(list(range(activation.shape[1]))), 0 )
                    x = activation * self.up_proj(x)
                    x = self.down_proj(x)
                    return x
    
                def bloom_forward(self, x: torch.Tensor):
                    x, _ = self.dense_h_to_4h(x)
                    x = self.gelu_impl(x)
                    x.index_fill_(2, mask, 0)
                    x, _ = self.dense_4h_to_h(x)
                    return x
    
                if is_llama:
                    return llama_forward_lrp
                else:
                    return bloom_forward
    
            for i, ilayer_mask_dict in enumerate(activation_mask_dict): 
                #print('ilayer:',i, layer_mask)
                #if is_llama:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
                #else:
                    # latest
                    #obj = model.llm_engine.model_executor.driver_worker.model_runner.model.transformer.h[i].mlp

                if is_llama:
                    obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
                else:
                    obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp

                ilayer_mask_dict_cuda = {}
                #print(ilayer_mask_dict)
                for ikey ,ival in ilayer_mask_dict.items():
                    if len(ival)>0:
                        ilayer_mask_dict_cuda[ikey] = torch.LongTensor(ival).to('cuda')
                
                
                obj.forward = MethodType(factory(ilayer_mask_dict_cuda), obj)
        return model