import torch
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama
from transformers import BitsAndBytesConfig

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens

# modify the LLaMA module to compute LRP in the backward pass
#monkey_patch(modeling_llama, verbose=True)

from util import save_grad_info, save_per_example_lrp_res
from tqdm import tqdm
import datasets

import os
MAX_LEN=2048

is_llama = True

print('=='*20)
print('is_llama:', is_llama)
print('=='*20)

def get_lrp_res(data_path, output_dir, model, tokenizer ):
    prompt = """Context: The Eiffel Tower, built in 1889, was the world's tallest man-made structure for 41 years. It is 330 meters tall and has three levels for visitors.
    Question: How tall is the Eiffel Tower?
    Answer: According to the text, the Eiffel Tower is"""
    
    prompt = """Context: The Eiffel Tower, built in 1889, was the world's tallest man-made structure for 41 years. It is 330 meters tall and has three levels for visitors.
    
    Question: When was the Eiffel Tower built?
    Answer: According to the text, the Eiffel Tower was built in"""
    
    prompt = """Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
    Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""
    
    prompt = datasets.load_dataset('json', data_files=data_path)['train']['text'] #[prompt]*2
    
    bar = tqdm(total=len(prompt))
    res_map = {}
    # get all parameter name
    # 遍历模型的所有命名参数
    for name, param in model.named_parameters():
        # 检查参数是否有梯度
        res_map[name] = []

    

    all_activation = []
    for iprompt in prompt:
        '''
            save for per prompt 


        '''
        # actuall is all activation
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
        def factory(idx):
            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
                i = gate_up.size(-1)
                gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
                activation = gate_up[:, :, : i // 2].float() # b, l, i
                over_zero[idx, :] += (activation).sum(dim=(0,1))
                x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
                x, _ = self.down_proj(x)
                return x
        
            def bloom_forward(self, x: torch.Tensor):
                x, _ = self.dense_h_to_4h(x)
                x = self.gelu_impl(x)
                activation = x.float()
                over_zero[idx, :] += (activation).sum(dim=(0,1))
                x, _ = self.dense_4h_to_h(x)
                return x
        
            if is_llama:
                return llama_forward
            else:
                return bloom_forward
        
        for i in range(num_layers):
            if is_llama:
                obj = model.model.layers[i].mlp
            else:
                obj = model.model.transformer.h[i].mlp
            obj.forward = MethodType(factory(i), obj)





        
        model.zero_grad()
        # get input embeddings so that we can compute gradients w.r.t. input embeddings
        input_ids = tokenizer(iprompt, max_length=MAX_LEN, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        
        print('=='*20)
        print('input_ids shape:', input_ids.shape)
        print('input_ids shape must not the same as MAX_LEN!!!!!')
        print('=='*20)

        
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.retain_grad()
        # inference and get the maximum logit at the last position (we can also explain other tokens)
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits



        # append
        all_activation.append(over_zero)
        


        
        
        
        
        
        bar.update(1)

    # save all result
    torch.save(res_map, os.path.join(output_dir, 'activation.pt'))
    
        
    
    
    

def run(iexp_map):
    #for iexp_map in exp_setting:
        
        path = iexp_map['model'] #"/root/autodl-fs/model_zoo/google/gemma-3-1b-it"#'google/gemma-3-4b-it'
        #striing_output='newsample_v2'
        data_path = iexp_map['data_path']
    
        output_dir =os.path.join(iexp_map['prefix_output_dir'],path.split('/')[-1], iexp_map['language'] ) #f'/root/autodl-fs/output_grad/{path.split('/')[-1]}'+'en'
        os.makedirs(output_dir, exist_ok=True)
        
        model = modeling_llama.LlamaForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16) #, quantization_config=quantization_config)
        #use quantized model
        #model = modeling_llama.LlamaForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16, quantization_config=quantization_config)
        
        # optional gradient checkpointing to save memory (2x forward pass)
        model.train()
        #model.gradient_checkpointing_enable()
        model.gradient_checkpointing_disable()
    
        # deactive gradients on parameters to save memory
        for param in model.parameters():
            param.requires_grad = True
    
        tokenizer = AutoTokenizer.from_pretrained(path)
    
        get_lrp_res(data_path, output_dir, model, tokenizer )

# before
exp_setting_org=[
    
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-chat-hf',
        'language':'vi',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20251204_5000samples_llama2',
        'data_path':'/root/autodl-fs/LRP_data/vi_random_5000.jsonl'
    },
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-chat-hf',
        'language':'zh',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20251204_5000samples_llama2',
        'data_path':'/root/autodl-fs/LRP_data/zh_random_5000.jsonl'
    },
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-chat-hf',
        'language':'en',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20251204_5000samples_llama2',
        'data_path':'/root/autodl-fs/LRP_data/en_random_5000.jsonl'
    }
]
exp_setting=[
    
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf',
        'language':'en',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20251210_5000samples_llama2_base',
        'data_path':'/root/autodl-fs/LRP_data/en_random_5000.jsonl'
    }
]

# 20260115 llama2-7b-hf
exp_setting=[
    
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf',
        'language':'vi',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20260115_newrandom5000samples_quantied_llama2_base',
        'data_path':'/root/autodl-fs/LRP_data/all_data_sampling_ver/c4_vi_uniform_5k.jsonl'
    }
]

# 20260123 llama2-7b-hf new vi + 20260127 c4===> wanjuan
exp_setting=[
    
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf',
        'language':'vi',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20260123_newrandom5000samples_quantied_llama2_base_v3',
        'data_path':'/root/autodl-fs/LRP_data/all_data_sampling_ver3/wanjuan_vi_uniform_5k.jsonl'
    }
]

# 20260128 new 1k 
exp_setting=[
    
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf',
        'language':'vi',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20260128_newrandom1000samples_llama2_base',
        'data_path':'/root/autodl-fs/LRP_data/all_data_sampling_ver3/1000_samples/vi_uniform_1k.jsonl'
    }
]
# 20260304 ARR paper 1k  reproduce

exp_setting=[
    
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf',
        'language':'vi',
        'prefix_output_dir':'/root/autodl-fs/output_grad/2026304_arr_paper_1000samples_llama2_base',
        'data_path':'/root/autodl-fs/LRP_data/vi_random_1000.jsonl'
    }
]


for iexp_map in exp_setting:
    run(iexp_map)






    