import torch
from transformers import AutoTokenizer
from transformers.models.llama import modeling_llama
from transformers import BitsAndBytesConfig

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens

# modify the LLaMA module to compute LRP in the backward pass
monkey_patch(modeling_llama, verbose=True)

from util import save_grad_info, save_per_example_lrp_res
from tqdm import tqdm
import datasets

import os
MAX_LEN=2048

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
            
    for iprompt in prompt:
        model.zero_grad()
        # get input embeddings so that we can compute gradients w.r.t. input embeddings
        input_ids = tokenizer(iprompt, max_length=MAX_LEN, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.retain_grad()
        # inference and get the maximum logit at the last position (we can also explain other tokens)
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
        max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)
        
        
        # get all next tokens' logits
        before_last_token_logits =torch.gather(output_logits[:,:-1,:], dim=-1,index=input_ids[:, 1:].unsqueeze(-1) ).squeeze(-1)
        print("before_last_token_logits shape", before_last_token_logits.shape)
        
        max_logits=(torch.sum(before_last_token_logits)+max_logits)/(before_last_token_logits.reshape(-1).shape[0]+1)
        
        
        # Backward pass (the relevance is initialized with the value of max_logits)
        # This initiates the LRP computation through the network
        max_logits.retain_grad()
        max_logits.backward()
        bar.update(1)
    
        save_per_example_lrp_res(res_map, model)
    
        
    
    save_grad_info(model, os.path.join(output_dir, 'weight_gradients.json'))
    
    torch.save(res_map, os.path.join(output_dir, 'lrp.pt'))
    
    # obtain relevance by computing Input * Gradient
    relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0] # cast to float32 before summation for higher precision
    
    # normalize relevance between [-1, 1] for plotting
    relevance = relevance / relevance.abs().max()
    
    # remove special characters from token strings and plot the heatmap
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    try:
        tokens = clean_tokens(tokens)
        
        #pdf_heatmap(tokens, relevance, path=f'{path.split()[-1]+striing_output}.pdf', backend='xelatex', delete_aux_files=False) # backend='xelatex' supports more characters
        pdf_heatmap(tokens, relevance, path=os.path.join(output_dir, f'res.pdf'), backend='xelatex', delete_aux_files=False) # backend='xelatex' supports more characters
    except:
        print('***')
        print('*cannot generate headmap!!!!!!')
        print('***')





# modify the LLaMA module to compute LRP in the backward pass
#monkey_patch(modeling_llama, verbose=True)

# optional 4bit quantization 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent overflow in gradients
)



def run(iexp_map):
    #for iexp_map in exp_setting:
        
        path = iexp_map['model'] #"/root/autodl-fs/model_zoo/google/gemma-3-1b-it"#'google/gemma-3-4b-it'
        #striing_output='newsample_v2'
        data_path = iexp_map['data_path']
    
        output_dir =os.path.join(iexp_map['prefix_output_dir'],path.split('/')[-1], iexp_map['language'] ) #f'/root/autodl-fs/output_grad/{path.split('/')[-1]}'+'en'
        os.makedirs(output_dir, exist_ok=True)
        model = modeling_llama.LlamaForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16) #, quantization_config=quantization_config)
    
        # optional gradient checkpointing to save memory (2x forward pass)
        model.train()
        model.gradient_checkpointing_enable()
    
        # deactive gradients on parameters to save memory
        for param in model.parameters():
            param.requires_grad = True
    
        tokenizer = AutoTokenizer.from_pretrained(path)
    
        get_lrp_res(data_path, output_dir, model, tokenizer )

exp_setting=[
    
   
    {
        'model':'/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf',
        'language':'zh',
        'prefix_output_dir':'/root/autodl-fs/output_grad/20251210_5000samples_llama2_base',
        'data_path':'/root/autodl-fs/LRP_data/zh_random_5000.jsonl'
    }
]


for iexp_map in exp_setting:
    run(iexp_map)






    