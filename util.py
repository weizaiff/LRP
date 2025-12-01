import transformers


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
# download checkpoint
from accelerate import load_checkpoint_and_dispatch
from tqdm import tqdm
import datasets
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

#from accelerate import load_checkpoint_and_dispatch

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def calc_ppl(model, tokenizer, textlist, max_len= 2048):
    
    bar = tqdm(total=len(textlist))
    
    ppl_sum = 0
    ppl_count = 0
    device='cuda' # cpu cuda
    for itext in textlist:
        #print('*'*20)
        input_text = itext
        #print('input:', input_text)
        input_dict = tokenizer(input_text, return_tensors="pt").to(device)
        

        input_ids = input_dict['input_ids'][:, :max_len][:,:-1].to(device)
        labels = input_dict['input_ids'][:, :max_len][:,1:].to(device)
            

        with torch.no_grad():
            outputs =model(input_ids, labels=labels)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            ppl = outputs.loss
            #print('ppl:', ppl)

        ppl_sum+= ppl
        ppl_count +=1
        bar.update(1)
    return ppl_sum, ppl_count

def get_test_data(data_pth):
    test = datasets.load_dataset('json', data_files = data_pth)
    test = list(test['train']['text'])

    return test

def get_open_ended_answer(model, tokenizer, textlist, max_new_tokens= 2048):
    
    bar = tqdm(total=len(textlist))
    device='cuda' # cpu cuda
    print('*'*20)
    result = []
    for itext in textlist:
        
        input_text = itext
        #print('input:', input_text)
        input_ids = tokenizer(input_text, return_tensors="pt").to(device)
        #print('input_ids:', input_ids)
    
        output_ids = model.generate(**input_ids, do_sample=False, num_beams=1,repetition_penalty=1.1, max_new_tokens= max_new_tokens) #max_length 
        output_ids = output_ids[0][len(input_ids.input_ids[0]):].tolist() 

        '''
        For qwen3

        '''
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

            
            
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        result.append(content)
        print('itext:',itext)
        print('content:',content)
        
        bar.update(1)
    return result
    
from vllm import LLM, SamplingParams
from tqdm import tqdm

def get_open_ended_answer_vllm(llm, textlist, max_new_tokens=2048):
    

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,       # 等价于 do_sample=False
        top_p=1.0,
        repetition_penalty=1.1,
    )

    results = []
    print("*" * 20)

    # vLLM 支持一次性批量输入
    outputs = llm.generate(textlist, sampling_params)

    for itext, out in zip(textlist, outputs):

        # vLLM 的 output 为 list，每个 item 里有 text
        full_output = out.outputs[0].text

        # 手动切掉输入前缀（vLLM 没有 return_full_text=False）
        # 直接通过字符串替换，不依赖 token 长度
        if full_output.startswith(itext):
            gen_text = full_output[len(itext):]
        else:
            gen_text = full_output

        # 解析 <think> 的 token
        # 你原来用的是 token ID 151668
        # 这里用字符串判断，不需要手动 decode tokens
        think_tag = "</think>"
        idx = gen_text.rfind(think_tag)
        if idx != -1:
            thinking_content = gen_text[:idx + len(think_tag)]
            content = gen_text[idx + len(think_tag):].strip()
        else:
            thinking_content = ""
            content = gen_text.strip()

        results.append(content)

        #print("itext:", itext)
        #print("content:", content)

    return results











    

    






    