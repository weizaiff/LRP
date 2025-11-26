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






    