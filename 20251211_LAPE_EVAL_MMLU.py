from evalscope import run_task, TaskConfig

# Configure evaluation task
task_cfg_task_1 = TaskConfig(
    model='/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf',
    datasets=['mmlu', 'ceval'],
    limit=5
)

# Start evaluation
run_task(task_cfg_task_1)


evalscope eval \
 --model /root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf \
 --datasets mmlu ceval \
 --model-args revision=master,precision=torch.bf16,device_map=auto \
 --generation-config temperature=1,seed=2025,do_sample=false,max_tokens=4,batch_size=8 \
 --limit 5

#    --limit 5 \
# org model 
lm_eval \
    --model vllm \
    --model_args pretrained=/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --device cuda:0 \
    --output_path /root/autodl-fs/llm_eval/test_mmlu_eval_org_model \
    --num_fewshot 5 \
    --batch_size auto \
    --gen_kwargs do_sample=false,max_new_tokens=5 \
    --seed 2025 \
    --log_samples

# LAPE
lm_eval \
    --model vllm \
    --model_args pretrained=/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_en,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --device cuda:0 \
    --output_path /root/autodl-fs/llm_eval/test_mmlu_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_en \
    --num_fewshot 5 \
    --batch_size auto \
    --gen_kwargs do_sample=false,max_new_tokens=5 \
    --seed 2025 \
    --log_samples

lm_eval \
    --model vllm \
    --model_args pretrained=/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_vi,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --device cuda:0 \
    --output_path /root/autodl-fs/llm_eval/test_mmlu_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_vi \
    --num_fewshot 5 \
    --batch_size auto \
    --gen_kwargs do_sample=false,max_new_tokens=5 \
    --seed 2025 \
    --log_samples


lm_eval \
    --model vllm \
    --model_args pretrained=/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_zh,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --device cuda:0 \
    --output_path /root/autodl-fs/llm_eval/test_mmlu_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_zh \
    --num_fewshot 5 \
    --batch_size auto \
    --gen_kwargs do_sample=false,max_new_tokens=5 \
    --seed 2025 \
    --log_samples

#LRP
lm_eval \
    --model vllm \
    --model_args pretrained=/root/autodl-tmp/LRP_llama2_7b_base_BaseNeuron_BaseMask_lang_en,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --device cuda:0 \
    --output_path /root/autodl-fs/llm_eval/test_mmlu_eval_LRP_llama2_7b_base_BaseNeuron_BaseMask_lang_en \
    --num_fewshot 5 \
    --batch_size auto \
    --gen_kwargs do_sample=false,max_new_tokens=5 \
    --seed 2025 \
    --log_samples








