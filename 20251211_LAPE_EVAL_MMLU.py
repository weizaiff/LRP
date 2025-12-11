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


lm_eval \
    --model vllm \
    --model_args pretrained=/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --output_path llm_eval/test_mmlu_eval \
    --limit 5 \
    --num_fewshot 5 \
    --batch_size auto \
    --gen_kwargs do_sample=false,max_new_tokens=5 \
    --seed 2025 \
    --log_samples