
export HF_ENDPOINT=https://hf-mirror.com
source /etc/network_turbo
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# model / task / output_dir / num_shot 
EXPERIMENTS=(
    #"('/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf' 'belebele_vie_Latn' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_ChatMask_lang_zh_belebele_vi' '5')"
    #"('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_en' 'belebele_vie_Latn' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_en_belebele_vi' '5')"
    #"('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_vi' 'belebele_vie_Latn' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_vi_belebele_vi' '5')"
    #"('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_zh' 'belebele_vie_Latn' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_zh_belebele_vi' '5')"
    "('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_en' 'ceval-valid' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_en_ceval-valid' '5')"
    #"('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_vi' 'ceval-valid' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_vi_ceval-valid' '5')"
    #"('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_zh' 'ceval-valid' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_zh_ceval-valid' '5')"
    "('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_en' 'mmlu' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_en_mmlu' '5')"
    #"('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_vi' 'mmlu' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_vi_mmlu' '5')"
    "('/root/autodl-tmp/LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_zh' 'mmlu' '/root/autodl-fs/llm_eval/test_eval_LRP_llama2_7b_chat_BaseNeuron_ChatMask_lang_zh_mmlu' '5')"
    
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_zh' 'belebele_vie_Latn' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_zh_belebele_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_en' 'belebele_vie_Latn' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_en_belebele_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_vi' 'belebele_vie_Latn' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_vi_belebele_vi' '5')"
    #"('/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf' 'ceval-valid' '/root/autodl-fs/llm_eval/test_eval_org_model_ceval' '5')"
   # "('/root/autodl-fs/model_zoo/meta-llama/Llama-2-7b-hf' 'global_mmlu_vi' '/root/autodl-fs/llm_eval/test_eval_org_model_global_mmlu_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_zh' 'ceval-valid' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_zh_ceval' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_zh' 'xquad_vi' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_zh_xquad_vi' '0')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_zh' 'global_mmlu_full_vi' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_zh_global_mmlu_full_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_zh' 'mmlu' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_zh_mmlu' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_en' 'ceval-valid' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_en_ceval' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_en' 'xquad_vi' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_en_xquad_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_en' 'global_mmlu_full_vi' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_en_global_mmlu_full_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_en' 'mmlu' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_en_mmlu' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_vi' 'ceval-valid' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_vi_ceval' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_vi' 'xquad_vi' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_vi_xquad_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_vi' 'global_mmlu_full_vi' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_vi_global_mmlu_full_vi' '5')"
    #"('/root/autodl-tmp/llama2_7b_base_BaseNeuron_BaseMask_lang_vi' 'mmlu' '/root/autodl-fs/llm_eval/test_eval_LAPE_llama2_7b_base_BaseNeuron_BaseMask_lang_vi_mmlu' '5')"
)

# mmlu ==5 
# --model_args max_length=4096 \
#        --log_samples
#,max_new_tokens=128
#        --cache_requests true
for EXP in "${EXPERIMENTS[@]}"; do
    # 把字符串转成数组
    eval "tuple=${EXP}"

    PRETRAINED="${tuple[0]}"
    TASKS="${tuple[1]}"
    OUTPUT_PATH="${tuple[2]}"
    NUM_FEWSHOT="${tuple[3]}"

    echo "PRETRAINED = ${PRETRAINED}"
    echo "TASKS = ${TASKS}"
    echo "OUTPUT_PATH = ${OUTPUT_PATH}"
    echo "NUM_FEWSHOT = ${NUM_FEWSHOT}"

    # ====== 开始跑实验 =======
    mkdir -p "${OUTPUT_PATH}"

    lm_eval \
        --model vllm \
        --model_args pretrained=${PRETRAINED},trust_remote_code=True,dtype=bfloat16,max_length=8192 \
        --tasks ${TASKS} \
        --device cuda:0 \
        --output_path ${OUTPUT_PATH} \
        --num_fewshot ${NUM_FEWSHOT} \
        --batch_size auto \
        --gen_kwargs do_sample=false,max_gen_toks=5 \
        --seed 2025 \



done
