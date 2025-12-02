import json
import asyncio
import csv
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import os
import aiofiles
from tenacity import retry, stop_after_attempt, wait_random_exponential
from datetime import datetime
from pydantic import BaseModel

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "f06fdc66815a7417b3df4a1fe6bd588fc8e10d1a4bc8dae16956cfcdd071b5f5"
client = AsyncOpenAI(
    base_url='https://compass.llm.shopee.io/compass-api/v1',
)

# Define input and output files
# input_file = '/home/work/llm_alignment_v3/qingtao.yu/new/dataset/esci/modified/task2_train_choice_sample.json'
# output_file = '/home/work/llm_alignment_v3/qingtao.yu/new/dataset/esci/modified/task2_train_choice_sample_cot.json'
# input_file = '/home/work/llm_alignment_v3/qingtao.yu/new/dataset/esci/modified/task1_train_rank_sample.json'
# output_file = '/home/work/llm_alignment_v3/qingtao.yu/new/dataset/esci/modified/task1_train_rank_sample_cot.json'
#input_file = '/home/work/llm_alignment_v3/qingtao.yu/new/dataset/AmazonM2/modified/AmazonM2_task1_multi_choice_sample.json'
#output_file = '/home/work/llm_alignment_v3/qingtao.yu/new/dataset/AmazonM2/modified/AmazonM2_task1_multi_choice_sample_gen_cot.jsonl'

input_file ='/home/work/lyf-sg-duo/intent/data/20251119_search_think_data/20251201_forGPTJudge.json' #'/home/work/lyf-sg-duo/multilingual-evaluation-pipeline/20251021_query.json'

output_file = f'/home/work/lyf-sg-duo/intent/data/20251119_search_think_data/20251201_forGPTJudge.jsonoutput_gpt5.1.json'

input_column = "input"
output_column='output'

# Set up semaphore for rate limiting
semaphore = asyncio.Semaphore(100)

# 每处理多少条数据保存一次
SAVE_FREQUENCY = 100

class QA(BaseModel):
    reasoning_process: str
    answer: str

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
async def generate_classification_response(user_input):
    async with semaphore:
        try:
            response = await client.beta.chat.completions.parse(
                #model="gpt-4o-2024-08-06",
                #model="gpt-4.1-2025-04-14",
                model="gpt-5.1",
                messages=[
                    {"role": "user", "content": user_input},
                ],
                temperature=0.1,
                #response_format=QA
            )
            response_text = response.choices[0].message.content.strip()
            # print("response_text:", response_text)
            #return json.loads(response_text)
            return response_text
        except Exception as e:
            print(f"Error generating response: {e}")
            raise


async def read_jsonl_file(file_path, num_records=None):
    records = []
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            i = 0
            async for line in f:
                try:
                    record = json.loads(line)
                    # print("record:", record)
                    if input_column in record.keys():
                        records.append(record)
                        i += 1
                        if num_records and i >= num_records:
                            break
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
    return records

async def process_record(record):
    try:
        # ext_info = record.get(input_column, {})
        def remove_last_sentence(text):
            # 使用句点分割文本
            parts = text.split('.')
            
            # 过滤掉空字符串部分
            parts = [part for part in parts if part.strip()]
            
            # 如果分割后长度小于等于1，直接返回原文本
            if len(parts) <= 1:
                return text
            
            # 去掉最后一个非空部分，重新连接并添加句点
            result = '.'.join(parts[:-1]) + '.'
            
            return result
        
        #instruction = remove_last_sentence(record.get(input_column, ''))
        instruction = record.get(input_column, '')
        
        # input_text = ext_info.get('input', '')
        # options = ext_info.get('options', '')
        # instruction = '' if instruction is None else instruction
        # instruction = instruction + "Thinking carefully and finally output the answer: yes or no."
        # input_text = '' if input_text is None else input_text
        # options = '' if options is None else options
        
        # 组合字段，使用换行符分隔
        # user_input = (f"{instruction}\n{input_text}\n{options}").strip()
        
        # 生成响应
        output = await generate_classification_response(instruction)
        record[output_column] = output
        # 返回格式化数据
        return record
    except Exception as e:
        print(f"Error processing record: {e}")
        return None

async def write_batch_to_file(batch, output_file, append=False):
    """将一批记录写入文件"""
    mode = 'a' if append else 'w'
    valid_count = 0
    
    async with aiofiles.open(output_file, mode, encoding='utf-8') as out:
        for item in batch:
            if item is None:
                continue
                
            try:
                # 尝试序列化以验证JSON格式
                json_str = json.dumps(item, ensure_ascii=False)
                
                # 如果到达这里，说明是有效的JSON
                await out.write(json_str + '\n')
                valid_count += 1
                
            except (TypeError, json.JSONDecodeError) as e:
                print(f"Invalid JSON format: {e}")
    
    return valid_count

async def process_file():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 从JSONL读取记录
    records = await read_jsonl_file(input_file)
    print(f"Read {len(records)} records from JSONL")
    
    # 创建一个空文件（如果不存在）
    if not os.path.exists(output_file):
        async with aiofiles.open(output_file, 'w', encoding='utf-8'):
            pass
    
    total_valid_count = 0
    total_invalid_count = 0
    
    # 分批处理记录
    for i in range(0, len(records), SAVE_FREQUENCY):
        batch_records = records[i:i+SAVE_FREQUENCY]
        batch_tasks = [process_record(record) for record in batch_records]
        
        # 处理当前批次
        print(f"Processing batch {i//SAVE_FREQUENCY + 1} ({i+1}-{min(i+SAVE_FREQUENCY, len(records))})")
        processed_batch = await tqdm.gather(*batch_tasks, desc=f"Batch {i//SAVE_FREQUENCY + 1}")
        
        # 统计有效和无效记录
        valid_batch = [item for item in processed_batch if item is not None]
        invalid_batch = len(processed_batch) - len(valid_batch)
        
        # 写入当前批次到文件
        append_mode = i > 0  # 第一批创建新文件，后续批次追加
        valid_written = await write_batch_to_file(valid_batch, output_file, append=append_mode)
        
        # 更新统计信息
        total_valid_count += valid_written
        total_invalid_count += invalid_batch
        
        print(f"Batch {i//SAVE_FREQUENCY + 1} complete: {valid_written} valid records written, {invalid_batch} invalid records skipped")
        print(f"Progress: {i + len(batch_records)}/{len(records)} records processed")
    
    print("\nProcessing summary:")
    print(f"Total records processed: {len(records)}")
    print(f"Total valid records written: {total_valid_count}")
    print(f"Total invalid records skipped: {total_invalid_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(process_file())