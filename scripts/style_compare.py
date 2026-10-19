# 文风对比验收脚本：底座 vs 底座+LoRA（在 sick-cloud 的 lf 环境运行）
# 用法：CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/style_compare.py
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base = 'Qwen/Qwen3-8B'
adapter = '/home/sick/LLaMA-Factory/saves/sy-qlora-v1'

# 从测试集挑一个写作样本
samples = [json.loads(l) for l in open('/home/sick/LLaMA-Factory/data/dataset/sy_test.json', encoding='utf-8')]
writings = [s for s in samples if '续写' in s['instruction']]
pick = writings[0]

tok = AutoTokenizer.from_pretrained(base)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def gen(model_path, adapter_path=None):
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map='auto', trust_remote_code=True)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    msgs = [{'role': 'user', 'content': pick['instruction'] + '\n\n' + pick['input']}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors='pt').to(model.device)
    out = model.generate(**ids, max_new_tokens=250, do_sample=True, temperature=0.7, top_p=0.9)
    result = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return result

print('===== 参考原文（作者本人写的续写）=====')
print(pick['output'][:300])
print()
print('===== 底座 Qwen3-8B 裸模型 =====')
print(gen(base)[:300])
print()
print('===== 底座 + 你的文风LoRA =====')
print(gen(base, adapter)[:300])
