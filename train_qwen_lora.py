import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'

class QwenNerSftDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=512):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 使用 Qwen 官方的 ChatML 对话模板构建 Prompt
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": f"文本：{item['input']}"}
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        #tokenize=False 表示返回字符串而不是 token ID 列表。add_generation_prompt=True 会在最后自动加上 <|im_start|>assistant\n，告诉模型"该你回答了"。
        answer_text = item["output"] + self.tokenizer.eos_token
        #标准答案后面加上 EOS token（<|im_end|>），告诉模型"回答结束了"

        # 分别编码 Prompt 和 Answer  ，转成token ID列表
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)

        input_ids = (prompt_ids + answer_ids)[:self.max_len]  #拼在一起，作为模型的完整输入
        # 核心考点：Prompt 部分的 label 全部设为 -100（不计算 Loss），只对 Answer 部分算 Loss！
        labels = ([-100] * len(prompt_ids) + answer_ids)[:self.max_len]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        #这就实现了监督微调：让模型学会根据 prompt 生成正确的 answer，而不是去预测 prompt 本身

def collate_fn(batch, pad_token_id): #batch打包，把同一个batch里所有样本补充到同一长度
    max_len = max(len(x[0]) for x in batch)
    #假设batch = 4
    # batch = [
    #     # x[0] 是 input_ids（张量），x[1] 是 labels（张量）
    #     (tensor([101, 2054, 3421, 999]), tensor([-100, -100, -100, 456])),  # 样本0，长度4
    #     (tensor([101, 2054, 3421, 999, 888, 777]), tensor([-100, -100, -100, -100, 123, 456])),  # 样本1，长度6
    #     (tensor([101, 2054]), tensor([-100, -100])),  # 样本2，长度2
    #     (tensor([101, 2054, 3421, 999, 888]), tensor([-100, -100, -100, -100, 123])),  # 样本3，长度5
    # ]
    batch_input_ids, batch_labels, batch_mask = [], [], []
    for input_ids, labels in batch:
        pad_len = max_len - len(input_ids)
        batch_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        batch_labels.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)]))#torch.cat([...])	把原始数据和 padding 拼接在一起
        batch_mask.append(torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))#batch_mask（attention_mask）	真实 token 标 1，padding 标 0，告诉模型"忽略 padding 部分"
    return torch.stack(batch_input_ids), torch.stack(batch_mask), torch.stack(batch_labels)

def generate_response(model, tokenizer, system_prompt, text):  #推理生成
    """用大模型做自回归生成推理（启用 KV Cache）"""
    model.eval()# 切换到评估模式（关闭 Dropout）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"文本：{text}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)#只负责排版，不负责切词
    inputs = tokenizer(prompt, return_tensors="pt").to(device)#切词，返回张量格式，搬运数据到显卡
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,      # 贪婪解码（每次首选概率最高的token，保证输出稳定）
            use_cache=True,       # 开启你刚学的 KV Cache 加速推理！
            pad_token_id=tokenizer.eos_token_id
        )
    # 只截取新生成的回答部分
    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
    #inputs.input_ids.shape[1] = prompt 的长度
    #outputs[0][prompt长度:] = 从 prompt 结束位置开始截取 = 只取新生成的部分
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

if __name__ == '__main__':
    print(f"📥 正在从 ModelScope 加载 {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # 使用 float16 半精度加载，适配 RTX 3050 4GB 显存
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)

    # 取验证集前 3 条数据，看看微调前原生模型的表现
    with open('data/sft_val.json', 'r', encoding='utf-8') as f:
        val_samples = json.load(f)

    print("\n================ 🔍 微调前 (Zero-Shot) 原生大模型输出测试 ================")
    for i, sample in enumerate(val_samples[:3]):
        pred = generate_response(model, tokenizer, sample['system'], sample['input'])
        print(f"[样本 {i+1}] 原文: {sample['input']}")
        print(f"  🎯 标准答案: {sample['output']}")
        print(f"  🤖 微调前输出: {pred}\n")

    # 挂载 LoRA 旁路矩阵
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,#因果语言模型任务
        r=8,
        lora_alpha=16,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],#对transformer的QKVO层都加LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05   #LoRA矩阵训练时，dropout概率5%，防止过拟合
    )
    model = get_peft_model(model, peft_config)
    print("🚀 已为 Qwen2.5 挂载 LoRA！可训练参数统计：")
    model.print_trainable_parameters()

    train_ds = QwenNerSftDataset('data/sft_train.json', tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)#b是动态参数，pad_token_id是固定的，因为clooate函数只能接收一个参数
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    EPOCHS = 20

    print("\n================ 🔥 开始 LoRA 指令微调 (SFT) ================")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train() #训练模式（Dropout生效）
        total_loss = 0.0
        for input_ids, attn_mask, labels in train_loader:
            input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
            optimizer.zero_grad()#清空上一轮的梯度
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)#前向传播
            loss = outputs.loss
            loss.backward()  #反向传播
            optimizer.step() #根据梯度更新参数
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | SFT Loss: {total_loss/len(train_loader):.4f}")

    print(f"✅ 微调完成！总耗时: {time.time() - start_time:.1f}s")
    model.save_pretrained("qwen_ner_lora_adapter")
    print("💾 LoRA 权重已保存至 'qwen_ner_lora_adapter' 文件夹（仅几 MB 大小！）")

    print("\n================ 🎉 微调后 (LoRA SFT) 大模型输出测试 ================")
    for i, sample in enumerate(val_samples[:3]):
        pred = generate_response(model, tokenizer, sample['system'], sample['input'])
        print(f"[样本 {i+1}] 原文: {sample['input']}")
        print(f"  🎯 标准答案: {sample['output']}")
        print(f"  🚀 微调后输出: {pred}\n")