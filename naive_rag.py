import json
import torch
import torch.nn.functional as F
from modelscope import AutoTokenizer,AutoModel,AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_corpus(json_path='data/train_label.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        corput = [item['data']['text'] for item in data]
        return corput

#提取文本向量（只用智源开源的轻量模型）
def get_embeddings(texts, model, tokenizer, batch_size = 16):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts, padding = True, truncation = True, max_length = 512, return_tensors = 'pt'
            ).to(device)  #truncation是否对超长文本进行截断，自动截断多余部分   。'pt'返回 PyTorch 的 torch.Tensor
            outputs = model(**encoded)
            cls_repr = outputs.last_hidden_state[:,0] #取出每句话的CLS向量作为整句的代表
            norm_repr = F.normalize(cls_repr, dim = -1) #对每个向量做归一化，模长为1
            all_embeddings.append(norm_repr)
    return torch.cat(all_embeddings, dim = 0)

def retrieve_top_k(query, doc_embeddings, corpus, model, tokenizer, top_k =2):
    #编码提问向量，并且归一化
    query_vec = get_embeddings([query],model, tokenizer)
    #矩阵相乘计算余弦相似度
    scores = torch.matmul(query_vec, doc_embeddings.T).squeeze(0)
    # 取出得分最高的前k个下标和分值
    top_scores, top_indices = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        results.append(
            {
                "doc_id": idx,
                "score": score,
                "text": corpus[idx]
            }
        )
    return results

#大模型开卷问答生成
def generate_rag_answer(query, retrieved_docs, llm_model, llm_tokenizer):
    llm_model.eval()
    # 将检索到的参考资料拼接成上下文
    context_text = "\n\n".join([f"【参考资料 {i + 1}】：{doc['text']}" for i, doc in enumerate(retrieved_docs)])
    system_prompt = "你是一个问答助手。请根据给出的参考资料，简明扼要地回答用户提问。"
    user_content = f"参考资料如下：\n{context_text}\n\n问题：{query}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad(): #因为只要推理，不需要更新参数
        outputs = llm_model.generate(
            **inputs, #把前面编码好的输入（input_ids,attention_mask)等
            max_new_tokens = 150,
            do_sample = False, #关闭解码，使用贪心解码，每次选概率最高的下一个词输出
            use_cache = True,
            pad_token_id = llm_tokenizer.eos_token_id #把pad_token设为结束符，防止生成时出现警告
        )
    gen_ids = outputs[0][inputs.input_ids.shape[1]:]#最终模型生成的回答
    #outputs[0]：取 batch 中第 0 条结果
    #inputs.input_ids.shape[1]：原始输入的长度（prompt 有多长）
    #[原始长度 : ]：从原始长度之后开始切片 → 只保留模型新生成的 token
    return llm_tokenizer.decode(gen_ids, skip_special_tokens=True)#跳过特殊字符


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 启动 Naive RAG 极简系统...")

    # 步骤 A：载入知识库
    corpus = load_corpus('data/train_label.json')
    print(f"✅ 知识库加载完毕，共包含 {len(corpus)} 条文本段落。")

    # 步骤 B：载入 BGE 向量模型并计算知识库的向量矩阵
    EMB_MODEL_ID = 'BAAI/bge-small-zh-v1.5'
    print(f"\n📥 正在加载向量模型 {EMB_MODEL_ID} ...")
    emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_ID)
    emb_model = AutoModel.from_pretrained(EMB_MODEL_ID).to(device)

    print("⚡ 正在对 41 段文本进行批量向量化...")
    doc_embeddings = get_embeddings(corpus, emb_model, emb_tokenizer)
    print(f"✅ 向量矩阵构建完成，形状为: {doc_embeddings.shape} (文档数 × 向量维度)")

    # 步骤 C：载入 Qwen2.5-0.5B-Instruct 生成模型
    LLM_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
    print(f"\n📥 正在加载大语言模型 {LLM_MODEL_ID} ...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)

    # 步骤 D：测试问答
    test_queries = [
        "面对房地产问题，姚洋有什么具体建议？打算筹集多少资金？",
        "美国把对中国输美商品的关税提高到了多少？"
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        print(f"❓ 用户提问: {q}")

        # 1. 语义检索
        retrieved = retrieve_top_k(q, doc_embeddings, corpus, emb_model, emb_tokenizer, top_k=2)
        print("\n🔍 检索出的最相关参考段落：")
        for r in retrieved:
            print(f"  [段落 #{r['doc_id']}] 相似度得分: {r['score']:.4f}")
            print(f"  内容: {r['text'][:90]}...")

        # 2. 开卷生成
        answer = generate_rag_answer(q, retrieved, llm_model, llm_tokenizer)
        print(f"\n🤖 大模型最终回答:\n{answer}")


