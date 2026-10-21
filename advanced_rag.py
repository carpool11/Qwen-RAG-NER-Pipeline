import json
import torch
import torch.nn.functional as F
import jieba
from rank_bm25 import BM25Okapi
from modelscope import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_corpus(json_path='data/train_label.json'):#只把文本提出来
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['data']['text'] for item in data]

def get_embeddings(texts, model, tokenizer, batch_size=16):#把一批文本转换成归一化后的向量矩阵，形状为 [文本数量, 向量维度]。
    model.eval()#转到评估模式，关闭dropout
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size): #每次取 batch_size 条文本进行处理，避免一次性把所有文本送进 GPU 导致显存溢出。
            encoded = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            outputs = model(**encoded)
            cls_repr = outputs.last_hidden_state[:, 0]#[:, 0] 表示取每个样本的第 0 个位置（即 [CLS] token）。只取[cls]作为整句的表示
            all_embeddings.append(F.normalize(cls_repr, p=2, dim=-1)) #对每个向量做 L2 归一化，模长变为 1。
    return torch.cat(all_embeddings, dim=0) #拼接所有batch，在第0维

def retrieve_hybrid_and_rerank(query, doc_embeddings, corpus, bm25, emb_model, emb_tok, rerank_model, rerank_tok, top_k=1):
    #BM25稀疏检索
    tokenized_query = jieba.lcut(query) #jieba是专门用来给中文分词的工具
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top5_idx = torch.tensor(bm25_scores).topk(5).indices.tolist()

    #embedding稠密检索
    query_vec = get_embeddings([query],emb_model, emb_tok)
    vec_scores = torch.matmul(query_vec, doc_embeddings.T).squeeze(0)
    vec_top5_idx = vec_scores.topk(5).indices.tolist()

    #合并去重
    candidate_indices = list(set(bm25_top5_idx + vec_top5_idx))

    #reranker重排序
    #召回：bm25+向量检索 ，速度快
    #重排序：cross——encoder，速度慢，但精度高
    #当前主流的RAG系统标准的两阶段，先快速召回，再精细重拍
    rerank_model.eval()
    rerank_results = []
    with torch.no_grad():
        for idx in candidate_indices:
            doc_text = corpus[idx]
            inputs = rerank_tok(query, doc_text, return_tensors='pt', truncation = True, max_length = 512).to(device)#把「查询 + 文档」拼成一对，送给重排序模型。
            score = rerank_model(**inputs).logits.item()#logits模型输出的原始分数 .item把张量转成python的普通浮点数
            rerank_results.append({"doc_id":idx,"score":score, "text":doc_text}) #把当前的结果存起来，doc_idx是文档在原始语料中的编号

        rerank_results.sort(key = lambda x: x['score'], reverse = True)
        return rerank_results[:top_k]


def generate_rag_answer(query, retrieved_docs, llm_model, llm_tokenizer):
    llm_model.eval()
    context_text = "\n".join([f"资料{i + 1}：{doc['text']}" for i, doc in enumerate(retrieved_docs)])

    system_prompt = "你是一个严谨的问答助手。请严格基于提供的参考资料字面事实回答问题，提取对应的人名或事实，禁止凭主观印象推测。"
    user_content = f"【参考资料】\n{context_text}\n\n【用户提问】\n{query}\n\n请提取与提问直接对应的实体或内容直接作答："

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=150, do_sample=False, use_cache=True,
                                     pad_token_id=llm_tokenizer.eos_token_id)
    return llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 启动工业级双路召回+重排 RAG 系统...")

    corpus = load_corpus('data/train_label.json')

    # 1. 初始化 BM25 索引
    print("✂️ 正在对知识库进行结巴分词与 BM25 索引构建...")
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # 2. 载入 BGE 向量模型
    print("📥 正在加载 BGE 向量模型...")
    emb_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
    emb_model = AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5').to(device)
    doc_embeddings = get_embeddings(corpus, emb_model, emb_tokenizer)

    # 3. 载入 BGE Reranker 模型 (使用 float16 节省显存)
    print("🧠 正在加载 BGE 重排模型 (Reranker)...")
    rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
    rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base',
                                                                      torch_dtype=torch.float16).to(device)

    # 4. 载入 Qwen 生成模型
    print("🤖 正在加载 Qwen2.5-0.5B-Instruct...")
    llm_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    llm_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.float16).to(device)

    # ================= 杀手级测试用例 =================
    # "柯锐"是典型的极低频专有名词，纯向量检索很容易漏掉，这是检验 BM25 的最佳试金石！
    test_queries = ["新京报评论员是谁？", "面对房地产问题，筹集了多少资金组建国家队？"]

    for q in test_queries:
        print("\n" + "=" * 60)
        print(f"❓ 用户提问: {q}")

        retrieved = retrieve_hybrid_and_rerank(
            q, doc_embeddings, corpus, bm25, emb_model, emb_tokenizer, rerank_model, rerank_tokenizer, top_k=1
        )

        print("\n🔍 交叉重排后的最相关参考段落：")
        for r in retrieved:
            print(f"  [段落 #{r['doc_id']}] 交叉打分: {r['score']:.4f} | 内容: {r['text'][:60]}...")

        answer = generate_rag_answer(q, retrieved, llm_model, llm_tokenizer)
        print(f"\n🤖 大模型最终回答:\n{answer}")