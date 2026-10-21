import json
import time
import torch
import torch.nn.functional as F
import jieba
from rank_bm25 import BM25Okapi
from modelscope import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= 1. 载入语料库 =================
def load_corpus(json_path='data/train_label.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['data']['text'] for item in data]


# ================= 2. 校准后的 12 道黄金评测题 (Golden Dataset) =================
EVAL_DATASET = [
    # 基础事实与数字题
    {"query": "2025年4月10日，美国政府将对中国输美商品征收的“对等关税”税率提高到了多少？", "gold_id": 2,
     "keywords": ["125%"]},
    {"query": "我国对美国出口产生的增加值大约是多少万亿元？中国对美贸易出口可能掉多少？", "gold_id": 24,
     "keywords": ["4万亿"]},
    {"query": "4月8日，货轮停靠在哪个港口集装箱码头进行装卸作业？", "gold_id": 26, "keywords": ["天津港"]},
    {"query": "出口美国占我国总出口的比例满打满算大概是多少？", "gold_id": 27, "keywords": ["20%"]},
    {"query": "房地产和地方政府的支出加起来占到全国总需求的百分之多少？", "gold_id": 30, "keywords": ["50%"]},
    {"query": "面对房地产问题，姚洋建议筹集多少资金组建“国家队”去购买住房？", "gold_id": 32, "keywords": ["2万亿"]},

    # 专有名词与低频词题（检验 BM25 + Reranker 的威力）
    {"query": "这篇专访文章的新京报评论员（撰稿人）是谁？", "gold_id": 40, "keywords": ["柯锐"]},
    {"query": "特朗普号称对75个贸易伙伴国暂停实施关税多少天？排除了哪三个国家？", "gold_id": 13,
     "keywords": ["墨西哥", "加拿大", "中国"]},
    {"query": "被视作拉动经济增长的“三驾马车”是哪三个？", "gold_id": 22, "keywords": ["消费", "投资", "出口"]},

    # 语义改写与长难句题（检验向量检索与 Reranker 的深度语义匹配）
    {"query": "为什么过去一段时间以旧换新等促消费政策有时成效不显著？", "gold_id": 29, "keywords": ["规模小"]},
    {"query": "为什么美国突然对75个国家暂停加税？它的股票和债券市场怎么了？", "gold_id": 14, "keywords": ["抛售美债"]},
    {"query": "图片中受访的北京大学国家发展研究院教授是谁？", "gold_id": 0, "keywords": ["姚洋"]}
]


def get_embeddings(texts, model, tokenizer, batch_size=16):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[i: i + batch_size], padding=True, truncation=True, max_length=512,
                                return_tensors='pt').to(device)
            outputs = model(**encoded)
            cls_repr = outputs.last_hidden_state[:, 0]
            all_embeddings.append(F.normalize(cls_repr, p=2, dim=-1))
    return torch.cat(all_embeddings, dim=0)


# ================= 3. 四种检索策略实现 (均返回排序后的 Top-5 段落下标列表) =================
def retrieve_bm25_only(query, bm25, top_k=5):
    tokenized_query = jieba.lcut(query)
    scores = bm25.get_scores(tokenized_query)
    return torch.tensor(scores).topk(top_k).indices.tolist()


def retrieve_dense_only(query, doc_embeddings, emb_model, emb_tok, top_k=5):
    q_vec = get_embeddings([query], emb_model, emb_tok)
    scores = torch.matmul(q_vec, doc_embeddings.T).squeeze(0)
    return scores.topk(top_k).indices.tolist()


def retrieve_hybrid_rrf(query, bm25, doc_embeddings, emb_model, emb_tok, top_k=5):
    """双路召回 + RRF (倒数排名融合，不使用深度学习重排模型)"""
    bm25_top = retrieve_bm25_only(query, bm25, top_k=10)
    dense_top = retrieve_dense_only(query, doc_embeddings, emb_model, emb_tok, top_k=10)
    rrf_scores = {}
    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
    for rank, idx in enumerate(dense_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_docs[:top_k]]


def retrieve_hybrid_rerank(query, bm25, doc_embeddings, corpus, emb_model, emb_tok, rerank_model, rerank_tok, top_k=5):
    """双路召回 (各取Top-5) + BGE-Reranker 交叉注意力重排"""
    bm25_top5 = retrieve_bm25_only(query, bm25, top_k=5)
    dense_top5 = retrieve_dense_only(query, doc_embeddings, emb_model, emb_tok, top_k=5)
    candidates = list(set(bm25_top5 + dense_top5))

    rerank_model.eval()
    scored_candidates = []
    with torch.no_grad():
        for idx in candidates:
            inputs = rerank_tok(query, corpus[idx], return_tensors='pt', truncation=True, max_length=512).to(device)
            score = rerank_model(**inputs).logits.item()
            scored_candidates.append((idx, score))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored_candidates[:top_k]]


# ================= 4. 评测指标计算函数 (Hit@1, Hit@3, MRR) =================
def evaluate_strategy(name, retrieve_fn):
    hit1, hit3, mrr = 0, 0, 0.0
    total = len(EVAL_DATASET)
    for item in EVAL_DATASET:
        q = item["query"]
        gold = item["gold_id"]
        preds = retrieve_fn(q)

        if gold == preds[0]:
            hit1 += 1
        if gold in preds[:3]:
            hit3 += 1
        if gold in preds:
            rank = preds.index(gold) + 1
            mrr += 1.0 / rank

    return {
        "name": name,
        "Hit@1": f"{hit1 / total * 100:.1f}%",
        "Hit@3": f"{hit3 / total * 100:.1f}%",
        "MRR": f"{mrr / total:.4f}"
    }


# ================= 5. 大模型端到端生成准确率评测 =================
def generate_answer(query, context_text, llm_model, llm_tok):
    system_prompt = "你是一个严谨的问答助手。请严格基于提供的参考资料字面事实回答问题，简明直接，禁止主观推测。"
    user_content = f"【参考资料】\n{context_text}\n\n【用户提问】\n{query}\n\n请直接回答："
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    prompt = llm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=80, do_sample=False, use_cache=True,
                                     pad_token_id=llm_tok.eos_token_id)
    return llm_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


if __name__ == '__main__':
    print("🚀 正在初始化 RAG 消融实验评测环境...")
    corpus = load_corpus('data/train_label.json')

    # 初始化各模型
    tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    emb_tok = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
    emb_model = AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5').to(device)
    doc_embeddings = get_embeddings(corpus, emb_model, emb_tok)

    rerank_tok = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
    rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base',
                                                                      torch_dtype=torch.float16).to(device)

    print("\n================ 📊 实验一：四种检索策略消融实验对比 ================")
    results = [
        evaluate_strategy("1. BM25 (仅稀疏检索)", lambda q: retrieve_bm25_only(q, bm25)),
        evaluate_strategy("2. BGE-Small (仅向量检索)",
                          lambda q: retrieve_dense_only(q, doc_embeddings, emb_model, emb_tok)),
        evaluate_strategy("3. Hybrid RRF (双路融合无重排)",
                          lambda q: retrieve_hybrid_rrf(q, bm25, doc_embeddings, emb_model, emb_tok)),
        evaluate_strategy("4. Hybrid + Reranker (双路+交叉重排)",
                          lambda q: retrieve_hybrid_rerank(q, bm25, doc_embeddings, corpus, emb_model, emb_tok,
                                                           rerank_model, rerank_tok))
    ]

    print(f"{'检索策略 (Strategy)':<34} | {'Hit@1':<8} | {'Hit@3':<8} | {'MRR':<8}")
    print("-" * 68)
    for r in results:
        print(f"{r['name']:<34} | {r['Hit@1']:<8} | {r['Hit@3']:<8} | {r['MRR']:<8}")

    print("\n================ 🤖 实验二：Top-1 检索 + Qwen2.5-0.5B 端到端问答评测 ================")
    llm_tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    llm_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.float16).to(device)

    correct_cnt = 0
    for idx, item in enumerate(EVAL_DATASET):
        q = item["query"]
        top1_doc_id = \
        retrieve_hybrid_rerank(q, bm25, doc_embeddings, corpus, emb_model, emb_tok, rerank_model, rerank_tok, top_k=1)[
            0]
        ans = generate_answer(q, corpus[top1_doc_id], llm_model, llm_tok)

        # 检查生成的答案是否包含所有核心事实关键词
        is_hit = all(kw in ans for kw in item["keywords"])
        if is_hit:
            correct_cnt += 1
        status = "✅ 准确" if is_hit else "⚠️ 偏差"
        print(f"[题 {idx + 1:02d}] {status} | 提问: {q[:28]}...")
        print(f"       检索段落: #{top1_doc_id} (标准:#{item['gold_id']}) | 模型回答: {ans}")

    print("-" * 68)
    print(
        f"🎯 端到端最终问答准确率 (End-to-End Accuracy): {correct_cnt}/{len(EVAL_DATASET)} ({correct_cnt / len(EVAL_DATASET) * 100:.1f}%)")