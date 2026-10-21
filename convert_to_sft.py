import json
import random


def convert_labelstudio_to_sft(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    sft_dataset = []
    system_prompt = (
        "你是一个专业的中文命名实体识别（NER）助手。"
        "请从给定的文本中抽取出 PERSON（人名）、COUNTRY（国家）、CITY（城市）三类实体，"
        "并严格按照 JSON 格式输出，不要输出任何额外解释。"
    )

    for item in raw_data:
        text = item['data']['text']
        annotations = item['annotations'][0]['result']

        # 整理标准答案字典
        entities = {"PERSON": [], "COUNTRY": [], "CITY": []}
        for anno in annotations:
            val = anno['value']
            ent_text = val['text']
            ent_label = val['labels'][0]
            if ent_label in entities and ent_text not in entities[ent_label]:
                entities[ent_label].append(ent_text)

        # 去掉空列表的键，让输出更干净（也可保留空列表，这里保留非空实体）
        clean_entities = {k: v for k, v in entities.items() if len(v) > 0}

        sft_dataset.append({
            "system": system_prompt,
            "input": text,
            "output": json.dumps(clean_entities, ensure_ascii=False)
        })

    return sft_dataset


if __name__ == '__main__':
    data = convert_labelstudio_to_sft('data/train_label.json')

    # 固定随机种子按 8:2 划分训练集和验证集
    random.seed(42)
    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    with open('data/sft_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open('data/sft_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成！共 {len(data)} 条，训练集 {len(train_data)} 条，验证集 {len(val_data)} 条。")
    print("\n🔍 转换后的大模型指令样本示例（第 1 条）：")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2))