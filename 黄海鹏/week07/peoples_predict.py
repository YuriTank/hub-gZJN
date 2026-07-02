# 5. 模型推理

"""
序列标注任务推理脚本

使用方式：
  # 单条推理
  python peoples_predict.py --text "任正非的华为发布了最新款手机"

"""

import argparse
import json
from pathlib import Path

import torch
from transformers import BertTokenizerFast

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "peoples_daily"
BERT_PATH = ROOT.parent.parent / "bert-base-chinese"
CKPT_DIR  = ROOT / "outputs" / "checkpoints_peoples"


def load_model_and_tokenizer(bert_path: str, ckpt_path: Path, device: torch.device):
    from peoples_dataset import build_label_schema
    from peoples_model import build_model

    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_crf = ckpt["args"].get("use_crf", False)

    model = build_model(
        use_crf=use_crf,
        bert_path=bert_path,
        num_labels=num_labels,
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    return model, tokenizer, id2label, use_crf
def decode_bio_tags(tokens: list[str], tags: list[str]) -> list[dict]:
    """将 BIO 标签序列解码为实体列表。"""
    entities = []
    current_entity = None

    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            # 记录实体在原字符列表中的真实 start 索引
            current_entity = {"type": tag[2:], "start": idx, "end": idx, "text": token}
        elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
            current_entity["text"] += token
            current_entity["end"] = idx  # 更新 end 为当前字符索引
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # 将闭区间 [start, end] 转为左闭右开区间 [start, end+1)，符合常见 NER 习惯
    for ent in entities:
        ent["end"] = ent["end"] + 1
    return entities

def predict_single(text: str, model, tokenizer, id2label: dict,
                   max_length: int, device: torch.device, use_crf: bool) -> dict:
    """单条文本推理，返回实体列表。"""
    chars = list(text)
    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.no_grad():
        if use_crf:
            pred_ids = model.decode(input_ids, attention_mask, token_type_ids)[0]
        else:
            outputs = model(input_ids, attention_mask, token_type_ids)
            print(type(outputs), len(outputs))  # 打印查看类型和长度
            # 与 predict_single 保持一致，取 outputs[0]
            logits = outputs[0]
            batch_pred_ids = logits.argmax(dim=-1).tolist()

    word_ids = encoding.word_ids(0)
    tokens, tags = [], []
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        # 对于中文字符，一个 word_id 只对应一个 token，跳过由 WordPiece 产生的 ##xx
        if len(tokens) == wid:
            tokens.append(chars[wid])
            tags.append(id2label.get(pred_ids[idx], "O"))

    entities = decode_bio_tags(tokens, tags)
    return {"text": text, "entities": entities}


def predict_batch(texts: list[str], model, tokenizer, id2label: dict,
                  max_length: int, batch_size: int, device: torch.device,
                  use_crf: bool) -> list[dict]:
    """批量推理，返回每条文本的实体列表。"""
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        chars_list = [list(t) for t in batch_texts]
        encoding = tokenizer(
            chars_list,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding["token_type_ids"].to(device)

        with torch.no_grad():
            if use_crf:
                batch_pred_ids = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids, attention_mask, token_type_ids)
                # 模型返回元组 (logits, loss)，推理时 loss 为 None，logits 在索引 0
                logits = outputs[0] 
                pred_ids = logits.argmax(dim=-1)[0].tolist()

        for j, (text, chars) in enumerate(zip(batch_texts, chars_list)):
            word_ids = encoding.word_ids(j)
            pred_ids = batch_pred_ids[j]
            tokens, tags = [], []
            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if len(tokens) == wid:
                    tokens.append(chars[wid])
                    tags.append(id2label.get(pred_ids[idx], "O"))

            entities = decode_bio_tags(tokens, tags)
            all_results.append({"text": text, "entities": entities})

    return all_results
def main():
    parser = argparse.ArgumentParser(description="BERT NER 推理")
    parser.add_argument("--use_crf",     action="store_true")
    parser.add_argument("--bert_path",   default=str(BERT_PATH), type=str)
    parser.add_argument("--ckpt_path",   default=None, type=str)
    parser.add_argument("--max_length",  default=128, type=int)
    parser.add_argument("--batch_size",  default=32, type=int)
    parser.add_argument("--text",        default=None, type=str, help="单条推理文本")
    parser.add_argument("--input_file",  default=None, type=str, help="批量推理输入 JSON")
    parser.add_argument("--output_file", default=None, type=str, help="批量推理结果输出路径")
    args = parser.parse_args()

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path \
                else CKPT_DIR / f"best_{run_tag}.pt"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    model, tokenizer, id2label, use_crf = load_model_and_tokenizer(
        args.bert_path, ckpt_path, device
    )
    print(f"模型加载完成，解码方式: {'CRF' if use_crf else 'Linear'}")

    # ── 单条推理 ──────────────────────────────────────────────────────────────
    if args.text:
        result = predict_single(args.text, model, tokenizer, id2label,
                                args.max_length, device, use_crf)
        print(f"\n文本：{result['text']}")
        if result["entities"]:
            print("抽取实体：")
            for ent in result["entities"]:
                print(f"  [{ent['type']}] {ent['text']} ({ent['start']}:{ent['end']})")
        else:
            print("未抽取到实体")
        return

    # ── 批量推理 ──────────────────────────────────────────────────────────────
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            data = json.load(f)
        texts = [item.get("text", "".join(item["tokens"])) for item in data]
        print(f"批量推理 {len(texts)} 条 ...")
        results = predict_batch(texts, model, tokenizer, id2label,
                                args.max_length, args.batch_size, device, use_crf)

        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存 → {out_path}")
        return

    # 未提供参数时给出示例
    print("请使用 --text 进行单条推理，或 --input_file 进行批量推理")
    print("\n示例（单条）：")
    examples = [
        "华为发布了最新款手机",
        "李克强总理主持会议",
    ]
    for text in examples:
        result = predict_single(text, model, tokenizer, id2label,
                                args.max_length, device, use_crf)
        print(f"  {text} -> {result['entities']}")


if __name__ == "__main__":
    main()
