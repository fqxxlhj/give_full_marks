# 步骤 04-5｜推理封装：BertNewsClassifierPredictor（单条 / 批量，softmax 置信度）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import config


class BertNewsClassifierPredictor:
    def __init__(self):
        # 1.读取模型原数据
        with open(config.best_model_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            # 类别列表
            self.class_list = metadata['class_list']
            # 最大长度
            self.max_length = metadata['max_length']
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(config.best_model_dir)
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(config.best_model_dir)
            self.model.to(config.device)
            self.device = config.device
            self.model.eval()
        # 预测单条数据

    def predict_one(self, text):
        # 去掉首位空格
        text = text.strip()
        # 分词
        tokens = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors='pt')
        # 转换为设备
        tokens = tokens.to(self.device)
        with torch.no_grad():
            # 模型预测
            logits = self.model(**tokens).logits
            # 预测结果
            prob = torch.softmax(logits, dim=1)
            # 获取最高概率类别
            score, pred = torch.max(prob, dim=1)
            pred = pred.item()
        return {
            'text': text,
            'pred': self.class_list[pred],
            'score': score.item()
        }

    # 预测多条数据
    # 批量预测
    def predict_batch(self, texts):
        # 清洗文本
        texts = [
            text.strip()
            for text in texts
        ]

        # 判断是否存在空文本
        if any(
                not text
                for text in texts
        ):
            raise ValueError(
                "texts 中不能包含空文本"
            )

        # 一次 Tokenize 整个 Batch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tensor 移动到设备
        encoded = {
            key: value.to(
                self.device
            )
            for key, value in encoded.items()
        }

        # 推理
        with torch.no_grad():

            # 模型前向传播
            outputs = self.model(
                **encoded
            )

            # 得到每个类别概率
            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            # 获取每条文本最大概率及类别
            scores, pred_indices = torch.max(
                probabilities,
                dim=1
            )

        # 构造结果
        results = []

        # 遍历每条文本
        for text, pred_idx, score in zip(
                texts,
                pred_indices.tolist(),
                scores.tolist()
        ):
            # 添加预测结果
            results.append(
                {
                    "text": text,
                    "pred_class": (
                        self.class_list[
                            pred_idx
                        ]
                    ),
                    "score": round(
                        score,
                        6
                    ),
                }
            )

        # 返回全部结果
        return results


if __name__ == '__main__':
    predictor = BertNewsClassifierPredictor()
    text = "56岁女子退休后考研上岸"
    result = predictor.predict_one(text)
    print(result)
