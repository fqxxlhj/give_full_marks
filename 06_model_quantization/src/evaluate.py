# 步骤 06-3｜精度对比：FP32 vs INT8 在 test 上的 accuracy / macro-F1

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import copy
from dataset import build_dataloader
import torch
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report
from config import config


# 加载FP32模型
def load_fp32_model():
    model = AutoModelForSequenceClassification.from_pretrained(config.fp32_model_dir)
    model.to(config.device)
    model.eval()
    return model


# 加载INT8模型
def load_int8_model():
    if not config.int8_state_path.exists():
        raise FileNotFoundError('Int8 model state not found')
    # 需要先加载和原型完全相同的FP32Bert结构
    fp32_model = AutoModelForSequenceClassification.from_pretrained(config.fp32_model_dir)
    fp32_model.to(config.device)
    fp32_model.eval()

    # 进行模型的复制
    int8_model = copy.deepcopy(fp32_model)
    # 构建量化的配置
    quantization_config = Int8DynamicActivationInt8WeightConfig()
    # 将模型的结构转换成INT8的量化结构
    quantize_(int8_model, quantization_config)

    # 读取量化模型的参数
    int8_model.load_state_dict(torch.load(config.int8_state_path))
    int8_model.eval()

    return int8_model

# 3.评估模型
def evaluate_model(model, dataloader):
    # 创建一个列表来保存所有预测的结果
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # 取出真实的标签
            labels = batch['labels'].to(config.device)
            # 创建模型的输入
            inputs = {key: value.to(config.device) for key, value in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            # 获取类别的分数
            logits = outputs.logits
            # 找到得分最大的类别
            pred = torch.argmax(logits, dim=-1)
            # 将真实的标签转换成列表进行保存
            true_list = labels.cpu().tolist()
            # 将预测的标签转换成列表进行保存
            pred_list = pred.cpu().tolist()

            # 保存真实的标签和预测的标签
            y_true.extend(true_list)
            y_pred.extend(pred_list)

    accuracy=accuracy_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred,average='macro')
    f1=f1_score(y_true,y_pred,average='macro')
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    tokenizer=AutoTokenizer.from_pretrained(config.fp32_model_dir)
    test_dataloader = build_dataloader(config.test_path,tokenizer,False)
    fp32_model = load_fp32_model()
    int8_model = load_int8_model()
    evaluate_model(fp32_model,test_dataloader)
    print('---------------------------------------')
    evaluate_model(int8_model,test_dataloader)