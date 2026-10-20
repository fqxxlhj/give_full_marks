# 步骤 04-4｜测试集评估：加载最优模型，在 test 上算 F1 与分类报告

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer,AutoModelForSequenceClassification
from config import config
from data import load_class_list,build_dataloader
from train import evaluate

# 加载类别文件
class_list=load_class_list()
# 加载模型
model=AutoModelForSequenceClassification.from_pretrained(config.best_model_dir)
tokenizer=AutoTokenizer.from_pretrained(config.best_model_dir)
model.to(config.device)
# 构建dataloader
dataloader=build_dataloader(config.test_path,tokenizer,shuffle=False)
# 测试集的评估
test_mertics=evaluate(model,dataloader,config.device,class_list)
print(f"f1的值为{test_mertics['f1']}")
print(f"报告:{test_mertics['report']}")