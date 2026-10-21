# 步骤 06-1｜INT8 动态量化：torchao quantize_ -> 保存 model_state.pt

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import copy
import torch
from transformers import AutoModelForSequenceClassification
# 导入Torch AO的INT8动态量化配置
from torchao.quantization import Int8DynamicActivationInt8WeightConfig
# 导入量化的函数
from torchao.quantization import quantize_
from config import config

# 1.加载原始FP32模型
fp32_model=AutoModelForSequenceClassification.from_pretrained(config.fp32_model_dir)
# 2.将模型放在CPU上
fp32_model.to(config.device)

# 3.将模型切换到推理模型
fp32_model.eval()

# 4.复制一份模型，使用复制的模型来进行INT8量化
int8_model=copy.deepcopy(fp32_model)

# 5.创建INT8动态量化配置
quantize_config = Int8DynamicActivationInt8WeightConfig()

# 6.执行量化
quantize_(int8_model, quantize_config)
# 7.进入推理模式
int8_model.eval()
# 8.保存量化的参数
torch.save(int8_model.state_dict(), config.int8_state_path)

print("INT8模型保存成功")






