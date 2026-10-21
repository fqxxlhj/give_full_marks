# 步骤 06-0｜配置：FP32/INT8 模型与数据路径 + 量化与基准参数（@dataclass）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 导入 dataclass
# dataclass 可以帮助我们更方便地定义配置类
from dataclasses import dataclass

# 导入 Path
# Path 用来处理文件路径
from pathlib import Path


# =========================================================
# 1. 获取项目根目录
# =========================================================

# __file__ 表示当前 config.py 文件
# resolve() 转换成完整路径
# parent 表示上一级目录
# 再 parent 一次回到项目根目录
BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# =========================================================
# 2. 定义配置类
# =========================================================

# 使用 dataclass 简化配置类写法
@dataclass
class Config:

    # FP32 BERT 模型所在目录
    fp32_model_dir: Path = (
        BASE_DIR
        /
        "models"
        /
        "bert_fp32"
    )

    # INT8 模型保存路径
    int8_state_path: Path = (
        BASE_DIR
        /
        "models"
        /
        "bert_int8"
        /
        "model_state.pt"
    )

    # 测试集路径
    test_path: Path = (
        BASE_DIR
        /
        "data"
        /
        "test.txt"
    )

    # 动态量化实验使用 CPU
    device: str = "cpu"

    # 每次取多少条样本
    batch_size: int = 32

    # BERT 最大文本长度
    max_length: int = 128

    # 正式计时前先预热多少次
    warmup_steps: int = 20

    # 正式重复测试多少次
    benchmark_steps: int = 100


# =========================================================
# 3. 创建配置对象
# =========================================================

# 创建 Config 对象
config = Config()


# =========================================================
# 4. 自动创建 INT8 模型目录
# =========================================================

# 如果 models/bert_int8 不存在就自动创建
config.int8_state_path.parent.mkdir(
    parents=True,
    exist_ok=True
)