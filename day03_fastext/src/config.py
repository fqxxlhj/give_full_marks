# 步骤 03-0｜配置中心：所有路径 + 当前启用的模型（被其它脚本 import，所以不加数字前缀）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 导入 Path，用来更加规范地处理文件路径
from pathlib import Path

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:

    def __init__(self):

        # 原始数据目录
        self.raw_data_dir = (
            BASE_DIR / "data" / "raw"
        )

        # 处理后的数据目录
        self.processed_data_dir = (
            BASE_DIR / "data" / "processed"
        )

        # 模型保存目录
        self.model_dir = (
            BASE_DIR / "models"
        )

        # 训练集
        self.train_path = (
            self.raw_data_dir / "train.txt"
        )

        # 验证集
        self.dev_path = (
            self.raw_data_dir / "dev.txt"
        )

        # 测试集
        self.test_path = (
            self.raw_data_dir / "test.txt"
        )

        # 类别文件
        self.class_path = (
            self.raw_data_dir / "class.txt"
        )

        # 当前启用哪个模型（evaluate / api / app 共用），换模型只改下面两行
        # 可选：fasttext_word_tuned.bin (100 MB, dev 0.9012)  <- 默认，体积与效果平衡
        #       fasttext_word.bin       (47 MB,  dev 0.8999)
        #       fasttext_char.bin       (1.9 MB, dev 0.8716)
        #       fasttext_char_autotune.bin (8.05 GB, dev 0.9118) 加载约 45 秒、吃内存
        #       fasttext_word_autotune.bin (11.65 GB, dev 0.9130) 最高分，占 11.65 GB 内存
        self.active_model_name = "fasttext_word_tuned.bin"

        # 与模型配套的预处理模式：word / char
        self.active_model_mode = "word"

        # 自动创建目录
        self.processed_data_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":

    # 创建配置对象
    config = Config()

    # 打印关键路径
    print("项目根目录：", BASE_DIR)
    print("训练集：", config.train_path)
    print("模型目录：", config.model_dir)