# 步骤 03-2｜模型训练：默认超参训练 + autotune 自动调参（autotune 会产出 GB 级模型，慎用）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import fasttext
from config import Config

# 创建config对象
conf=Config()

# 训练字符级
def train_char():
    train_path=conf.processed_data_dir/'train_char.txt'
    model=fasttext.train_supervised(input=str(train_path))
    model_name="fasttext_char"
    save_model(model,model_name)

# 训练词级
def train_word():
    train_path=conf.processed_data_dir/'train_word.txt'
    model=fasttext.train_supervised(input=str(train_path))
    model_name="fasttext_word"
    save_model(model,model_name)


# 自动调参
def auto_param_tuning(mode='char',duration=60):
    train_path=conf.processed_data_dir/f"train_{mode}.txt"
    dev_path=conf.processed_data_dir/f'dev_{mode}.txt'
    model=fasttext.train_supervised(
        input=str(train_path),
        autotuneValidationFile=str(dev_path),   #验证集
        autotuneDuration=duration,      # 训练时间
        # 输出详细训练日志
        verbose=3
    )
    model_name=f'fasttext_{mode}_autotune'
    save_model(model,model_name)

# 保存模型
def save_model(model,model_name):
    # 构建模型保存的路径
    model_path=conf.model_dir/f"{model_name}.bin"
    # 保存模型
    model.save_model(str(model_path))
    print('模型保存成功',model_path)

if __name__ == '__main__':
    # train_char()
    # train_word()
    auto_param_tuning(mode='char',duration=300)
    auto_param_tuning(mode='word',duration=300)
