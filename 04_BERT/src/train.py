# 步骤 04-3｜训练循环：AdamW + 线性 warmup + 梯度裁剪 + 按 dev macro-F1 早停存最优

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 导包
import random
import json
import numpy as np
import torch
from tqdm import tqdm
from config import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup
from data import load_class_list,build_dataloader
from metrics import cal_metrics

# 固定随机种子
def set_seed(seed):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)    # Numpy 随机种子
    torch.manual_seed(seed)  # CPU 随机种子
    torch.cuda.manual_seed_all(seed)    # GPU 随机种子

# 将一个batch移动到指定设备
def move_batch_to_device(batch,device):
    return {k: v.to(device) for k,v in batch.items()}

# 模型的评估
def evaluate(model,data_loader,device,class_name):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="模型评估"):
            # 将batch数据移动到指定设备
            batch = move_batch_to_device(batch,device)
            # 前向传播
            outputs = model(**batch)
            # 拿到预测结果
            logits = outputs.logits
            # 获取分数最大的预测结果
            pred = torch.argmax(logits,dim=-1)
            y_true.extend(batch['labels'].cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    # 计算评估指标
    return cal_metrics(y_true,y_pred,class_name)



# 训练模型
def train():
    # 1.固定随机种子
    set_seed(config.seed)

    # 2.数据准备
    # 加载类别列表
    class_list = load_class_list()
    # 类别-->ID
    class_to_id = {class_name: i for i, class_name in enumerate(class_list)}
    id_to_class = {i: class_name for i, class_name in enumerate(class_list)}
    # 类别数量
    num_classes = len(class_list)

    # 3.加载bert模型
    model=AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_path,
        num_labels=num_classes,
        id2label=id_to_class,
        label2id=class_to_id
    )
    model.to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)

    # 4.构建dataloader
    train_dataloader=build_dataloader(config.train_path, tokenizer, shuffle=True)
    dev_dataloader=build_dataloader(config.dev_path, tokenizer, shuffle=False)

    # 5.优化器、学习率、损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # 获取训练的step数
    total_steps = len(train_dataloader) * config.num_epochs
    # warmup step数
    warmup_steps = int(total_steps * config.warmup_ratio)
    # 创建学习率调度器
    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    # 设置早停的参数
    no_improve_epochs=0
    best_dev_f1=0

    # 6.循环训练（核心5步）
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        # 创建进度条
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in progress_bar:
            # 将batch数据移动到指定设备
            batch = move_batch_to_device(batch, config.device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(**batch)
            # 计算损失
            loss = outputs.loss
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 累加损失
            total_loss += loss.item()
        # 计算平均损失
        train_avg_loss=total_loss / len(train_dataloader)
        # 每个epoch结束，需要进行dev
        dev_metrics = evaluate(model, dev_dataloader, config.device, class_list)
        # 获取f1值
        dev_f1 = dev_metrics['f1']
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_avg_loss}, Dev F1: {dev_f1}")

        # 7.根据评估中F1值找到最优的模型并保存
        # 判断当前的f1是否为最优的f1值
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            # 重置为提升的次数
            no_improve_epochs = 0
            # 保存模型
            model.save_pretrained(config.best_model_dir)
            tokenizer.save_pretrained(config.best_model_dir)

            # 保存原数据信息
            metadata={
                'class_list': class_list,
                'max_length': config.max_length,
                'best_dev_f1': best_dev_f1,
                'pretrained_model':str(config.pretrained_model_path)
            }
            # 将原始数据信息写入json文件
            with open(config.best_model_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print('最新模型已保存')

        else:
            # 8.早停策略
            no_improve_epochs += 1
            if no_improve_epochs >= config.patience:
                print("Early stopping: No improvement for")
                break

if __name__ == '__main__':
    train()