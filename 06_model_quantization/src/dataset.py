# 步骤 06-2｜数据管线：Dataset + collate_fn + DataLoader

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from torch.utils.data import Dataset,DataLoader
from config import config

def load_data(path):
    # 创建列表来保存所有的样本
    sample = []
    with open(path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            pars = line.split('\t', 1)
            if len(pars) != 2:
                raise ValueError(f"Invalid line at line {idx + 1}")
            # print(pars)
            text, label = pars
            sample.append((text, int(label)))
    return sample
# 自定义dataset
class NewsDataset(Dataset):
    # 初始化样本，告诉pytorch样本数据
    def __init__(self,samples):
        self.samples=samples
    # 告诉pytorch样本的数量
    def __len__(self):
        return len(self.samples)
    # 如何根据索引获取样本
    def __getitem__(self, idx):
        text,label=self.samples[idx]
        return text,label

# 自定义批量的tokenizer
def build_collect_fn(tokenizer):
    def collate_fn(batch):
        # 提取batch的文本数据
        texts=[item[0] for item in batch]
        # 提取所有的标签
        labels=[item[1] for item in batch]
        # 批量tokenizer
        encoded=tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors='pt'
        )
        # 将标签加入到Batch
        encoded['labels']=torch.tensor(labels,dtype=torch.long)
        return encoded
    return collate_fn
# 构建Dataloader
def build_dataloader(file_path,tokenizer,shuffle):
    # 加载原始样本
    samples=load_data(file_path)
    # 创建DataSet
    dataset=NewsDataset(samples)
    # 创建DataLoader
    loader=DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=build_collect_fn(tokenizer), # collate_fn 批处理Tokenizer
        num_workers=0,  # 设置工作进程的数量 一般设置为0   在主线程上运行
        #pin_memory=config.device.type=='cuda'  # GPU训练时可以提高搬运的效率
    )
    return loader
