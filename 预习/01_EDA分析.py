# 步骤 01｜EDA 分析（脚本版）：读数据 -> 看结构 -> 查数据质量 -> 画分布图
import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符（\ufffd）会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（预习 的上一级）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题


# 1.检查数据是否可读取并查看基本结构
data = pd.read_csv(
    str(PROJECT_ROOT / 'data' / 'train.txt'),
    sep='\t',
    header=None,
    names=['sentence', 'label'], # 为两列数据指定字段列名
    encoding='utf-8',
    keep_default_na=False  # 不自动将空值转换为NaN
)
print(data.head())      # 查看数据集的前5行数据
print(data.shape)   # 查看数据集的形状
print(data.columns.tolist())  # 查看字段的名称
print(data.dtypes)  # 查看字段的数据类型

# 2.分析标签的类别分类  类别是否均衡
label_counts = Counter(data['label'].values)  #统计每个数字标签的样本数量
print(label_counts)
# 计算每个标签的占比情况
for label_id in range(10):
    count=label_counts[label_id] #获取每个标签的数量
    ratio=count/len(data)
    print(f"标签{label_id}的占比为{ratio:.2%}")


# 3.文本分析的长度   查看文本长度是否影响后续的分词、截断、padding和cut_len
data['sentence_length'] = data['sentence'].map(len)
print(data[['sentence', 'sentence_length','label']].head(10))

# 统计句子的文本长度分布
length_mean=np.mean(data['sentence_length']) # 长度均值 了解新闻标题的一般长度
length_var=np.var(data['sentence_length']) # 计算长度的方差 方差越大，文本长度的分布越分散
length_std=np.std(data['sentence_length'])# 标准差 标准差越小 文本长度接近，标准差越大，说明存在异常文本
print(f"句子长度的均值为{length_mean:.2f},方差为{length_var:.2f},标准差为{length_std:.2f}")
length_p50=np.percentile(data['sentence_length'],50) # 计算文本长度的中位数
length_p90=np.percentile(data['sentence_length'],90) # 计算p90  90%的文本长度不超过这个值
length_p95=np.percentile(data['sentence_length'],95) # 计算p95  95%的文本长度不超过这个值
length_p99=np.percentile(data['sentence_length'],99) # 计算p99的文本长度 99%的文本长度不超过这个值
print(f"句子长度的中位数为{length_p50:.2f},p90为{length_p90:.2f},p95为{length_p95:.2f},p99为{length_p99:.2f}")

# 4.检查数据质量
# 检查文本
empty_sentence_count=(data['sentence'].str.strip()=='').sum()  # 计算空文本的数量
print(f"空文本的数量为{empty_sentence_count}")
# 非法标签
numeric_labels=pd.to_numeric(data['label'],errors='coerce')   # 将标签转为数值，无法转的设为NaN
print(f"非法标签的数量为{numeric_labels.isna().sum()}")
print(f"越界标签的数量为{(~numeric_labels.between(0,9)).sum()}")   # ~表示范围之外

# 完全重复的样本
duplicated_count=data.duplicated(subset=['sentence','label']).sum()
print(f"完全重复的样本数量为{duplicated_count}")
# 同文异标
label_count_pre_text=data.groupby('sentence')['label'].nunique()   # 统计每条相同的文本对应多少个不同的标签
texts=label_count_pre_text[label_count_pre_text>1]   # 筛选对应多个标签的文本
print(texts.head())
# 检查乱码和异常字符
grabled_label=data['sentence'].str.contains(r'[^\u4e00-\u9fa5]',regex=False).sum()
print(f"疑似乱码和异常字符的数量为{grabled_label}")

# 5.检查数据集之间是否泄露  检查训练集、验证集和测试集之间不能存在大量的相同文本，否则会出现指标虚高
dev_data=pd.read_csv(
    str(PROJECT_ROOT / 'data' / 'dev.txt'),
    sep='\t',
    header=None,
    names=['sentence','label'],
    encoding='utf-8',
    keep_default_na=False)

test_data=pd.read_csv(
    str(PROJECT_ROOT / 'data' / 'test.txt'),
    sep='\t',
    header=None,
    names=['sentence','label'],
    encoding='utf-8',
    keep_default_na=False
)
# 将数据集中的数据进行处理
train_tests=set(data['sentence'].str.strip())
dev_tests=set(dev_data['sentence'].str.strip())
test_tests=set(test_data['sentence'].str.strip())

# 获取数据集之间的交叉样本的情况
train_dev_overlop=train_tests & dev_tests    # 训练集和验证集的交叉样本
train_test_overlop=train_tests & test_tests    # 训练集和测试集的交叉样本
dev_test_overlop=dev_tests & test_tests        # 验证集和测试集的交叉样本

# 如果重复率高需要重复划分数据集
print(f"训练集和验证集的交叉样本数量为{len(train_dev_overlop)}")
print(f"训练集和测试集的交叉样本数量为{len(train_test_overlop)}")
print(f"验证集和测试集的交叉样本数量为{len(dev_test_overlop)}")

# 6.绘制类别分布图
# 类别绘制柱状图显示样本的数量
class_names=[
"finance",
    "realty",
    "stocks",
    "education",
    "science",
    "society",
    "politics",
    "sports",
    "game",
    "entertainment"
]  # 定义类别名称

# 按照0-9统计类别数量
label_counts=data['label'].value_counts().sort_index().reindex(range(len(class_names)), fill_value=0)
# 将标签的数量和类别名称放在dataframe
label_df=pd.DataFrame({
    'class_name':class_names,
    'label_count':label_counts
})
# 绘制柱状图
plt.figure(figsize=(10,6))
sns.barplot(x='class_name',y='label_count',data=label_df)
plt.title('类别分布图')
plt.xlabel('类别名称')
plt.ylabel('样本数量')
plt.savefig(str(PROJECT_ROOT / 'artifacts' / 'eda' / '标签分布图.png'))
plt.show()
plt.close() # 释放资源

# 绘制文本长度分布图
plt.figure(figsize=(10,6))
sns.histplot(
    data['sentence_length'],
    bins=30, # 区间数量
)
# 使用虚线来标注p95的位置
plt.axvline(
    x=length_p95,
    color='r',
    linestyle='--',
    label=f'p95: {length_p95:.2f}'
)
plt.title('文本长度分布图')
plt.xlabel('句子长度')
plt.ylabel('样本数量')
plt.savefig(str(PROJECT_ROOT / 'artifacts' / 'eda' / '文本长度分布图.png'))
plt.show()
plt.close() # 释放资源










