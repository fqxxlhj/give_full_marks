# 导包
import pandas as pd
import numpy as np
import pickle # 保存模型和向量化器
from sklearn.feature_extraction.text import TfidfVectorizer  # 将文本转换成TF-IDF特征
from sklearn.model_selection import train_test_split  # 将数据集分割为训练集和测试集
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from config import Config

# 1.加载数据
conf=Config()  # 创建配置对象
data=pd.read_csv(conf.train_path)[:20000] # 读取训练数据
words=data['words'] # 获取分词的结果，作为模型的输入
labels=data['label'] # 获取标签，作为模型学习的目标

print(data.head())
print(words.head())
print(labels.head())

# 2.文本特征提取
# 读取停用词文件
stop_words=open(conf.stop_words,encoding='utf-8').read().split()
print(stop_words[:5])
# 创建TF-IDF词向量化器
tfidf=TfidfVectorizer(stop_words=stop_words)
# 学习词汇表,并把words转换成TF-IDF的特征向量
X=tfidf.fit_transform(words)
print(X.shape)
print(X.toarray()[:5])
# 数据集的划分
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# 3.模型的训练和评估
'''
随机森林算法:
思想: bagging+决策树
原理:
1.随机选择样本
2.随机选择特征
3.使用样本和特征生成一颗决策树
4.重复1~3生成多颗决策树
5.分类问题平权投票，回归问题求平均
api: RandomForestClassifier
n_estimators: 树的数量
max_depth: 树的最大深度
min_samples_split: 分裂内部节点所需的最小样本数
min_samples_leaf: 叶子节点所需的最小样本数
bootstrap: 是否使用自助法样本抽样
oob_score: 是否使用袋外样本评估模型
'''
model=RandomForestClassifier()
# 模型训练
model.fit(X_train, y_train)
# 模型预测
y_pred=model.predict(X_test)
print(y_pred[:10])
# 模型评估
'''
average : 表示分类报告中平均值的计算方式
micro微平均: 按照类别进行平均，类别分布不平衡，关注整体性能   结果等同于准确率
macro宏平均: 不考虑样本类别的不均衡,平等对待每一个类别，关注稀有类别，或者异常类别
weighted 加权平均: 按照样本量来进行加权,偏向于样本多的类别
准确率: 0.7322
精确率: 0.7697
召回率: 0.7316
F1分数: 0.7417
'''
print(f'准确率: {accuracy_score(y_test, y_pred):.4f}')
print(f'精确率: {precision_score(y_test, y_pred, average="macro"):.4f}')
print(f'召回率: {recall_score(y_test, y_pred, average="macro"):.4f}')
print(f'F1分数: {f1_score(y_test, y_pred, average="macro"):.4f}')
print('分类报告:', classification_report(y_test, y_pred))
# 4.保存模型
# 保存训练完成的随机森林模型
with open(conf.rf_model_save_path, 'wb') as f:
    pickle.dump(model, f)

# 保存词向量化器
with open(conf.tfidf_model_save_path, 'wb') as f:
    pickle.dump(tfidf, f)

print("模型保存完成！")

