# 步骤 05-5｜评测：读 dev_predictions.csv，算 accuracy / macro-P-R / 分类报告

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,f1_score,classification_report,recall_score
from config import config
from prompts import CATEGORY_DESCRIPTIONS

# 读取预测的结果
df=pd.read_csv(config.output_dir/'dev_predictions.csv')

# 只评估成功调用的样本
success_df=df[df['status']=='success'].copy()
# 获取真实标签
y_true=success_df['true_label']
# 获取预测的标签
y_pred=success_df['pred_label']
# 将预测的标签转换成数字
# 获取所有的类别
category=CATEGORY_DESCRIPTIONS.keys()
class_name2idx={class_name:idx for idx,class_name in enumerate(category)}
pred_list=[]
for pred in y_pred:
    pred_list.append(class_name2idx.get(pred))
print(pred_list)

# 获取评估指标
accuracy=accuracy_score(y_true, pred_list)
precision=precision_score(y_true,pred_list,average='macro')
recall=recall_score(y_true,pred_list,average='macro')
report=classification_report(y_true,pred_list)
print(f"准确率:{accuracy}\n精确率:{precision}\n召回率:{recall}\n综合报告:{report}")