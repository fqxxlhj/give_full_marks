# 步骤 04-2｜评估指标：accuracy / macro 精确率-召回率-F1 / 分类报告

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

# 声明函数，来获取评估指标
def cal_metrics(y_true,y_pred,class_name):
    # 准确率
    accuracy = accuracy_score(y_true,y_pred)
    # 精确率
    precision = precision_score(y_true,y_pred,average='macro',zero_division=0)
    # 召回率
    recall = recall_score(y_true,y_pred,average='macro',zero_division=0)
    # F1分数
    f1 = f1_score(y_true,y_pred,average='macro',zero_division=0)
    # 分类报告
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_name,
        digits=4,
        zero_division=0
    )
    return {
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'report':report
    }
