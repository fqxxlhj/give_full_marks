# 步骤 03-3｜模型评估：sklearn 指标 + fastText 原生 test()

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import fasttext
from sklearn.metrics import accuracy_score,classification_report
from config import Config
conf=Config()

# 定义评估的函数
def evaluate_model(model_path,dev_path):
    # 加载模型
    model=fasttext.load_model(model_path)
    # 获取真实标签和预测的标签
    y_true=[]
    y_pred=[]
    for line in open(dev_path,'r',encoding='utf-8'):
        line=line.strip()
        if not line:
            continue
        # __label__sport 今天有个体育新闻
        label,text=line.split(" ",1)
        # 模型的预测
        pred_label,pred_probs=model.predict(text,k=1)
        # 保存真实标签
        y_true.append(label)
        # 保存预测的标签
        y_pred.append(pred_label)
    # 计算准确率
    accuracy=accuracy_score(y_true,y_pred)
    report=classification_report(y_true,y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    result=model.test(dev_path) # fastText原生的评估方式
    print(f"原生的评估方式:{result}")

if __name__ == '__main__':
    # 统一从 config 读当前启用的模型，避免 evaluate / api 各指向一个模型
    model_path = conf.model_dir / conf.active_model_name
    dev_path = conf.processed_data_dir / f'dev_{conf.active_model_mode}.txt'
    print(f'评估模型：{model_path.name}（{conf.active_model_mode}）')
    evaluate_model(str(model_path), str(dev_path))
