# 步骤 05｜模型预测评估：加载 pkl，对 dev 集预测并计算准确率
import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符（\ufffd）会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import pandas as pd
import pickle
from config import Config
from sklearn.metrics import accuracy_score

conf=Config()

# 1.加载rf_model和tfidf模型
rf_model = pickle.load(open(conf.rf_model_save_path, 'rb'))

tfidf_model = pickle.load(open(conf.tfidf_model_save_path, 'rb'))

# 2.加载测试数据
dev_data=pd.read_csv(conf.dev_path)
# dev中的words列
dev_words=dev_data['words']
# 将words进行向量化
dev_words_tfidf=tfidf_model.transform(dev_words)
# 使用随机森林模型进行预测
dev_pred=rf_model.predict(dev_words_tfidf)
# 将预测的结果保存到dev_pred中
dev_data['pred']=dev_pred
print('dev数据预测完成')
print(dev_data[['text','label','pred']].head())
print('dev数据预测准确率:', accuracy_score(dev_data['label'], dev_data['pred']))