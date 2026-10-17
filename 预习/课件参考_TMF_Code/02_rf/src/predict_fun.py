import pandas as pd
import pickle
from config import Config
import jieba
from sklearn.metrics import accuracy_score
conf=Config()

# 1.加载rf_model和tfidf模型
rf_model = pickle.load(open(conf.rf_model_save_path, 'rb'))

tfidf_model = pickle.load(open(conf.tfidf_model_save_path, 'rb'))

def predict_fun(data):
    # 1.分词
    word=' '.join(jieba.lcut(data['text'])[:30])
    # 2.词向量化
    x=tfidf_model.transform([word])
    # 3.预测
    pred = rf_model.predict(x)[0]
    print('预测结果:', pred)
    # 4.设置标签索引和名称对应的关系 {0:'finance',.....}
    id2class={i:line.strip() for i,line in enumerate(open(conf.class_name,encoding='utf-8'))}
    print('预测结果:', id2class[pred])
    data['label'] = id2class[pred]
    return data

if __name__ == '__main__':
    predict_fun('这个电影真的太差了，我不会推荐给任何人')
