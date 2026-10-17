from pathlib import Path
# 文件的路径、模型参数、超参数配置
class Config:
    # 保存项目中需要使用的文件路径
    def __init__(self):
        PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 获取项目根目录
        # 训练数据的路径
        self.train_path=PROJECT_ROOT / 'data' / 'train.csv'
        # 验证数据的路径
        self.dev_path = PROJECT_ROOT / 'data' / 'dev.csv'
        # 测试数据的路径
        self.test_path=PROJECT_ROOT / 'data' / 'test.csv'
        # 停用词的路径
        self.stop_words=PROJECT_ROOT / 'data' / 'stopwords.txt'
        # 类别路径
        self.class_name = PROJECT_ROOT / 'data' / 'class.txt'
        # 随机森林模型保存路径
        self.rf_model_save_path = PROJECT_ROOT / 'model' / 'rf_model.pkl'
        # TF-IDF 词向量化器保存路径
        self.tfidf_model_save_path=PROJECT_ROOT / 'model' / 'tfidf_vectorizer.pkl'