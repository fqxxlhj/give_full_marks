# 步骤 03-4｜预测器封装：FastTextPredictor.predict(文本) -> {label, probability}（被 api 导入）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import fasttext
from preprocess import preprocess_text
from config import Config
conf=Config()

class FastTextPredictor:
    def __init__(self, model_path, mode='word'):
        self.model = fasttext.load_model(str(model_path))
        self.mode = mode

    def predict(self, text):
        pre_text = preprocess_text(text, mode=self.mode)
        # 模型的预测
        labels, probs = self.model.predict(pre_text, k=1)
        # 获取标签
        label = labels[0].removeprefix('__label__')
        # 获取概率
        probability = float(probs[0])
        result = {
            'label': label,
            'probability': probability
        }
        return result

# model_path=conf.model_dir/'fasttext_char_autotune.bin'
# predictor=FastTextPredictor(model_path,'char')
# result=predictor.predict('这个电影真的太差了，我不会推荐给任何人')
# print(result)