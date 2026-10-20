# 步骤 04-6｜Flask 接口：/health、/predict、/predict_batch（端口 5002）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from flask import Flask,jsonify,request
from predictor import BertNewsClassifierPredictor


app=Flask(__name__)
predictor=BertNewsClassifierPredictor()

@app.get("/health")
def health():
    return jsonify({'status':'ok'})

@app.post("/predict")
def predict():
    # silent=True：请求体不是 JSON 时返回 None，而不是直接抛异常
    data=request.get_json(silent=True)
    # 缺字段 / null / 空串 / 全空格 都算非法请求，统一返回 HTTP 400
    text=str((data or {}).get('text') or '').strip()
    if not text:
        return jsonify({'error':'text is required'}),400
    result=predictor.predict_one(text)
    return jsonify(
        {"data":result}
    )

@app.post("/predict_batch")
def predict_batch():
    data=request.get_json(silent=True)
    texts=(data or {}).get('texts')
    # texts 必须是非空列表
    if not isinstance(texts,list) or not texts:
        return jsonify({'error':'texts must be a non-empty list'}),400
    try:
        result=predictor.predict_batch(texts)
    except ValueError as error:
        # predictor 里遇到空文本会抛 ValueError，这里转成 400 而不是 500
        return jsonify({'error':str(error)}),400
    return jsonify(
        {"data":result}
    )

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5002)