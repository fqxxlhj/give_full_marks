# 步骤 05-6｜Flask 接口：/health、/predict（端口 5003）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from flask import Flask,jsonify,request

from classifier import  classify_news

app=Flask(__name__)

@app.get('/health')
def health():
    return jsonify({
        'status':'ok'
    })

@app.post('/predict')
def predict():
    # silent=True：请求体不是 JSON 时返回 None，而不是抛异常
    data=request.get_json(silent=True)
    # 缺字段 / null / 空串 / 全空格 都算非法请求
    title=str((data or {}).get('text') or '').strip()
    if not title:
        return jsonify({
            'code':400,
            'msg':'text is required'
        }),400
    try:
        # 调用分类器
        result = classify_news(title)
        return jsonify({
            'code':0,
            'data': {
                'title': title,
                'category': result['category'],
                'reason': result['reason']
            }
        })
    except Exception as error:
        # 上游模型/网络问题用 502，和"参数错 400"区分开
        return jsonify({
            'code':502,
            'msg':'模型服务调用失败',
            'error':str(error)
        }),502

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5003)