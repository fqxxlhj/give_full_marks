# 步骤 03-5｜Flask 接口：GET /health、GET /、POST /predict（端口 5001）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from flask import Flask,jsonify,request # jsonify 将python数据转换成json数据   request接收收据  requests发送请求
from pathlib import Path
from predictor import FastTextPredictor
from config import Config
conf=Config()

# 构建flask的应用
app=Flask(__name__)

# 创建FastTextPredictor对象
predictor=FastTextPredictor(str(conf.model_dir/conf.active_model_name),conf.active_model_mode)

# 编写一个健康接口，用于服务的检测
@app.route('/health')
def health():
    return jsonify({'status':'OK'})

@app.route('/')
def index():
    return jsonify({'msg': 'Welcome to FastText API. Use /predict to POST data.'})


# 编写一个预测接口，用于接收用户数据并返回预测结果
@app.route('/predict',methods=['POST'])
def predict():
    # 接收数据；silent=True 让"非 JSON 请求体"返回 None 而不是抛异常
    data=request.get_json(silent=True)
    # 统一取出 text 并做空值兜底：缺字段 / null / 空串 / 全空格 都视为非法请求
    text=str((data or {}).get('text') or '').strip()
    if not text:
        # 注意：HTTP 状态码也要是 400，不能只在 JSON body 里写 code=400
        return jsonify({
            'code':400,
            'msg':'text is required'
        }),400
    # 获取数据
    result=predictor.predict(text)
    return jsonify(result)



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001,debug=True)




