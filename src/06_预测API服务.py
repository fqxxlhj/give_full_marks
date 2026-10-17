# 步骤 06｜预测 API 服务：Flask 封装 /predict 接口（端口 8001）
import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符（\ufffd）会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from flask import Flask,request,jsonify
from predict_fun import predict_fun
from config import Config

app=Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict_api():
    # 接收用户数据
    data=request.get_json()
    '''
    常见的状态码：
    200  OK 
    4XX  路径或者提交方式
    5XX  服务器内部问题
    3XX  重定向的问题
    '''
    if not data or 'text' not in data:
        return jsonify({'error':'Invalid data'}),400
    # 调用函数进行预测
    result=predict_fun(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8001)