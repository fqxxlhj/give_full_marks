from flask import Flask,request,jsonify  # request接收数据
from predict_fun import predict_fun
from config import Config

app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict_api():
    # 接受用户的数据
    data = request.get_json()  # {'text':'今晚月亮历史最圆'}
    '''
    常见的状态码:
    200  OK 
    4XX  路径或者提交方式
    5XX  服务器内部问题
    3XX  重定向的问题
    '''
    if not data or 'text' not in data:
        return jsonify({'error':'Invalid data'}),400
    # 调用函数进行预测
    result=predict_fun(data) # {'text':'今晚月亮历史最圆','label':'finance'}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8001)