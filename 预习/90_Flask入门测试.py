# 测试 90｜Flask 入门 demo：路由 / 与 /xxx，验证 flask 环境是否可用
import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符（\ufffd）会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# flask是将函数对外开始的轻量级的框架   另外一个fastapi
from flask import Flask
# 1.创建flask的应用
app=Flask(__name__)

# 2.给函数绑定路由，方便用户访问  路由是访问的路径
# url: http://127.0.0.1:5001
@app.route('/')
def get_msg():
    return 'hello,flask'

@app.route('/xxx')
def send_msg():
    return "马上就下课吃饭,好开心呀!"

if __name__ == '__main__':
    # 3.启动flask的服务，方便用户访问
    app.run(host='0.0.0.0',port=5001)