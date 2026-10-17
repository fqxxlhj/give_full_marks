# 测试 91｜预测 API 接口测试：向 127.0.0.1:8001/predict 发 POST 请求
import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符（\ufffd）会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


import requests   # 发送请求

# 定义请求地址
url='http://127.0.0.1:8001/predict'

# 构造请求的数据
data={
    'text':'这个景点太拉了，我是不会去的'
}
# 发送请求
response=requests.post(url,json=data)

try:
    # 检查请求的状态  response.status_code 获取请求的状态码
    if response.status_code==200:
        # 获取请求的响应  response.json() 将响应的JSON数据转换为Python字典
        result=response.json()
        print(result)
    else:
        print(f"请求失败，状态码：{response.status_code},{response.json()['error']}")
except Exception as e:
    print(f"发生错误：{e}")
