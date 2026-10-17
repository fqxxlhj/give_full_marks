import requests  # 发送请求

# 定义请求的地址
url='http://127.0.0.1:8001/predict'

# 构造请求的数据
data={'text':'这个电影真的太差了，我不会推荐给任何人'}

# 发送请求
response=requests.post(url,json=data)

try:
    # 检查请求的状态  response.status_code 获取请求的状态码
    if response.status_code == 200:
        result = response.json()
    else:
        print(f"请求失败，状态码：{response.status_code},{response.json()['error']}")
except Exception as e:
    print(f"请求失败，错误信息：{e}")

