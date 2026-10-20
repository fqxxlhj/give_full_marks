# 步骤 04-8｜接口冒烟测试：向 /predict_batch 发 3 条，打印结果与耗时

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 导入 requests
import requests

# 导入 time
import time

# 接口地址
url = (
    "http://127.0.0.1:5002/predict_batch"
)

# 构造 JSON 请求
payload = {
    "texts": ["56岁女子退休后考研上岸","大二学生炒股月入2000,但是考试分数为0","建议花几秒钟跟AI抬个杠"]
}

# 记录开始时间
start_time = time.time()


try:

    # 发送 POST 请求
    response = requests.post(
        url,
        json=payload,
        timeout=10
    )

    # 请求耗时
    elapsed_ms = (
        time.time()
        -
        start_time
    ) * 1000

    # 如果状态码不是 2xx
    # 自动抛出异常
    response.raise_for_status()

    # 解析 JSON
    result = response.json()

    # 打印结果
    print(
        "接口结果：",
        result
    )

    # 打印耗时
    print(
        f"请求耗时："
        f"{elapsed_ms:.2f} ms"
    )

except requests.RequestException as error:

    # 打印网络错误
    print(
        "请求失败：",
        error
    )