# 步骤 05-2｜API 客户端：OpenAI 兼容 + tenacity 重试 + json_object 模式

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os

from dotenv import load_dotenv  # 加载.env文件
# 常见一些可重试的异常
from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential  # 重试的工具

from config import config
from prompts import SYSTEM_PROMPT

# 加载.env文件
load_dotenv()

# 读取key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("没有读取到api-key")
# 构建客户端
client = OpenAI(
    api_key=api_key,
    base_url=config.base_url,
    timeout=config.timeout,
)


# 声明临时性错误进行重试
@retry(
    retry=retry_if_exception_type(
        (APIConnectionError,APITimeoutError,RateLimitError,InternalServerError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1,min=4,max=10),
    reraise=True
)
def call_deepseek(messages):
    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        stream=False,
        max_tokens=config.max_tokens,
        extra_body={"thinking": {"type": "disabled"}},  # 关闭思考模式
        temperature=0,  # 非思考模式使用较低的温度
        response_format={'type': 'json_object'}
    )
    return response



if __name__ == '__main__':
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "GPT给我重新刷新"}
    ]
    resp = call_deepseek(messages)
    print(resp.choices[0].message.content)
