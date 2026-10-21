# 步骤 05-3｜单条分类：调模型 -> JSON 解析 -> 类别白名单校验 -> 返回 reason 与 token 用量

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
from prompts import SYSTEM_PROMPT,CATEGORY_DESCRIPTIONS
from deepseek_client import call_deepseek


# 获取所有的类别
category=CATEGORY_DESCRIPTIONS.keys()

def classify_news(news):
    # 参数校验
    news=news.strip()
    if not news:
        raise ValueError("新闻不能为空")
    # 构建messages
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"请分析下面这条新闻的类别：{news}"
        }
    ]
    # 调用模型
    response=call_deepseek(messages)
    content=response.choices[0].message.content
    # 解析json
    result=json.loads(content)
    # 获取类别
    class_name=result.get('category')
    # 类似白名单校验
    if class_name not in category:
        raise ValueError(f"模型返回非法类别：{class_name}")
    # 获取原因
    reason=result.get('reason')
    # 获取token的使用量
    usage=response.usage
    return {
        "category": class_name,
        "reason": reason,
        "usage": usage,
        'prompt_tokens': usage.prompt_tokens,
        'completion_tokens': usage.completion_tokens,
        'total_tokens': usage.total_tokens
    }

if __name__ == '__main__':
   print(classify_news('国家数据集团成立'))