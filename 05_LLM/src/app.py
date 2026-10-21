# 步骤 05-7｜Streamlit 前端：调用 5003 接口展示类别与理由

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import streamlit as st
import requests  # 发送请求

# 设置标题
st.title('新闻分类预测')
# 创建输入框
text_input = st.text_area("请输入需要预测的文本:", '郑州马拉松今日正式开跑')
# 创建预测按钮
if st.button('预测'):
    if not text_input.strip():
        st.warning('请输入文本后再点击预测')
    else:
        # 构造请求数据
        data = {'text': text_input}
        # 构造请求的 url（LLM 接口是 5003，比 BERT 的 5002 慢，超时给足）
        url = 'http://127.0.0.1:5003/predict'
        try:
            response = requests.post(url, json=data, timeout=60)
            # 非 2xx 直接抛异常，避免后面取 data 时 KeyError
            response.raise_for_status()
            result = response.json()
            if result.get('code') != 0:
                st.error(f"接口返回异常：{result}")
            else:
                payload = result['data']
                st.success(f"预测类别：{payload.get('category')}")
                # 大模型还会给出判断理由，这是它比 BERT 多出来的信息
                if payload.get('reason'):
                    st.caption(f"理由：{payload['reason']}")
        except Exception as e:
            st.error(f"请求失败，请确认 api.py 已启动：{e}")
