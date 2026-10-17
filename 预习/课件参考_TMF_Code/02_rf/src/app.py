import streamlit as st
import requests

# 设置标题页面
st.title('新闻分类预测')
# 创建输入框
text_input = st.text_input('请输入需要预测的新闻标题','清华大学2027年硕士研究生招生目录')
# 创建预测的按钮
if st.button('预测'):
    # 构造请求数据
    data={'text':text_input}
    # 构造请求的url
    url='http://127.0.0.1:8001/predict'

    try:
        # 发送请求
        response = requests.post(url, json=data)
        # 检查请求的状态  response.status_code 获取请求的状态码
        if response.status_code == 200:
            result = response.json()
            st.success(f'预测结果为：{result["label"]}')
        else:
            print(f"请求失败，状态码：{response.status_code},{response.json()['error']}")
    except Exception as e:
        print(f"请求失败，错误信息：{e}")
st.write('请确保Flask API的服务已经在本地的8001端口启动！')
