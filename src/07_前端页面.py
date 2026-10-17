# 步骤 07｜前端页面：Streamlit 页面，调用 8001 接口展示预测结果
import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符（\ufffd）会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import os
import sys
import subprocess

# 检测是否以 bare mode（python app.py）运行：若是则自动改用 streamlit run 启动
# 这样在 PyCharm 里直接点运行按钮（python app.py）也能正常跑起来
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    _is_bare_mode = get_script_run_ctx() is None
except Exception:
    _is_bare_mode = True

if _is_bare_mode:
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', os.path.abspath(__file__)])
    sys.exit()

import streamlit as st
import requests

# 设置标题页面
st.title('新闻分类预测')
# 创建输入框
text_input=st.text_input(
    '请输入需要预测的新闻标题',
    '清华大学2027年硕士研究生招生目录'
)
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
            # 获取请求的响应  response.json() 将响应的JSON数据转换为Python字典
            result = response.json()
            st.success(f"预测类别：{result['label']}")
        else:
            st.error(f"请求失败，状态码：{response.status_code}，{response.json().get('error', '')}")
    except Exception as e:
        st.error(f"发生错误：{e}，请确认后端 predict_api.py 已启动")