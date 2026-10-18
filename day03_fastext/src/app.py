# 步骤 03-6｜Streamlit 前端：调用 5001 接口展示类别与置信度

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import streamlit as st
import requests

# 设置页面配置
st.set_page_config(page_title="新闻分类预测", page_icon="📰")

st.title('📰 新闻分类预测')
st.markdown("输入新闻文本，系统将自动预测它属于哪一类。")

# 创建输入框（默认值可以清空）
text_input = st.text_area("请输入需要预测的文本:", '郑州马拉松今日正式开跑')

# 创建预测按钮
if st.button('开始预测', type="primary"):
    # 简单的输入校验
    if not text_input.strip():
        st.warning("请输入文本后再点击预测！")
    else:
        # 构造请求数据
        data = {'text': text_input}
        # 构造请求的url
        url = 'http://127.0.0.1:5001/predict'

        # 显示一个加载动画，防止用户重复点击
        with st.spinner('正在预测中，请稍候...'):
            try:
                response = requests.post(url, json=data, timeout=10)  # 加上超时时间更安全

                if response.status_code == 200:
                    result = response.json()
                    st.success("预测成功！")

                    # 假设后端返回 {'label': '科技', 'probability': 0.95}
                    # 你可以用 st.json 把完整结果展示出来看看结构
                    st.write("**预测标签：**", result.get('label', '未知'))

                    # 如果后端返回了概率，也可以展示进度条
                    if 'probability' in result:
                        st.progress(float(result['probability']))
                        st.write(f"置信度：{result['probability']:.2%}")
                else:
                    st.error(f"请求失败，状态码：{response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error(f"连接服务器失败，请检查 Flask 服务是否正常启动。错误信息：{e}")