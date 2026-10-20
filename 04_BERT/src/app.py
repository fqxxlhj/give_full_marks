# 步骤 04-7｜Streamlit 前端：调用 Flask 5002 接口，单条 + 批量预测

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests
import streamlit as st

# =========================
# 页面配置
# =========================
st.set_page_config(page_title="BERT 新闻分类预测", page_icon="📰", layout="centered")

DEFAULT_API = "http://127.0.0.1:5002"
MAX_BATCH_SIZE = 200


# =========================
# 接口调用封装
# =========================
def call_health(base_url):
    """GET /health，返回 True/False"""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200 and str(resp.json().get("status", "")).lower() == "ok"
    except Exception:
        return False


def call_predict(base_url, text, timeout=10):
    """POST /predict，返回 data 字段"""
    resp = requests.post(f"{base_url}/predict", json={"text": text}, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if "data" not in body:
        raise ValueError(f"接口返回格式异常：{body}")
    return body["data"]


def render_score(score):
    """score 在 0~1 视为概率，否则视为原始 logits"""
    if 0.0 <= score <= 1.0:
        st.progress(float(score))
        st.caption(f"置信度：{score:.2%}")
    else:
        st.caption(f"模型原始分数（logits，未做 softmax 归一化）：{score:.4f}")


# =========================
# 页面主体
# =========================
st.title("📰 BERT 新闻分类预测")
st.markdown("输入新闻标题，系统将调用 BERT 模型预测它属于哪一类（共 10 类）。")

# 侧边栏：服务设置
with st.sidebar:
    st.header("服务设置")
    base_url = st.text_input("API 地址", DEFAULT_API).rstrip("/")
    if st.button("检测后端服务"):
        if call_health(base_url):
            st.success("后端服务正常")
        else:
            st.error("无法连接后端，请先启动 api.py")
    st.caption("提示：day03 的 FastText 接口同样占用 5001，两个服务不能同时启动。")

tab_single, tab_batch = st.tabs(["单条预测", "批量预测"])

# -------------------------
# 单条预测
# -------------------------
with tab_single:
    text_input = st.text_area(
        "请输入需要预测的新闻文本：",
        "56岁女子退休后考研上岸",
        height=120,
    )

    if st.button("开始预测", type="primary", key="btn_single"):
        if not text_input.strip():
            st.warning("请输入文本后再点击预测！")
        else:
            with st.spinner("正在预测中，请稍候..."):
                try:
                    result = call_predict(base_url, text_input.strip())
                except Exception as e:
                    st.error(f"请求失败，请检查 Flask 服务是否已启动。错误信息：{e}")
                else:
                    st.success("预测成功！")
                    col1, col2 = st.columns(2)
                    col1.metric("预测类别", result.get("pred", "未知"))
                    col2.metric("模型分数", f"{float(result.get('score', 0.0)):.4f}")

                    render_score(float(result.get("score", 0.0)))

                    with st.expander("查看原始返回"):
                        st.json(result)

# -------------------------
# 批量预测
# -------------------------
with tab_batch:
    st.caption(f"每行一条文本，一次最多 {MAX_BATCH_SIZE} 条。")
    batch_input = st.text_area(
        "请输入多条新闻文本（每行一条）：",
        "郑州马拉松今日正式开跑\n沪指收盘上涨1.2%\n新赛季游戏主机销量公布",
        height=200,
        key="batch_input",
    )

    if st.button("批量预测", type="primary", key="btn_batch"):
        texts = [line.strip() for line in batch_input.splitlines() if line.strip()]

        if not texts:
            st.warning("请输入至少一条文本！")
        elif len(texts) > MAX_BATCH_SIZE:
            st.warning(f"一次最多 {MAX_BATCH_SIZE} 条，当前 {len(texts)} 条，请分批提交。")
        else:
            rows = []
            failed = 0
            progress = st.progress(0.0)

            for i, text in enumerate(texts):
                try:
                    result = call_predict(base_url, text)
                    rows.append({
                        "index": i + 1,
                        "text": text,
                        "pred": result.get("pred"),
                        "score": round(float(result.get("score", 0.0)), 4),
                    })
                except Exception:
                    failed += 1
                    rows.append({
                        "index": i + 1,
                        "text": text,
                        "pred": "预测失败",
                        "score": None,
                    })
                progress.progress((i + 1) / len(texts))

            progress.empty()

            if failed == 0:
                st.success(f"全部 {len(texts)} 条预测完成！")
            elif failed < len(texts):
                st.warning(f"完成 {len(texts) - failed} 条，失败 {failed} 条。")
            else:
                st.error("全部预测失败，请检查 Flask 服务是否已启动。")

            st.dataframe(rows, use_container_width=True)