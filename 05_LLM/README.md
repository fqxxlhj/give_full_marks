# 05 · LLM 零样本新闻分类

> 2026-10-21 完成评测。**500 条 dev 抽样：accuracy 0.8000 / macro-F1 0.7976**
>
> **结论：零样本大模型（0.80）打不过微调的小模型**（FastText 0.9012、BERT 0.9411）。这个结果本身就是这个子项目最大的价值。

---

## 1. 结论先说

| 模型 | dev 准确率 | 训练成本 | 单条延迟 | 模型体积 |
|---|---|---|---|---|
| TF-IDF + 随机森林 | 0.7306 | 分钟级 | 毫秒 | 169 MB |
| FastText 词级 | 0.9012 | 十几秒 | 毫秒 | 100 MB |
| **BERT 微调** | **0.9411** | 18.56 小时（CPU） | 几十毫秒 | 390 MB |
| **LLM 零样本（本项目）** | **0.8000** | **0** | **882 ms** | **0** |

**为什么零样本会输 14 个点**：
1. **这是一个"窄而深"的任务** —— 10 个类别的边界是**数据集约定**的，不是常识。比如 `stocks` 只收股票行情、`finance` 收银行利率，模型没有见过这个数据集，只能按自己的理解分。
2. **零样本没有"校准"** —— 18 万条训练数据里"股票新闻长什么样"这个分布，prompt 里那几行界定语根本表达不完。
3. 反过来看：**如果标注数据很少**（比如每类 50 条），零样本 0.80 会是非常强的起点，这时候它才划算。

---

## 2. 目录结构

```
05_LLM/
├── README.md              ← 本文件
├── .env                   ← API Key（gitignore，绝不入库）
├── .gitignore             ← .env / outputs/ / __pycache__ / .idea
├── data/                  ← 原始数据（10.8 MB，不入库）
│   ├── train.txt  dev.txt  test.txt  class.txt
├── results/               ← 评测结果留档（入库）
│   └── llm_zero_shot_summary.json
├── outputs/               ← 运行产物（不入库）
│   └── dev_predictions.csv     500 条逐条结果（含 reason / 延迟 / token）
└── src/
    ├── config.py          ← 05-0 配置（@dataclass）
    ├── prompts.py         ← 05-1 Prompt 模板（10 类界定 + 防注入 + JSON 示例）
    ├── deepseek_client.py ← 05-2 API 客户端（tenacity 重试 + json_object）
    ├── classifier.py      ← 05-3 单条分类（JSON 解析 + 类别白名单 + token 统计）
    ├── predict_one.py     ← 05-4 批量预测（断点续跑 + 逐条落盘）
    ├── evaluate.py        ← 05-5 评测（accuracy / macro-F1 / 分类报告）
    ├── api.py             ← 05-6 Flask 接口（端口 5003）
    └── app.py             ← 05-7 Streamlit 前端
```

---

## 3. 运行顺序

先准备 `05_LLM/.env`（**不要提交**）：

```ini
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

```powershell
conda activate fasttext_env     # 需要 openai / python-dotenv / tenacity / pandas
cd 05_LLM\src

python classifier.py      # 05-3 单条冒烟（会消耗一次 API）
python predict_one.py     # 05-4 跑 500 条抽样，写 outputs/dev_predictions.csv
python evaluate.py        # 05-5 算准确率 / 分类报告
python api.py             # 05-6 起接口（5003）
python app.py             # 05-7 前端页面
```

---

## 4. 批量预测的三个工程细节（`predict_one.py`）

这三条是"能跑完 500 条不崩、不重复花钱"的关键：

| 设计 | 实现 | 为什么需要 |
|---|---|---|
| **断点续跑** | 启动时读已有 CSV，把 `status=='success'` 的 `row_id` 收进集合，逐条跳过 | 中途 Ctrl+C 或断网，重跑不会重复计费 |
| **逐条落盘** | 每预测一条就 `to_csv(mode='a')` 追加 | 进程崩了最多丢一条，不会丢整批 |
| **失败也记一行** | `try/except` 里写 `status='fail'` + `error` 字段 | 知道哪些失败、为什么失败，而不是静默跳过 |

> 代价是逐条 `to_csv` 有 IO 开销（500 条约几分钱的时间），但对"调 API 要花钱"的场景，**这个取舍是对的**。

---

## 5. 评测结果（500 条随机抽样，`random_state=42`）

**accuracy 0.8000 ｜ macro-F1 0.7976 ｜ 加权 F1 0.7984 ｜ 错 100/500**

> 抽样 500 条，95% 置信区间约 ±3.5 个百分点，即真实值大概在 0.76~0.84 之间。

| 类别 | precision | recall | f1 | 样本数 |
|---|---|---|---|---|
| sports | 0.9455 | 1.0000 | **0.9720** | 52 |
| game | 0.9250 | 0.9024 | 0.9136 | 41 |
| education | 0.9091 | 0.8929 | 0.9009 | 56 |
| entertainment | 0.8462 | 0.9167 | 0.8800 | 36 |
| realty | 0.8511 | 0.8511 | 0.8511 | 47 |
| science | 0.9200 | 0.7797 | 0.8440 | 59 |
| society | 0.7108 | 0.9077 | 0.7973 | 65 |
| politics | 0.8438 | 0.5400 | **0.6585** | 50 |
| finance | 0.5636 | 0.6200 | **0.5905** | 50 |
| stocks | 0.5682 | 0.5682 | **0.5682** | 44 |

### 错误集中在这两个地方

| 真实类别 | 被判成 | 条数 | 说明 |
|---|---|---|---|
| finance | → stocks | 17 | 银行/利率新闻被判成股票行情 |
| politics | → society | 16 | 政策新闻被判成社会新闻 |
| stocks | → finance | 15 | 反向混淆 |

**`finance ↔ stocks` 共错 32 条（占全部错例 32%）**，**`politics → society` 错 16 条（16%）**。

被判成 `society` 的 83 条里，只有 59 条真是 society，另外 16 条是 politics —— **模型把"政策性社会新闻"一律归到 society**。这正是 prompt 里"politics 收政府政策、外交、国际政治"这条界定不够锐利导致的。

---

## 6. 延迟与成本（实测）

| 指标 | 数值 |
|---|---|
| 平均延迟 | 877 ms/条 |
| p50 / p90 / p95 / p99 | 882 / 1061 / 1155 / 1445 ms |
| 最小 / 最大 | 421 / 1823 ms |
| 每条 prompt tokens | 339 |
| 每条 completion tokens | 28 |
| 500 条总 token | 183,630（prompt 169,490 + completion 14,140） |
| 外推 1 万条 dev | 约 **3.67M token** |

**对比**：BERT 在 CPU 上单条推理是几十毫秒，这里 882 ms 是**慢 20~30 倍**，而且还要按 token 付费。所以"用大模型省事"这个直觉，在**数据充足 + 类别固定**的场景下并不成立。

---

## 7. 已知问题

1. **只跑了 500 条抽样**（为了控制费用），没有全量 10000 条。
2. **串行请求**：`predict_one.py` 是逐条跑，500 条花了约 7 分钟；全量 1 万条要 2 个多小时。应改成 8~16 并发 + 限流。
3. **没有缓存**：同一条文本重复调用会重复计费。
4. **类别界定语可以再调**：从混淆看，`politics` / `society` 的界定要收紧（比如给 politics 加"国务院、外交部、白宫"等词面线索）。
5. `evalute.py` 原文件名拼写错误（应为 `evaluate.py`）→ 本次已改名。
6. **`.env` 里的 Key 是明文**，只靠 `.gitignore` 保护；共享环境建议改用环境变量注入。

---

## 8. 下一步

- [ ] few-shot：每类给 2~3 个真实样本，看能不能从 0.80 拉到 0.88+
- [ ] 改进 prompt：把混淆最重的两类加上"词面线索 + 反例"
- [ ] 并发 + 缓存，跑全量 dev，得到无偏的基线数字
- [ ] 成本核算：按当前 token 单价算出"1 万条要多少钱"，和 BERT 的训练成本对比
- [ ] 对比"LLM 生成伪标签 → 蒸馏到小模型"的性价比
