# day03 · FastText 新闻标题分类

> 2026-10-18 完成。目标：把昨天的「TF-IDF + 随机森林」换成 **FastText**，对比效果与工程复杂度。
>
> **结果：dev 准确率 0.7306 → 0.9012**（词级 + 合理超参，模型 100 MB）；**最优 0.9130**（autotune，但模型 11.65 GB）。
> 训练从昨天的"分钟级"降到"十几秒~几十秒"，模型从 169 MB 降到 47~100 MB（autotune 版除外）。

---

## 1. 这个子项目做了什么

在昨天的 10 类新闻标题数据上，用 FastText 跑通「预处理 → 训练 → 评估 → 封装预测器 → Flask 接口 → Streamlit 页面」，并且对比了四种做法：

| 做法 | 做法说明 | dev 准确率 | 模型体积 |
|---|---|---|---|
| 字符级 + 默认超参 | `list(text)` 逐字切分 | 0.8716 | 1.9 MB |
| 词级 + 默认超参 | `jieba.lcut(text)` | 0.8999 | 47.1 MB |
| 词级 + 固定合理超参 | `dim=100, epoch=10, lr=0.5, wordNgrams=2, bucket=200000` | 0.9012 | 100.2 MB |
| autotune 自动调参 | 让 fastText 自己搜 300 秒 | **0.9118 / 0.9130** | **8.05 / 11.65 GB** |

**结论**：autotune 确实能多拿 1.2 个点，但模型膨胀 100~250 倍；日常开发/演示用 100 MB 的固定超参版本，想冲最高分再用 autotune。

---

## 2. 环境

```powershell
conda activate fasttext_env        # 本项目实测环境：D:\AnacondaEnvs\fasttext_env
pip install -r requirements.txt
python -c "import fasttext, jieba, flask, streamlit; print('env ok')"
```

实测版本：Python 3.10.21 / fasttext 0.9.x / jieba / flask / streamlit 1.x

---

## 3. 目录结构

```
day03_fastext/
├── README.md                  ← 本文件
├── requirements.txt
├── data/
│   ├── raw/                   ← 原始数据（180k/10k/10k，10.5 MB，不入库）
│   │   ├── train.txt  dev.txt  test.txt   class.txt
│   └── processed/             ← 预处理产物（32 MB，不入库，可随时重生成）
│       ├── train_word.txt  dev_word.txt  test_word.txt
│       └── train_char.txt  dev_char.txt  test_char.txt
├── models/                    ← 模型权重（20 GB，不入库）
│   ├── fasttext_char.bin               1.9 MB   dev 0.8716
│   ├── fasttext_word.bin               47.1 MB  dev 0.8999
│   ├── fasttext_word_tuned.bin         100.2 MB dev 0.9012  ← 当前默认
│   ├── fasttext_char_autotune.bin      8.05 GB  dev 0.9118 / 加载 45 s
│   └── fasttext_word_autotune.bin      11.65 GB dev 0.9130 / 加载 48 s
└── src/
    ├── config.py              ← 03-0 配置中心（被 import，不加数字）
    ├── preprocess.py          ← 03-1 预处理
    ├── train.py               ← 03-2 训练（含 autotune）
    ├── evaluate.py            ← 03-3 评估
    ├── predictor.py           ← 03-4 预测器封装（被 api 导入）
    ├── api.py                 ← 03-5 Flask 接口（5001）
    └── app.py                 ← 03-6 Streamlit 前端
```

---

## 4. 运行顺序（按编号，全部在 `src/` 下执行）

```powershell
cd day03_fastext\src

python preprocess.py     # 03-1 生成 processed/*.txt（约 1 分钟，jieba 分词 20 万条）
python train.py          # 03-2 训练（默认只训 word/char 两个小模型）
python evaluate.py       # 03-3 评估，打印准确率 + 分类报告 + fastText 原生 test()
python api.py            # 03-5 起接口（保持窗口不关）
python app.py            # 03-6 新窗口起前端页面
```

`train.py` 的 `__main__` 里默认只跑 `train_char()` / `train_word()`；`auto_param_tuning()` 默认注释掉，**要跑请先看第 7 节**（会生成 8~11.65 GB 的模型）。

---

## 5. 数据格式（fastText 要求）

原始数据是 `文本<TAB>标签数字`，fastText 要求 `__label__类别名 分词后的文本`：

```
原文： 郑州马拉松今日正式开跑	7
转换： __label__sports 郑州 马拉松 今日 正式 开跑
```

转换逻辑在 `preprocess.py`：
- `mode='word'` → `jieba.lcut(text)`，用空格连接
- `mode='char'` → `list(text)`，逐字用空格连接（中文短文本常用兜底方案）
- 类别名从 `class.txt` 读取，**行号即标签编号**

---

## 6. 实测结果（2026-10-18，dev 10000 条）

| 模型 | 大小 | dim | bucket | epoch | wordNgrams | minn/maxn | dev 准确率 | 加载耗时 |
|---|---|---|---|---|---|---|---|---|
| `fasttext_char.bin` | 1.9 MB | 100 | 0 | 5 | 1 | 0/0 | 0.8716 | 秒开 |
| `fasttext_word.bin` | 47.1 MB | 100 | 0 | 5 | 1 | 0/0 | 0.8999 | 秒开 |
| `fasttext_word_tuned.bin` | 100.2 MB | 100 | 200,000 | 10 | 2 | 0/0 | **0.9012** | 秒开 |
| `fasttext_char_autotune.bin` | 8.05 GB | 216 | 10,000,000 | 4 | 3 | 0/0 | **0.9118** | 45.3 s |
| `fasttext_word_autotune.bin` | 11.65 GB | 309 | 10,000,000 | 2 | 1 | 2/5 | **0.9130** | 47.8 s |

> 超参不是猜的：直接读 `.bin` 文件头解出来的（`magic=793712314, version=12`），并用 fastText 官方 `getArgs()` 逐项校验过 12 个字段全部一致。
> `fasttext_word_tuned.bin` 是我用固定合理超参重训的替代方案；前两个是 `train.py` 默认超参的产物。

### 分类别表现（`fasttext_word.bin`）

| 类别 | 最差 | | 类别 | 最好 |
|---|---|---|---|---|
| stocks | 0.8336 F1 | | sports | 0.9606 F1 |
| science | 0.8420 F1 | | education | 0.9531 F1 |
| finance | 0.8854 F1 | | game | 0.9168 F1 |

**结论**：`stocks`（股票）、`science`（科技）两类最难分，因为它们的标题用词和 finance / society 高度重叠；`sports`、`education` 特征词最明显。这也是下一步换 BERT 时最该盯的两个类。

---

## 7. 关键发现：autotune 的收益与代价

`train.py` 里：

```python
model = fasttext.train_supervised(
    input=train_path,
    autotuneValidationFile=dev_path,
    autotuneDuration=300,      # 让它自己搜 5 分钟
    verbose=3,
)
```

跑完模型变成 **8.05 GB / 11.65 GB**，原因是 autotune 把搜索空间打满了：

| 项 | 默认 | autotune 选出来的 | 后果 |
|---|---|---|---|
| `bucket` | 0（minn=maxn=0 时不分配子词桶） | **10,000,000** | 输入矩阵行数 ×1000 万 |
| `dim` | 100 | **216 / 309** | 输入矩阵列数 ×2~3 |
| `epoch` | 5 | 4 / 2 | 反而更少 |
| `wordNgrams` | 1 | 3 / 1 | char 版抓了 3-gram |
| `minn/maxn` | 0/0 | 0/0 · **2/5** | word 版开了子词（缓解 OOV） |

输入矩阵 ≈ `bucket × dim × 4 字节` → 10,000,000 × 309 × 4 ≈ **11.5 GB**，和实际文件大小吻合。

### 实测：能加载，但很重

本机物理内存 15.8 GB、可用 3.4 GB。`fasttext.load_model()` 要把整份权重读进内存：

| 模型 | 结果 | 耗时 |
|---|---|---|
| `fasttext_char_autotune.bin`（8.05 GB） | ✅ 加载成功，dev **0.9118** | 45.3 s |
| `fasttext_word_autotune.bin`（11.65 GB） | ✅ 加载成功，dev **0.9130** | 47.8 s |

不是"打不开"，而是**靠页面文件扛过去的**：加载后内存基本被占满，再同时跑浏览器/其他程序就容易卡。所以它属于"能跑，但不适合当日常服务"。

### 收益 vs 代价

| 对比 | 准确率 | 体积 |
|---|---|---|
| 默认 word（47.1 MB） | 0.8999 | 1× |
| 固定超参 tuned（100.2 MB） | 0.9012（**+0.13**） | 2× |
| autotune word（11.65 GB） | 0.9130（**+1.31**） | **248×** |

**多 1.3 个点，体积涨 248 倍、加载 48 秒。** 学习/演示阶段用 tuned；要冲最高分、且机器有 16 GB 以上内存时再用 autotune。

### 两条修法

1. **固定超参**（推荐日常用）：

```python
model = fasttext.train_supervised(
    input=str(train_path),
    dim=100, epoch=10, lr=0.5, wordNgrams=2,   # wordNgrams=2 抓二元短语
    bucket=200000, minCount=2, loss='softmax',
)
```

→ 100.2 MB，dev **0.9012**。

2. **量化压缩**（想既要 0.913 又要小体积时）：对 autotune 模型调 `model.quantize()`，fastText 官方说明是**体积可降到 1/10 左右、准确率基本不掉**。11.65 GB → 约 1.2 GB，day04 之后可以拿这个做对比实验。


---

## 8. 当前用哪个模型：`config.active_model_name`

原来 `evaluate.py` 指向 `word_autotune`、`api.py` 指向 `char_autotune`，两边评估的不是同一个模型，结论自然对不上。现在统一收敛到 `config.py`：

```python
# 当前启用哪个模型（evaluate / api / app 共用）
self.active_model_name = "fasttext_word_tuned.bin"
# 与模型配套的预处理模式：word / char
self.active_model_mode = "word"
```

想换模型只改这两行。可选值：

| 文件 | 大小 | 模式 | dev | 说明 |
|---|---|---|---|---|
| `fasttext_word_tuned.bin` | 100.2 MB | word | 0.9012 | **默认：体积和效果最平衡** |
| `fasttext_word.bin` | 47.1 MB | word | 0.8999 | 更省内存的备选 |
| `fasttext_char.bin` | 1.9 MB | char | 0.8716 | 极小，适合演示 |
| `fasttext_word_autotune.bin` | 11.65 GB | word | **0.9130** | 最高分，加载 48 s、吃 11.65 GB 内存 |
| `fasttext_char_autotune.bin` | 8.05 GB | char | 0.9118 | 同上，稍轻一点 |

---

## 9. 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: fasttext` | 用 `fasttext_env` 环境，不是 `tensorflow_env` |
| `load_model` 要等 45~48 秒、内存被占满 | 你在加载 `*_autotune.bin`（8~11.65 GB）。日常开发改用 `config.active_model_name` 指向的 tuned 模型 |
| `ValueError: ... cannot be opened for saving` | 模型目录不可写，或路径含非法字符 |
| 接口返回 `400 text is required` | 请求体必须是 `{"text": "..."}` 且非空；缺字段 / `null` / 空串都会被拒（本次已加校验） |
| `preprocess.py` 报 `Invalid mode` | 只支持 `word` / `char` 两种 |
| 前端连不上 | 先起 `api.py`（5001），再起 `app.py` |

---

## 10. 下一步

- [ ] 对 `fasttext_word_autotune.bin` 做 `model.quantize()`，看能否 1.2 GB 左右保持 0.91
- [ ] `wordNgrams=2` 之外再试 `3`，看是否还涨
- [ ] 加 `loss='ova'` 对比
- [ ] 把训练好的模型接到 `api.py`，实测端到端延迟（首请求 vs 预热后）
- [ ] 与 day04 的 BERT 做横向对比（同数据、同评估口径）
- [ ] 确认不再需要后，删掉两个 20 GB 的 autotune 模型腾空间
