# 投满分 · 新闻标题分类项目（give_full_marks）

> 一条新闻标题进，一个新闻类别出。技术栈：**jieba 分词 → TF-IDF → 随机森林 → Flask API → Streamlit 前端**。
>
> 当前版本：**v1.0.0**（2026-10-17）｜dev 集准确率：**0.7306**

---

## 1. 这个项目在做什么

输入一句中文新闻标题（例如「清华大学 2027 年硕士研究生招生目录」），输出它属于下面 10 个类别中的哪一类：

| 标签 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 类别 | finance | realty | stocks | education | science | society | politics | sports | game | entertainment |

为什么用 **TF-IDF + 随机森林** 做基线：训练几分钟就能跑完、不需要显卡、结果可解释（能看特征词），非常适合先把「数据 → 模型 → 接口 → 页面」整条链路打通，再考虑换成 BERT / 大模型。

---

## 2. 数据集说明

| 文件 | 内容 | 规模 |
|---|---|---|
| `data/train.txt` | 训练集，格式 `新闻标题<TAB>标签` | 180,000 行 |
| `data/dev.txt` | 验证集，同上 | 10,000 行 |
| `data/test.txt` | 测试集，同上 | 10,000 行 |
| `data/class.txt` | 类别名称，一行一个，**行号即标签编号** | 10 行 |
| `data/stopwords.txt` | 停用词表 | 810 个 |

原始 TXT 无表头、Tab 分隔。步骤 03 会把它清洗成 CSV：

| 处理后文件 | 字段 |
|---|---|
| `data/train.csv` / `dev.csv` / `test.csv` | `text`（清洗后原文）、`label`（0-9）、`words`（jieba 分词后、最多 30 个词，空格分隔） |

> `data/` 里的 CSV 和 `artifacts/model/*.pkl` 体积大（模型 169 MB），**不入 git 仓库**，按下面步骤自己生成即可。

---

## 3. 环境准备（从零）

**Python 3.10**（本项目实测环境：`D:\AnacondaEnvs\tensorflow_env`，Python 3.10.22）。

```powershell
# 1) 建虚拟环境（或用你自己的 conda 环境）
conda create -n give_full_marks python=3.10 -y
conda activate give_full_marks

# 2) 装依赖
pip install -r requirements.txt

# 3) 自检：能 import 就说明环境通了
python -c "import pandas, sklearn, jieba, flask, streamlit, seaborn; print('env ok')"
```

依赖清单见 `requirements.txt`。实测版本：pandas 2.3.3 / scikit-learn 1.7.2 / streamlit 1.65.0。

---

## 4. 目录结构

```
give_full_marks/
├── README.md                 ← 你正在看的这份：从零开始的开发步骤
├── requirements.txt
├── data/                     ← 数据（CSV 与 txt 不入库）
│   ├── train.txt  dev.txt  test.txt   class.txt  stopwords.txt
│   └── train.csv  dev.csv  test.csv   （步骤 03 生成）
├── artifacts/                ← 产物
│   ├── eda/                      分布图示例（标签分布图.png、文本长度分布图.png）
│   └── model/                    rf_model.pkl、tfidf_vectorizer.pkl（不入库）
├── src/                      ← 正式流程代码（按步骤编号，可长期使用）
│   ├── config.py                  配置中心：统一管理所有路径（步骤 04.0）
│   ├── predict_fun.py             预测函数封装：一句标题 → 类别名（步骤 06.0）
│   ├── 03_数据处理.py              清洗 + 分词 + 截断 → 三个 CSV
│   ├── 04_模型训练.py              TF-IDF + 随机森林 + 评估 + 保存 pkl
│   ├── 05_模型预测评估.py           加载 pkl，对 dev 集算准确率
│   ├── 06_预测API服务.py           Flask /predict 接口（端口 8001）
│   └── 07_前端页面.py              Streamlit 页面（调用 8001 接口）
├── 预习/                     ← 预习与测试：学习、试错、复习用的地方
│   ├── 01_EDA分析.py               脚本版 EDA（对应课件 01_EDA分析.py）
│   ├── 02_EDA函数封装.py            函数版 EDA + JSON 报告（对应课件 02_EDA函数封装.py）
│   ├── 90_Flask入门测试.py          Flask 最小 demo（练习用）
│   ├── 91_API接口测试.py            接口联调测试脚本
│   ├── README.md                   预习目录说明与续学路线
│   └── 课件参考_TMF_Code/          整套课件原始代码副本（对照用，勿直接改）
└── docs/
    └── release-notes-v1.0.0.md    版本发布说明
```

---

## 5. 从零开始的完整开发步骤

> 每一步都写了：**做什么 / 怎么做 / 命令 / 产物 / 验收标准**。按 00 → 08 顺序做，中途不要跳步。
> `01`、`02` 在 `预习/`，`03` 以后在 `src/`。步骤号与文件名前缀**严格一一对应**。

### 步骤 00 · 建工程骨架

- 建目录：`data/`、`artifacts/eda/`、`artifacts/model/`、`src/`、`预习/`
- 数据放到 `data/`：`train.txt`、`dev.txt`、`test.txt`、`class.txt`、`stopwords.txt`
- 验收：`class.txt` 有 10 行、`train.txt` 有 18 万行且是 Tab 分隔

### 步骤 01 · EDA 分析 → `预习/01_EDA分析.py`（看清数据长什么样）

**做什么**：读数据 → 看结构 → 统计标签分布 → 统计文本长度 → 查数据质量（空文本、非法标签、重复、同文异标、乱码）→ 检查 train/dev/test 之间是否"数据泄漏"→ 画图。

**怎么跑**：

```powershell
cd 预习
python 01_EDA分析.py
```

**产物**：`artifacts/eda/标签分布图.png`、`artifacts/eda/文本长度分布图.png`

**验收 / 结论（本项目实测数据）**：
- **类别完全均衡**：dev 与 test 各 10000 条，每类正好 1000 条（10.00%）→ 不需要重采样，`average='macro'` 与 accuracy 口径接近
- **文本长度**：均值 19.1 字，中位数 19，P90=24，**P95=25**，P99=27，最长 33 字 → 所以"按词截断到 30"是安全且几乎无损的选择
- **数据质量**：空文本 0 条；同文异标 33 条（需人工复核，占比可忽略）
- **无数据集泄漏**：train/dev 交叉 91 条、train/test 交叉 108 条、dev/test 交叉 8 条，占比均 < 0.1% → 三集可直接使用，不必重新划分

### 步骤 02 · EDA 函数封装 → `预习/02_EDA函数封装.py`（把脚本变成可复用代码）

**做什么**：把步骤 01 的一整段流程拆成函数：`load_classes` / `load_dataset` / `analyze_split` / `check_split_overlap` / `draw_*`，并输出机器可读的 `eda_summary.json`。

**怎么跑**：

```powershell
cd 预习
python 02_EDA函数封装.py
```

**产物**：`artifacts/eda/eda_summary.json`（含每个 split 的样本数、类别分布、长度分位数、重复/乱码统计）+ `label_distribution.png`、`text_length_distribution.png`

**验收**：终端打印 train/dev/test 三段统计 + 三组两两重复数；JSON 能正常用编辑器打开。
**为什么要做这一步**：脚本改一个参数就要重跑全部；函数化之后，换个数据集只改一个字典就行 —— 这是"能维护"和"只能跑一次"的分界线。

### 步骤 03 · 数据处理 → `src/03_数据处理.py`（生成训练用的 CSV）

**做什么**：读原始 TXT → 统一字段名 → 删空值/非法标签/越界标签 → 文本清洗（HTML 标签、换行、零宽字符、乱码、连续空白）→ jieba 分词 → 按词数截断到 30 → 校验字段和标签范围 → 保存 CSV。

**怎么跑**：

```powershell
cd src
python 03_数据处理.py
```

**产物**：`data/train.csv`、`data/dev.csv`、`data/test.csv`，字段固定为 `text, label, words`

**验收**：终端打印"字段验证通过"；CSV 里 `words` 每条不超过 30 个词；`label` 全在 0~9。
**坑**：路径已改成基于 `PROJECT_ROOT` 的绝对路径，所以**在任何目录下运行都不会找不到文件**。

### 步骤 04 · 模型训练 → `src/04_模型训练.py`

**4.0 先写配置中心 `config.py`**：把所有路径收敛到一个类里（数据、停用词、类别、模型保存位置），其它脚本只 `from config import Config`，不再硬编码 `../data/xxx`。

```python
from pathlib import Path

class Config:
    def __init__(self):
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        self.train_path = PROJECT_ROOT / 'data' / 'train.csv'
        ...
```

> `config.py` 不带数字前缀：它被 04/05/06 三个脚本 `import`，而 Python 模块名不能以数字开头。

**验收**：`python -c "from config import Config; c=Config(); print(c.train_path)"` 打印出真实存在的路径。

**4.1 训练**：读 `train.csv` → 用停用词表建 `TfidfVectorizer` 把 `words` 转成稀疏向量 → 8:2 划分训练/测试 → 训练 `RandomForestClassifier` → 用 accuracy / precision / recall / F1（macro）评估 → 保存模型和向量化器两个 pkl。

**怎么跑**：

```powershell
cd src
python 04_模型训练.py
```

**产物**：`artifacts/model/rf_model.pkl`（约 169 MB）、`artifacts/model/tfidf_vectorizer.pkl`

**验收**：打印出 `X.shape`、分类报告，两个 pkl 文件生成。
**关键点**：
- `average='macro'` 是宏平均，10 个类别**平权**；类别不均衡时它比 accuracy 更诚实
- **向量化器必须和模型一起保存**，否则线上预测时分词口径不一致，准确率会莫名下降
- 训练集只用了前 2 万条（`[:20000]`）跑通流程，想提升就把这一行切片去掉

### 步骤 05 · 模型预测评估（用 dev 集体检） → `src/05_模型预测评估.py`

**做什么**：加载两个 pkl → 读 `dev.csv` → `transform`（注意是 transform 不是 fit_transform）→ `predict` → 和真实标签比准确率。

**怎么跑**：

```powershell
cd src
python 05_模型预测评估.py
```

**产物**：终端输出预测样例和 `dev 准确率`。
**当前结果**：**0.7306**（`dev` 1 万条）。
**为什么用 dev 而不是 test**：dev 用来调参、看效果；test 只在最后验收用一次，避免"对着测试集调参"。

### 步骤 06 · 预测 API 服务 → `src/06_预测API服务.py`

**6.0 先把预测封装成函数 `predict_fun.py`**：把"一句原始标题 → 类别名"封装成函数，供接口调用：
分词（同样截断 30 词）→ `tfidf.transform` → `model.predict` → 用 `class.txt` 把 0~9 映射成类别名。

```python
def predict_fun(data):
    word = ' '.join(jieba.lcut(data['text'])[:30])
    x = tfidf_model.transform([word])
    pred = rf_model.predict(x)[0]
    data['label'] = id2class[pred]
    return data
```

**验收**：`python predict_fun.py` 能对内置样例输出一个类别名。

**6.1 Flask 接口（端口 8001）**：用 Flask 暴露 `POST /predict`，收 `{"text": "..."}`，返回 `{"text": ..., "label": "sports"}`；参数缺失返回 400。

**怎么跑**（**要单独开一个终端，保持运行**）：

```powershell
cd src
python 06_预测API服务.py
# 另开终端验证：
curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d "{\"text\":\"这个景点太拉了\"}"
```

**验收**：返回 200 和 JSON；把 `text` 去掉再请求，返回 400 `{"error": "Invalid data"}`。

### 步骤 07 · 前端页面 → `src/07_前端页面.py`

**做什么**：输入框 + 预测按钮，把文本 POST 给 8001 接口，把返回的类别显示成绿色成功提示；后端没起时给出友好报错。

**怎么跑**：

```powershell
cd src
python 07_前端页面.py        # 脚本里已做 bare mode 兼容，会自动转成 streamlit run
```

**验收**：浏览器自动打开页面，输入标题点"预测"，显示出类别。

### 步骤 08 · 联调与测试 → `预习/90_Flask入门测试.py`、`预习/91_API接口测试.py`

- `预习/90_Flask入门测试.py`：Flask 最小 demo（两个路由），用来确认 `flask` 装好了、端口能通
- `预习/91_API接口测试.py`：向 8001 发一条 POST，检查状态码与返回体

**联调顺序（必须按这个顺序）**：
1. 终端 A：`python 06_预测API服务.py`（后端）
2. 终端 B：`python 07_前端页面.py`（前端）
3. 终端 C：`python 91_API接口测试.py`（可选，验证接口）

**验收**：前端页面能出结果；`91` 脚本打印出 `{'text': ..., 'label': ...}`。

---

## 6. 一次性跑通全流程

```powershell
cd D:\AI_model_tudy\give_full_marks\src

# ① 数据（1~2 分钟，jieba 分词 20 万条）
python 03_数据处理.py

# ② 训练（几分钟，生成 169MB 的 pkl）
python 04_模型训练.py

# ③ 体检
python 05_模型预测评估.py

# ④ 起后端（保持窗口不关）
python 06_预测API服务.py

# ⑤ 新开窗口起前端
python 07_前端页面.py
```

---

## 7. 命名规范（为什么有的文件带数字、有的不带）

| 规则 | 例子 | 原因 |
|---|---|---|
| 按执行顺序加两位数字前缀 | `03_数据处理.py`、`04_模型训练.py` | 一眼看出先后，复习时按序读 |
| **数字前缀只加在"能直接运行"的脚本上** | `03`~`07` | 直接 `python xx.py` 即可 |
| **被 `import` 的公共模块不加数字** | `config.py`、`predict_fun.py` | Python 的模块名不能以数字开头，`import 04_config` 是语法错误 |
| 测试/练习统一放 `预习/`，用 90+ 编号 | `90_Flask入门测试.py`、`91_API接口测试.py` | 和正式流程物理隔离，不会被误当生产代码 |

---

## 8. 当前进度

| 步骤 | 内容 | 对应文件 | 状态 |
|---|---|---|---|
| 01 | EDA 分析 | `预习/01_EDA分析.py` | ✅ 完成 |
| 02 | EDA 函数封装 | `预习/02_EDA函数封装.py` | ✅ 完成 |
| 03 | 数据处理 | `src/03_数据处理.py` | ✅ 完成 |
| 04 | 配置中心 + 模型训练 | `src/config.py`、`src/04_模型训练.py` | ✅ 完成（2 万条样本） |
| 05 | dev 评估 | `src/05_模型预测评估.py` | ✅ 准确率 0.7306 |
| 06 | 预测函数 + Flask API | `src/predict_fun.py`、`src/06_预测API服务.py` | ✅ 完成 |
| 07 | Streamlit 前端 | `src/07_前端页面.py` | ✅ 完成 |
| 08 | 联调测试 | `预习/90`、`预习/91` | ✅ 完成（实测返回 `entertainment`） |

版本历史见 `docs/`，每个版本一个文件。

---

## 9. 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: No module named 'config'` | 在 `src/` 目录下运行，别在项目根目录跑 |
| `模型文件不存在` / `FileNotFoundError: rf_model.pkl` | 先跑 `04_模型训练.py` 生成模型 |
| 前端点预测报「请确认后端 predict_api.py 已启动」 | 先起 `06_预测API服务.py`，端口 8001 |
| `Address already in use` | 8001 被占用：`netstat -ano \| findstr 8001` → `taskkill /PID xxx /F` |
| 图表中文变方块 | matplotlib 缺中文字体，装 SimHei / 微软雅黑，或在 `rcParams` 里换已有字体 |
| 命令行跑脚本报 `UnicodeEncodeError: 'gbk' codec can't encode character '\ufffd'` | 数据里有乱码字符，GBK 控制台打不出来。本项目 9 个脚本已加 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`；**你自己新写的脚本也要加这两行** |
| `01_EDA分析.py` 跑完不弹图 | 脚本用了 `matplotlib.use('TkAgg')`，服务器/无桌面环境请改成 `Agg` 或注释掉 `plt.show()` |
| 训练到一半内存不足 | 训练集切片 `[:20000]` 改更小，或把 `n_estimators` 调到 50 |

---

## 10. 后续计划

- [ ] 训练集从 2 万条放开到 18 万条，看准确率天花板
- [ ] 加 `GridSearchCV` 调 `max_depth` / `n_estimators`
- [ ] 对比逻辑回归、朴素贝叶斯、LinearSVC 基线
- [ ] 换 BERT（`03_NLP/day05` 的 AutoModel）做对比实验
- [ ] 模型压缩（对应课件 `04-模型压缩.pdf`）
- [ ] 用 Docker 打包 API + 前端

---

## 11. 致谢

数据集与项目框架来自投满分（TMF）新闻分类课程；`预习/课件参考_TMF_Code/` 是课程原始代码副本，仅用于对照学习，**不要直接在那里改代码**，改动请落在 `src/` 或 `预习/` 的新文件里。
