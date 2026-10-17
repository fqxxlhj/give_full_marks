# v1.0.0 — 首个可运行版本：数据 → EDA → 随机森林 → API → 前端 全链路打通

**发布日期**：2026-10-17
**Tag**：`v1.0.0`

---

## 一、这个版本是什么

第一个跑通完整链路的版本。一句中文新闻标题，经过「清洗 → 分词 → TF-IDF → 随机森林 → Flask 接口 → Streamlit 页面」，输出 10 个新闻类别之一。

- 分类类别：finance / realty / stocks / education / science / society / politics / sports / game / entertainment
- dev 集（10,000 条）准确率：**0.7306**（随机森林，训练用了 train 的前 20,000 条）
- 模型文件：`rf_model.pkl` 约 169 MB，`tfidf_vectorizer.pkl` 约 0.7 MB（不入库，本地生成）

---

## 二、今日（2026-10-17）完成的步骤

### 1. 工程结构重整（本次重点）

原来的 `src/` 里 11 个 py 混在一起，分不清「正式流程 / 测试 / 学习脚本」。今天按**步骤编号**重排，并把学习和测试代码物理隔离到 `预习/`：

| 原文件 | 新位置 | 说明 |
|---|---|---|
| `src/EDA分析.py` | `预习/01_EDA分析.py` | 预习：脚本版 EDA |
| `src/eda.py` | `预习/02_EDA函数封装.py` | 预习：函数版 EDA + JSON 报告 |
| `src/数据处理.py` | `src/03_数据处理.py` | 正式：数据处理 |
| `src/train.py` | `src/04_模型训练.py` | 正式：训练 + 评估 + 保存 |
| `src/predict.py` | `src/05_模型预测评估.py` | 正式：dev 集评估 |
| `src/predict_api.py` | `src/06_预测API服务.py` | 正式：Flask 接口 |
| `src/app.py` | `src/07_前端页面.py` | 正式：Streamlit 页面 |
| `src/flask_api.py` | `预习/90_Flask入门测试.py` | 测试：Flask 最小 demo |
| `src/predict_test.py` | `预习/91_API接口测试.py` | 测试：接口联调 |
| `src/config.py` | `src/config.py`（保持） | 公共模块，被 import，不能加数字前缀 |
| `src/predict_fun.py` | `src/predict_fun.py`（保持） | 同上 |

> 为什么 `config.py` 和 `predict_fun.py` 不加数字：它们被别的脚本 `import`，而 Python 的模块名不能以数字开头（`import 04_config` 是语法错误）。规则已写进 README 第 7 节。

### 2. 新增 `预习/` 目录

- 移入 `01`、`02` 两个 EDA 脚本和 `90`、`91` 两个测试脚本
- 复制**整套课件原始代码**到 `预习/课件参考_TMF_Code/`（33 个文件，约 226 MB，含课件自己的 data 和 model），用于逐段对照复习
- 新增 `预习/README.md`：目录用途、课件视频 ↔ 代码映射表、续学路线、待预习清单

### 3. 让脚本不再依赖「当前工作目录」

- `预习/01_EDA分析.py`：`../data/train.txt` 等相对路径 → 改为基于 `PROJECT_ROOT` 的绝对路径
- `src/03_数据处理.py`：同上，6 个路径常量全部改为 `PROJECT_ROOT` 派生

**原因**：PyCharm 点运行和命令行 `python xx.py` 的工作目录可能不同，原来换个方式运行就 `FileNotFoundError`；改完后在任何目录运行都能找到数据。

### 4. 每个脚本加步骤表头

每个文件第一行写明「步骤 0X｜做什么」，打开就知道自己在流程的哪一环。

### 5. 修复 Windows 控制台乱码崩溃

数据里有 33 条含乱码字符（`U+FFFD`）的标题，在 GBK 编码的 cmd 里 `print` 出来会直接 `UnicodeEncodeError` 崩掉（实测 `01_EDA分析.py` 崩溃）。已在 9 个脚本开头统一加：

```python
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
```

PyCharm 里本来不报错，但这个改动让「命令行 / PyCharm / 计划任务」三种跑法都安全。

### 5. 文档与仓库

- `README.md`：从零开始的 11 个步骤（每步含做法/命令/产物/验收）、一键跑通命令、命名规范、常见问题
- `requirements.txt`：依赖清单 + 实测版本
- `.gitignore`：排除数据 CSV、原始 txt、模型 pkl、IDE 目录
- 建立 GitHub 公开仓库 `give_full_marks` 并打 tag

---

## 三、今日验收结果

| 检查项 | 结果 |
|---|---|
| 11 个 py 脚本语法校验 | ✅ 全部通过 |
| `预习/01_EDA分析.py` 实跑 | ✅ 端到端跑完（GBK 控制台不再崩） |
| `预习/02_EDA函数封装.py` 实跑 | ✅ 生成 `eda_summary.json`，每类 1000 条均衡 |
| `05_模型预测评估.py` 实跑 | ✅ dev 准确率 0.7306 |
| `06_预测API服务.py` + `91_API接口测试.py` | ✅ 返回 `{'label': 'entertainment'}` |
| `07_前端页面.py`（Streamlit） | ✅ 页面 HTTP 200 正常启动 |
| 数据处理输出字段 | ✅ `text, label, words`，`words` ≤ 30 词，`label` ∈ [0,9] |
| 目录结构 | ✅ `src/` 正式流程 7 个文件 + `预习/` 学习区 |
| 仓库体积 | ✅ 42 个文件 / 450 KB（数据与模型已排除） |
| 文档 | ✅ README / 预习 README / 本发布说明 |

---

## 四、已知问题

1. **训练只用了 2 万条**（`04_模型训练.py` 里 `[:20000]`），18 万条全量还没跑，准确率有上升空间。
2. **模型 169 MB**，超出 GitHub 单文件 100 MB 限制，所以 `artifacts/model/` 被 gitignore —— 克隆仓库后必须自己跑一次 `04_模型训练.py`。
3. **随机森林是基线**，没有做参数搜索，`n_estimators` 用默认值。
4. 前端只做了单条标题输入，没有做批量预测和置信度展示。

---

## 五、下一步计划

- [ ] 训练集放开到全量 18 万条，记录准确率变化
- [ ] `GridSearchCV` 调参（`max_depth`、`n_estimators`、`min_samples_leaf`）
- [ ] 增加 baseline 对比：朴素贝叶斯、逻辑回归、LinearSVC、FastText
- [ ] 换 BERT 做对比（`03_NLP/day05` 的 AutoModel）
- [ ] 模型压缩（对应课件 `04-模型压缩.pdf`）
- [ ] 前端增加置信度和 Top-3 类别
