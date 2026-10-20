# v1.1.0 — FastText 全流程落地 + BERT 起步

**发布日期**：2026-10-18
**Tag**：`v1.1.0`
**基线**：`v1.0.0`（2026-10-17，TF-IDF + 随机森林，dev 0.7306）

> 本版本是**在 v1.0.0 之上新增**，不修改、不覆盖 `v1.0.0` 的提交与 Release。
> 昨天的 `src/` 主流程原样保留，今天的工作独立放在 `03_fastext/` 和 `04_BERT/` 两个子项目里。

---

## 一、这个版本是什么

一句话：**把模型从「TF-IDF + 随机森林」换成 FastText，准确率从 0.7306 提到 0.9012（最优 0.9130），并把 BERT 的数据管线搭起来。**

| 版本 | 模型 | dev 准确率 | 模型体积 | 训练耗时 |
|---|---|---|---|---|
| v1.0.0 | TF-IDF + 随机森林 | 0.7306 | 169 MB | 分钟级 |
| **v1.1.0（今天）** | **FastText 词级 + 固定超参** | **0.9012** | **100.2 MB** | **十几秒** |
| v1.1.0（今天，最优） | FastText autotune | 0.9130 | 11.65 GB | 5 分钟搜索 + 训练 |

**+17.06 个百分点**，模型体积反而更小，训练时间从"几分钟"降到"十几秒"。

---

## 二、今日（2026-10-18）完成的步骤

### 1. day03 · FastText 完整小项目（7 个脚本，全链路跑通）

| 步骤 | 文件 | 做了什么 | 状态 |
|---|---|---|---|
| 03-0 | `03_fastext/src/config.py` | 配置中心：数据/模型路径、当前启用的模型 | ✅ |
| 03-1 | `preprocess.py` | 预处理：`文本<TAB>标签` → `__label__类别名 分词串`，支持词级(jieba)/字符级 | ✅ |
| 03-2 | `train.py` | 训练：默认超参 + autotune 自动调参 | ✅ |
| 03-3 | `evaluate.py` | 评估：sklearn 指标 + fastText 原生 `test()` | ✅ |
| 03-4 | `predictor.py` | 预测器封装：`FastTextPredictor.predict()` → `{label, probability}` | ✅ |
| 03-5 | `api.py` | Flask 接口：`/health`、`/`、`POST /predict`（5001） | ✅ |
| 03-6 | `app.py` | Streamlit 前端：输入框 + 置信度进度条 | ✅ |

工程结构：`data/raw`（原始）→ `data/processed`（预处理产物）→ `models/`（权重）三层分离，路径全部走 `config.py`，不写死相对路径。

### 2. 四种做法横向对比（今天最核心的产出）

| 做法 | dev 准确率 | 模型体积 | 结论 |
|---|---|---|---|
| 字符级 + 默认超参 | 0.8716 | 1.9 MB | 省事但最弱 |
| 词级 + 默认超参 | 0.8999 | 47.1 MB | word 比 char 高 2.8 个点 |
| **词级 + 固定合理超参** | **0.9012** | **100.2 MB** | ← 当前默认，性价比最高 |
| autotune（char / word） | 0.9118 / **0.9130** | 8.05 / 11.65 GB | 最高分，但体积 ×100~250 |

### 3. 一个踩到的坑：autotune 会把模型撑到 GB 级

`autotuneDuration=300` 让 fastText 自己搜超参，搜出来的组合是 **`bucket=10,000,000` + `dim=216/309`**，输入矩阵 ≈ `bucket × dim × 4 字节` ≈ 8~11.5 GB：

| 模型 | 大小 | 加载耗时 | dev |
|---|---|---|---|
| `fasttext_char_autotune.bin` | 8.05 GB | 45.3 s | 0.9118 |
| `fasttext_word_autotune.bin` | 11.65 GB | 47.8 s | 0.9130 |

**能加载**（靠页面文件扛过去），但加载完内存基本占满，不适合当日常服务。所以：
- 日常开发/演示：用 100 MB 的固定超参版本（0.9012）
- 冲最高分：用 autotune 版本（0.9130），机器需 16 GB 以上内存
- 后续优化方向：对 autotune 模型做 `model.quantize()`，体积可降到约 1/10

实测数据不是估计值：`.bin` 文件头按 fasttext 二进制格式解析出超参（`magic=793712314, version=12`），并用官方 `getArgs()` 逐项校验 12 个字段全部一致。

### 4. day04 · BERT 起步（数据管线完成）

| 步骤 | 文件 | 状态 |
|---|---|---|
| 04-0 | `04_BERT/src/config.py` | ✅ 路径 + 预训练模型 + 训练超参（batch 16 / max_len 64 / lr 2e-5 / warmup 0.1 / early stop patience 2）+ 设备自动选择 |
| 04-1 | `04_BERT/src/data.py` | ✅ `load_news_data` / `NewsDataset` / `collate_fn`（批量 tokenize + 动态 padding）/ `build_dataloader` |
| 04-2~04-6 | `train.py` / `evaluate.py` / `predictor.py` / `api.py` / `app.py` | ❌ 未开始（骨架已写进 `04_BERT/README.md`） |

**重要提醒**：本机是 `torch 2.14.1+cpu`，无 CUDA。18 万条 × 5 轮在 CPU 上约 **3 小时/轮**，建议先用前 1~2 万条跑通流程。

### 5. 仓库整理（按 v1.0.0 的同一套规则）

- **大文件全部隔离，绝不入库**：`*.bin`（20 GB）、`day*/data/`（42 MB）、`04_BERT/pretrained/`（1.24 GB）全部写进 `.gitignore`
- **统一模型选择**：原来 `evaluate.py` 用 `word_autotune`、`api.py` 用 `char_autotune`，两边评估的不是同一个模型；现收敛为 `config.active_model_name` + `config.active_model_mode`，换模型只改两行
- **步骤表头**：9 个脚本第一行统一注明 `# 步骤 03-x｜...`，打开就知道在流程哪一环
- **控制台编码保护**：沿用 v1.0.0 的 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，避免 GBK 控制台打印乱码字符崩溃
- **修掉一个硬编码**：`04_BERT/src/data.py` 里 tokenizer 路径 `'../pretrained/bert-base-chinese'` → `config.pretrained_model_path`
- **新增文档**：`03_fastext/README.md`（含实测表、autotune 坑的完整分析、常见问题）、`04_BERT/README.md`（进度表、数据管线说明、明天要写的骨架、验收标准）、两个 `requirements.txt`

---

## 三、今日验收结果

| 检查项 | 结果 |
|---|---|
| 9 个脚本语法 | ✅ 全部通过 |
| `preprocess.py` 实跑 | ✅ 生成 6 个 `processed/*.txt`（train 180k / dev 10k / test 10k，word + char 两套） |
| `train.py` 实跑 | ✅ 产出 4 个模型（默认 2 个 + autotune 2 个） |
| `evaluate.py` 口径 | ✅ 词级 dev **0.9012**、字符级 0.8716、autotune 0.9118 / **0.9130** |
| 模型体积对比 | ✅ 1.9 MB / 47.1 MB / 100.2 MB / 8.05 GB / 11.65 GB 全部实测 |
| `.bin` 超参解析 | ✅ 12 个字段与官方 `getArgs()` 完全一致 |
| 大文件隔离 | ✅ 模型 20 GB、数据 42 MB、预训练 1.24 GB 均未进 Git |
| `evaluate.py` 实跑 | ✅ 打印「评估模型：fasttext_word_tuned.bin（word）」，Accuracy **0.9012**，原生 test() 一致 |
| `api.py` 实跑 | ✅ `/health` → `{"status":"OK"}`；`/predict`「足球比赛国足获胜」→ **sports 0.9999** |
| `api.py` 参数校验 | ✅ 缺 `text` / `null` / 空串 / 全空格 → 统一 **HTTP 400** `text is required` |
| `app.py` 实跑 | ✅ Streamlit 页面 HTTP 200（6.4 KB） |

---

## 四、本次改动清单（相对 v1.0.0）

**新增**
- `03_fastext/`（`src/` 7 个脚本 + README + requirements）
- `04_BERT/`（`src/` 2 个脚本 + README + requirements）
- `docs/release-notes-v1.1.0.md`（本文件）

**修改**
- `.gitignore`：新增 `*.bin`、`day*/models/`、`day*/data/`、`04_BERT/pretrained/`、`**/data/processed/`
- `README.md`：新增「分天项目」章节和今日进度（原有 1~10 节内容保持不变，致谢顺延为第 12 节）
- `03_fastext/src/config.py`：新增 `active_model_name` / `active_model_mode`，统一模型选择
- `03_fastext/src/api.py`：预测器改用 `config.active_model_name`；补齐参数校验（缺 `text` / `null` / 空串 → **HTTP 400**，原来空 body 返回 200、`text:null` 直接 500）
- `03_fastext/src/evaluate.py`：`__main__` 改为从 config 取模型（原来写死 `word_autotune`，和 api 评估的不是同一个模型）
- `04_BERT/src/data.py`：tokenizer 路径 `'../pretrained/bert-base-chinese'` → `config.pretrained_model_path`
- 9 个脚本：统一加 `# 步骤 0X-Y｜` 表头 + `sys.stdout.reconfigure(encoding="utf-8")` 输出保护

**未改动**
- `src/` 下的 v1.0.0 主流程（`03_数据处理.py` ~ `07_前端页面.py`、`config.py`、`predict_fun.py`）一行没动
- `预习/` 目录、`data/`、`artifacts/` 均未改动

---

## 五、已知问题

1. **20 GB 的 autotune 模型**：精度只涨 1.3 个点，日常用不上，确认后可删（或先做量化压缩）。
2. **模型仍有离谱错判**：实跑「郑州马拉松今日正式开跑」被判成 `stocks`（0.97），「台风登陆」被判成 `politics`。FastText 是纯词袋模型，遇到训练集里少见的词组合就会瞎猜 —— 这正是下一步上 BERT（有上下文）的理由。
3. **day04 BERT 只到数据管线**：`train.py` 的骨架写在 README 里，还没落成代码。
4. **CPU 训练 BERT 太慢**：全量 5 轮不现实，需先小样本验证或换 GPU。
5. `04_BERT/pretrained/` 里 TF 和 Flax 两套权重（887 MB）用不到，可删。

---

## 六、下一步

- [ ] `04_BERT/src/train.py`：训练循环（warmup + early stopping + 保存 best_model）
- [ ] `04_BERT/src/evaluate.py`：与 FastText 同口径对比（目标 > 0.9130）
- [ ] `03_fastext`：起 `api.py` + `app.py` 实测端到端延迟
- [ ] autotune 模型 `quantize()` 压缩实验
- [ ] 根目录 README 补一张「随机森林 / FastText / BERT」三模型对比表
