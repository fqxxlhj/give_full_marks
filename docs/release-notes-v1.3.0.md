# v1.3.0 — LLM 零样本闭环（dev 0.8000）+ BERT INT8 量化（体积 2.67×更小）

**发布日期**：2026-10-21
**Tag**：`v1.3.0`
**基线**：`v1.2.0`（2026-10-20，BERT dev 0.9411 / test 0.9441）

> 本版本**在 v1.2.0 之上新增**，`v1.0.0` / `v1.1.0` / `v1.2.0` 的提交与 Release 均未改动。
> `src/`（随机森林）、`03_fastext/`、`04_BERT/` 的代码一行没动（仅 04 的模型被 06 复制引用，未修改）。

---

## 一、这个版本是什么

一句话：**把"要不要上大模型"和"要不要压缩模型"这两个问题，用实测数据回答了。**

| 问题 | 今天的答案 | 依据 |
|---|---|---|
| 数据充足时，要不要用大模型零样本？ | **不用**：0.8000 < BERT 0.9411（差 14 个点），且慢 20~30 倍、按 token 付费 | `05_LLM` 500 条实测 |
| 390 MB 的 BERT 能不能压？ | **能**：INT8 后 146 MB（2.67×），精度 0.9447 → 0.9448 **完全不变** | `06_model_quantization` 实测 |
| 压缩能省时间吗？ | **几乎不能**：CPU 上 p50 只从 746 ms 降到 663 ms（1.13×） | 同上 |

---

## 二、今日（2026-10-21）完成的步骤

### 1. `05_LLM` 闭环（5 个新脚本）

| 步骤 | 文件 | 做了什么 | 状态 |
|---|---|---|---|
| 05-3 | `05_LLM/src/classifier.py`（原名 `classifer.py`） | 单条分类：调模型 → `json.loads` → **类别白名单校验** → 返回 reason + token 用量 | ✅ |
| 05-4 | `src/predict_one.py` | 批量预测：500 条抽样、**断点续跑**、逐条落盘、失败也记一行 | ✅ |
| 05-5 | `src/evaluate.py`（原名 `evalute.py`） | 读结果 CSV → accuracy / macro-P / macro-R / 分类报告 | ✅ |
| 05-6 | `src/api.py` | Flask：`/health`、`/predict`（端口 5003） | ✅ |
| 05-7 | `src/app.py` | Streamlit 前端，调 5003 接口 | ✅ |

### 2. LLM 零样本实测结果（500 条 dev 抽样，`random_state=42`）

**accuracy 0.8000 ｜ macro-F1 0.7976 ｜ 错 100/500**

| 最好的类 | F1 | 最差的类 | F1 |
|---|---|---|---|
| sports | 0.9720 | stocks | **0.5682** |
| game | 0.9136 | finance | **0.5905** |
| education | 0.9009 | politics | **0.6585** |

**错误高度集中，只错在两处**：

| 真实 → 预测 | 条数 | 占错例 |
|---|---|---|
| finance → stocks | 17 | 17% |
| politics → society | 16 | 16% |
| stocks → finance | 15 | 15% |

`finance ↔ stocks` 合计 32 条（32%）、`politics → society` 16 条（16%）。被判成 society 的 83 条里只有 59 条真是 society —— **模型把"政策性社会新闻"一律归到 society**。

**延迟与成本**：平均 877 ms/条（p50 882 / p95 1155），每条 339 prompt + 28 completion token，500 条共 183,630 token，外推 1 万条约 **3.67M token**。

### 3. `06_model_quantization` 新项目（6 个脚本 + 1 个 demo）

| 步骤 | 文件 | 做了什么 | 状态 |
|---|---|---|---|
| 06-0 | `06_model_quantization/src/config.py` | `@dataclass` 配置：模型/数据/INT8 输出路径 + batch 32 + max_length 128 + warmup 20 + 100 次计时 | ✅ |
| 06-1 | `src/quantize.py`（原名 `quanzite.py`） | `torchao` INT8 动态量化，保存 `model_state.pt` | ✅ |
| 06-2 | `src/dataset.py` | 数据管线（Dataset / collate_fn / DataLoader） | ✅ |
| 06-3 | `src/evaluate.py` | FP32 vs INT8 精度对比（各跑 1 万条） | ✅ |
| 06-4 | `src/benchmark.py` | 延迟基准：warmup 20 + 重复 100 次，算 p50/p95 与加速比 | ✅ |
| 06-5 | `src/model_compare.py` | 体积对比与压缩倍数 | ✅ |
| 06-6 | `test/01_快速演示模型量化.py` | 玩具模型最小 demo（两层全连接，3 分钟看懂量化前后结构） | ✅ |

**实测结果**：

| 指标 | FP32 | INT8 | 变化 |
|---|---|---|---|
| 体积 | 390.17 MB | 145.97 MB | **2.67× 更小** |
| test accuracy | 0.9447 | 0.9448 | ±0.0001（噪声） |
| test macro-F1 | 0.9448 | 0.9449 | ±0.0001 |
| p50（batch 32） | 746.6 ms | 663.4 ms | 1.13× |
| p95（batch 32） | 1201.9 ms | 820.0 ms | **1.47×** |

**为什么只快 13%**：序列只有 128（矩阵乘占比低）、LayerNorm/softmax/GELU 仍是 fp32、动态量化每层要现场算激活范围、CPU 上 int8 算力优势不如 GPU 明显。

### 4. 仓库整理（延续前两轮规则）

- **改掉 3 个拼写错误的文件名**（都会被 import 或直觉找不到）：
  - `05_LLM/src/classifer.py` → `classifier.py`（`predict_one.py`、`api.py` 的 import 同步修改）
  - `05_LLM/src/evalute.py` → `evaluate.py`
  - `06_model_quantization/src/quanzite.py` → `quantize.py`
- **步骤表头 + UTF-8 输出保护**：12 个新脚本统一加 `# 步骤 0X-Y｜...`
- **结果留档**（权重不进仓库，指标进仓库）：
  - `05_LLM/results/llm_zero_shot_summary.json`
  - `06_model_quantization/reports/quantization_report.{md,json}`
- **`05_LLM/api.py` 参数校验**：原来 `data.get('text').strip()` 在缺字段时抛 500，且所有错误都返回 HTTP 200（body 里写 code 500）→ 改为缺 `text` 返回 **HTTP 400**，模型调用失败返回 **HTTP 502**
- **`05_LLM/app.py` 前端健壮性**：原来只看 `status_code == 200` 就取 `result['data']`，一旦后端报错就 KeyError → 改为 `raise_for_status()` + 展示 `category` 与 `reason`
- **大文件隔离**：`06/models/`（390 MB fp32 副本 + 146 MB int8）由 `0*/models/` 覆盖；`05_LLM/outputs/`（100 KB CSV）由项目内 `.gitignore` 覆盖

---

## 三、今日验收结果

| 检查项 | 结果 |
|---|---|
| 12 个脚本语法 | ✅ 全部通过 |
| `05_LLM` 评测 | ✅ 500 条全 success，accuracy **0.8000** / macro-F1 0.7976 |
| `05_LLM` 延迟/token | ✅ 平均 877 ms/条，183,630 token/500 条 |
| `06` 量化 | ✅ `model_state.pt` 145.97 MB 生成成功 |
| `06` 体积对比 | ✅ 390.17 → 145.97 MB，**2.67×** |
| `06` 精度对比 | ✅ FP32 0.9447 / INT8 0.9448（无衰减） |
| `06` 性能基准 | ✅ p50 746.6 → 663.4 ms（**1.13×**），p95 1201.9 → 820.0 ms |
| 大文件隔离 | ✅ 536 MB 模型产物、100 KB CSV、`.env` 全部未入库 |
| 接口/前端 | ⚠️ `05_LLM/api.py` 与 `app.py` **未实跑**（会消耗 API 额度且需起服务，留给你验证） |

---

## 四、本次改动清单（相对 v1.2.0）

**新增**
- `05_LLM/src/`：`classifier.py`、`predict_one.py`、`evaluate.py`、`api.py`、`app.py`（5 个）
- `05_LLM/results/llm_zero_shot_summary.json`
- `06_model_quantization/`：`src/` 6 个脚本 + `test/01_快速演示模型量化.py` + `reports/` 2 个报告 + `README.md`
- `docs/release-notes-v1.3.0.md`

**修改**
- `05_LLM/src/deepseek_client.py`（你的改动：去掉编码保护、`__main__` 里多打印 `resp` —— 保留）
- `05_LLM/README.md`：改写为「已完成」版，含结论、混淆分析、token 成本
- `05_LLM/src/api.py`：参数校验 + 正确状态码
- `05_LLM/src/app.py`：错误处理 + 展示 reason
- `README.md`：分天项目表加 `06`、模型对比表加 LLM（0.8000）与 BERT-INT8 行、逐天结论加第 4 条

**未改动**
- `src/`（随机森林）、`03_fastext/`（FastText）、`04_BERT/`（BERT，仅被 06 复制了模型）、`预习/`、`data/`、`artifacts/`

---

## 五、已知问题

1. **`05_LLM` 只评测了 500 条抽样**（5%），95% 置信区间约 ±3.5 个百分点，真实值大概 0.76~0.84。
2. **串行请求**：500 条约 7 分钟，全量 1 万条要 2 小时+，应改并发 + 缓存。
3. **`06` 的 INT8 加载很绕**：只存了 `state_dict`，必须先建 FP32 结构再 `quantize_` 再 `load_state_dict`。
4. **`06` 与 `04` 的 max_length 不同**（128 vs 64），两次的 0.9447 / 0.9441 不能直接对比。
5. **`06` 只做了动态量化**，没有静态量化对比。
6. **`05_LLM/api.py`、`app.py` 未实跑**（避免消耗 API 额度）。

---

## 六、下一步

- [ ] `05_LLM`：few-shot（每类 2~3 例）看能否从 0.80 拉到 0.88+
- [ ] `05_LLM`：针对 `politics`/`society` 与 `finance`/`stocks` 收紧 prompt 界定语
- [ ] `05_LLM`：并发 + 缓存 + 全量 dev，得到无偏基线
- [ ] `06`：静态量化对比；batch=1 的在线延迟；INT8 存成完整模型目录
- [ ] 四模型 + INT8 的统一对比表（成本 / 延迟 / 精度三维）
- [ ] "LLM 伪标签 → 蒸馏到 BERT" 的性价比实验
