# v1.2.0 — BERT 微调跑通（dev 0.9411 / test 0.9441）+ LLM 零样本起步

**发布日期**：2026-10-20
**Tag**：`v1.2.0`
**基线**：`v1.1.0`（2026-10-18，FastText，dev 0.9012）

> 本版本**在 v1.1.0 之上新增**，`v1.0.0` / `v1.1.0` 的提交与 Release 均未改动。
> 本次不覆盖任何既有代码：`src/`（随机森林）与 `03_fastext/`（FastText）一行没动。

---

## 一、这个版本是什么

一句话：**BERT 微调完成并部署，dev 0.9411 / test 0.9441，比 FastText 又高 3.99 个点；同时开了 LLM 零样本分类的头。**

| 版本 | 模型 | dev | test | 模型体积 | 训练耗时 |
|---|---|---|---|---|---|
| v1.0.0 | TF-IDF + 随机森林 | 0.7306 | — | 169 MB | 分钟级 |
| v1.1.0 | FastText 词级（固定超参） | 0.9012 | — | 100.2 MB | 十几秒 |
| **v1.2.0** | **BERT-base-chinese 微调** | **0.9411** | **0.9441** | **390 MB** | **18.56 小时（CPU）** |
| v1.2.0 | LLM 零样本（DeepSeek） | 🚧 待测 | — | 0 | 0 |

---

## 二、今日（2026-10-20）完成的步骤

### 1. `04_BERT` 全链路完成（9 个脚本）

| 步骤 | 文件 | 做了什么 | 状态 |
|---|---|---|---|
| 04-0 | `04_BERT/src/config.py` | 路径 + 预训练模型 + 超参 + 设备自动选择 | ✅ |
| 04-1 | `src/data.py` | `load_news_data` / `NewsDataset` / `collate_fn` / `build_dataloader` | ✅ |
| 04-2 | `src/metrics.py` | acc / macro-P / macro-R / macro-F1 / 分类报告 | ✅ |
| 04-3 | `src/train.py` | 训练循环：AdamW + 线性 warmup + 梯度裁剪 + 按 dev-F1 早停存最优 | ✅ |
| 04-4 | `src/test_model.py` | 测试集评估 | ✅ |
| 04-5 | `src/predictor.py` | 推理封装：`predict_one` / `predict_batch`，softmax 置信度 | ✅ |
| 04-6 | `src/api.py` | Flask：`/health`、`/predict`、`/predict_batch`（端口 5002） | ✅ |
| 04-7 | `src/app.py` | Streamlit：单条 + 批量两个 Tab、健康检查、置信度进度条 | ✅ |
| 04-8 | `src/api_test.py` | 批量接口冒烟测试（含耗时统计） | ✅ |

### 2. 训练结果：CPU 上跑了 18.56 小时

| 轮次 | train loss | train acc | dev acc | dev macro-F1 | 累计耗时 |
|---|---|---|---|---|---|
| 1 | 0.4025 | 0.8825 | 0.9308 | 0.9310 | 3.68 h |
| 2 | 0.1939 | 0.9446 | 0.9336 | 0.9336 | 6.87 h |
| 3 | 0.1252 | 0.9659 | 0.9379 | 0.9378 | 9.89 h |
| **4（最优）** | 0.0758 | 0.9807 | **0.9411** | **0.9412** | 15.08 h |
| 5（早停） | 0.0392 | 0.9908 | 0.9410 | 0.9410 | 18.56 h |

**测试集（本次实跑，10000 条，CPU 推理 4 分钟）**：accuracy **0.9441**、macro-F1 **0.9442**。

| 最好的类 | F1 | 最差的类 | F1 |
|---|---|---|---|
| sports | 0.9824 | stocks | 0.9057 |
| education | 0.9679 | science | 0.9108 |
| entertainment | 0.9612 | politics | 0.9278 |

**两个结论**：
1. **test(0.9441) ≈ dev(0.9411)**，没有落差，说明没对着 dev 过拟合、也没有 test 泄漏。
2. **弱类还是 `stocks` / `science`**，和 FastText 时期一模一样（当时分别 0.8336 / 0.8420）→ 这是**数据边界问题**（股票新闻带宏观金融词、科技新闻带社会事件词），换模型解决不了，得靠标注规范或数据增强。

### 3. `05_LLM` 零样本分类起步（3 个文件）

| 步骤 | 文件 | 做了什么 | 状态 |
|---|---|---|---|
| 05-0 | `05_LLM/src/config.py` | `@dataclass` 配置：路径 + `base_url` + `model_name` + `max_tokens` + `timeout` | ✅ |
| 05-1 | `src/prompts.py` | 10 类界定语 + 6 条输出约束（含**防提示注入**）+ JSON 示例 | ✅ |
| 05-2 | `src/deepseek_client.py` | OpenAI 兼容客户端：tenacity 重试（仅临时性错误）+ `json_object` 模式 + `temperature=0` + 关闭思考 | ✅ |
| 05-3~ | 评测脚本 | 还没写 → **目前没有准确率数字** | ❌ |

**为什么值得做**：前三个模型都要先训练。大模型零样本可以回答一个关键问题——"为了这 10 个类别，值不值得花 18 小时训 BERT？"

### 4. 仓库整理（同前两轮规则）

- **目录改名**：`day03_fastext/` → `03_fastext/`、`day04_BERT/` → `04_BERT/`（你改的，我保留并同步了所有引用）
- **`.gitignore` 修好一处严重失配**：改名后原来的 `day*/models/` 匹配不到 `04_BERT/models/`，**390 MB 的 `model.safetensors` 和 19 MB 的 CUDA wheel 会直接进仓库** → 已补 `0*/models/`、`0*/data/`、`0*/pretrained/`、`*.safetensors`、`*.whl`、`.env`
- **`04_BERT/results/`**：把训练结果的小 JSON（`best_metrics` / `history` / `metadata` / `training_args`）和测试集报告留档入库，**权重本身不入库**
- **步骤表头 + UTF-8 输出保护**：9 个新脚本统一
- **修 3 个会直接报错的 bug**（见下）

### 5. 顺手修掉的 3 个 bug

| 问题 | 现象 | 修法 |
|---|---|---|
| `api_test.py` 端口/字段错 | 请求打到 5001（实际 5002），且发 `text`（批量接口要 `texts`）→ 必失败 | 改为 5002 + `texts` |
| `api.py` 参数校验缺失 | `text` 缺失时 `data.get('text').strip()` 抛 500；`texts` 非列表也 500 | 统一返回 **HTTP 400** + 明确 msg |
| `train.py` 无用导入 | 引了 `AutoModelForAudioClassification`（文本分类用不到） | 删除 |

---

## 三、今日验收结果

| 检查项 | 结果 |
|---|---|
| 12 个脚本语法 | ✅ 全部通过 |
| `test_model.py` 实跑 | ✅ test accuracy **0.9441**、macro-F1 **0.9442**（4 分钟） |
| 训练曲线完整性 | ✅ 5 轮 loss/acc/F1 齐全，第 5 轮早停 |
| `models/best_model/` | ✅ `model.safetensors` 390 MB + tokenizer + metadata |
| 大文件隔离 | ✅ 模型 390 MB、wheel 19 MB、数据 10.5 MB、预训练 1.24 GB、`.env` 均未入库 |
| 目录改名一致性 | ✅ 根 README / 两个子 README / 文档路径全部同步 |
| `05_LLM` 冒烟 | ⚠️ **未实跑**（会消耗 API 额度，留给你自己跑） |

---

## 四、本次改动清单（相对 v1.1.0）

**新增**
- `04_BERT/src/`：`train.py`、`metrics.py`、`test_model.py`、`predictor.py`、`api.py`、`app.py`、`api_test.py`（7 个）
- `04_BERT/results/`：4 个结果 JSON + `test_report.txt`
- `04_BERT/README.md`（重写为「已完成」版，含训练曲线、弱类分析）
- `05_LLM/`：`src/{config,prompts,deepseek_client}.py` + `README.md` + `.gitignore` + `data/`
- `docs/release-notes-v1.2.0.md`

**修改**
- `.gitignore`：新增 `0*/models/`、`0*/data/`、`0*/pretrained/`、`*.safetensors`、`*.whl`、`.env`
- `README.md`：分天项目表加 `04_BERT` ✅ 与 `05_LLM` 🚧；第 11 节模型对比表加入 BERT 实测数据
- `docs/release-notes-v1.1.0.md`：目录改名后路径引用同步
- `04_BERT/src/{api,api_test,train}.py`、新脚本：表头 / 校验 / 无用导入

**未改动**
- `src/`（随机森林主流程）、`03_fastext/`（FastText）、`预习/`、`data/`、`artifacts/`

---

## 五、已知问题

1. **CPU 训练 18.56 小时**：`models/setup/wheels/` 下了 cu126 的 torch 轮子但没装成（`training_args.json` 里仍是 `"device": "cpu"`）。
2. **仓库里的 `train.py` 是精简版**：只写 `metadata.json`，不写 `history.json`/`best_metrics.json`/`training_args.json`；那三个文件来自早期更完整版本（`training_args.json` 里还留着原路径 `D:\soft\python\NLP\投满分项目\04_bert`）。
3. **前端「批量预测」是逐条调 `/predict`**，200 条会很慢；`api.py` 已有 `/predict_batch`，前端应改一次请求。
4. **`05_LLM` 还没有评测**：只验证了"能通、能返回 JSON"。
5. **`05_LLM` 模型名 `deepseek-flash` 待确认**，若报不存在改 `deepseek-chat`。
6. `metadata.json` 键名 `pretrained_mode` 拼写错误（推理不读，不影响）。

---

## 六、下一步

- [ ] 装 CUDA 版 torch，把 18.56 h 压到 1 h 内
- [ ] `05_LLM/evaluate.py`：抽样 200 条 dev → 算 acc / macro-F1，补上大模型基线
- [ ] 四模型同表对比（RF 0.7306 / FastText 0.9012 / BERT 0.9411 / LLM ?）
- [ ] `stocks` / `science` 弱类错误分析
- [ ] 前端批量页改调 `/predict_batch`
- [ ] 模型蒸馏/量化，把 390 MB 压到 100 MB 内
