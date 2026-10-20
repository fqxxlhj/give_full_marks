# 04 · BERT 新闻标题分类

> 2026-10-20 完成微调与部署。**dev 准确率 0.9411（macro-F1 0.9412）**，比 FastText 的 0.9012 **+3.99 个百分点**。
>
> 代价：**CPU 上训练 18.56 小时 / 5 轮**，模型 390 MB。

---

## 1. 结果一览

### 训练曲线（dev 10000 条，按 macro-F1 选最优）

| 轮次 | train loss | train acc | dev acc | dev macro-F1 | 学习率 | 累计耗时 | 是否最优 |
|---|---|---|---|---|---|---|---|
| 1 | 0.4025 | 0.8825 | 0.9308 | 0.9310 | 1.78e-05 | 3.68 h | ✅ |
| 2 | 0.1939 | 0.9446 | 0.9336 | 0.9336 | 1.33e-05 | 6.87 h | ✅ |
| 3 | 0.1252 | 0.9659 | 0.9379 | 0.9378 | 8.89e-06 | 9.89 h | ✅ |
| **4** | **0.0758** | **0.9807** | **0.9411** | **0.9412** | 4.44e-06 | 15.08 h | ✅ **最终采用** |
| 5 | 0.0392 | 0.9908 | 0.9410 | 0.9410 | 0.00e+00 | 18.56 h | ❌ 未提升 |

**怎么读这张表**：
- 第 1 轮就有 0.9308 —— BERT 预训练权重已经把中文语义学到了，微调只是"改作业"
- 每轮约 **3.7 小时**（CPU），5 轮 18.56 小时
- **第 5 轮 train acc 0.9908 但 dev 只有 0.9410**，说明开始过拟合，早停在这个位置是对的

### 测试集结果（test 10000 条，用第 4 轮的最优模型）

**accuracy 0.9441 ｜ macro-F1 0.9442**（`test_model.py` 实跑，CPU 推理 4 分钟）

| 类别 | precision | recall | f1 | 类别 | precision | recall | f1 |
|---|---|---|---|---|---|---|---|
| sports | 0.9859 | 0.9790 | **0.9824** | society | 0.9276 | 0.9480 | 0.9377 |
| education | 0.9699 | 0.9660 | 0.9679 | finance | 0.9401 | 0.9260 | 0.9330 |
| entertainment | 0.9573 | 0.9650 | 0.9612 | politics | 0.9237 | 0.9320 | 0.9278 |
| realty | 0.9561 | 0.9590 | 0.9576 | science | 0.9028 | 0.9190 | 0.9108 |
| game | 0.9712 | 0.9440 | 0.9574 | stocks | 0.9085 | 0.9030 | **0.9057** |

**两个值得注意的点**：
1. **test（0.9441）≈ dev（0.9411）**，没有明显落差 → 模型没有对着 dev 过拟合，泛化正常
2. **最难的两类始终是 `stocks`(0.9057) 和 `science`(0.9108)**，和 FastText 时期完全一致（当时 stocks 0.8336 / science 0.8420）。说明这不是模型问题，是**数据本身就存在边界模糊**（股票新闻常带宏观金融词，科技新闻常带社会事件词），要进一步提升得靠更细的标注规范或数据增强

### 与前面几个模型对比

| 模型 | dev 准确率 | dev macro-F1 | 模型体积 | 训练耗时 | 设备 |
|---|---|---|---|---|---|
| TF-IDF + 随机森林（v1.0.0） | 0.7306 | — | 169 MB | 分钟级 | CPU |
| FastText 词级 + 固定超参 | 0.9012 | 0.9002 | 100.2 MB | 十几秒 | CPU |
| FastText autotune | 0.9130 | — | 11.65 GB | 5 分钟搜索 | CPU |
| **BERT-base-chinese** | **0.9411** | **0.9412** | **390 MB** | **18.56 小时** | **CPU** |

> 测试集成绩：**accuracy 0.9441 / macro-F1 0.9442**

**结论**：BERT 用 18.56 小时换来 +3.99 个点（相对 FastText）。**换算成"每分钟训练换多少点"**，FastText 完胜；但只要上 GPU，这个 18 小时会变成 30 分钟以内，BERT 就全面占优了。

---

## 2. 目录结构

```
04_BERT/
├── README.md                 ← 本文件
├── requirements.txt
├── results/                  ← 训练结果留档（小 JSON，入库；权重不入库）
│   ├── best_metrics.json         第 4 轮指标：acc 0.9411 / f1 0.9412
│   ├── history.json              5 轮完整训练曲线
│   ├── metadata.json             类别列表 + max_length + 最优 f1
│   └── training_args.json        训练参数快照
├── data/                     ← 原始数据（10.5 MB，不入库）
│   ├── train.txt  dev.txt  test.txt  class.txt
├── pretrained/
│   └── bert-base-chinese/    ← 1.24 GB 预训练权重（不入库）
├── models/
│   └── best_model/           ← 微调产物（390 MB，不入库）
│       ├── model.safetensors    390 MB
│       ├── config.json  tokenizer.json  tokenizer_config.json
│       ├── metadata.json        推理时读它拿 class_list / max_length
│       └── best_metrics.json  history.json  training_args.json
└── src/
    ├── config.py             ← 04-0 配置（路径 + 超参 + 设备自动选择）
    ├── data.py               ← 04-1 数据管线（Dataset / collate_fn / DataLoader）
    ├── metrics.py            ← 04-2 评估指标（acc / P / R / macro-F1 / 报告）
    ├── train.py              ← 04-3 训练循环（warmup + 早停 + 存最优）
    ├── test_model.py         ← 04-4 测试集评估
    ├── predictor.py          ← 04-5 推理封装（单条 / 批量）
    ├── api.py                ← 04-6 Flask 接口（端口 5002）
    ├── app.py                ← 04-7 Streamlit 前端（单条 + 批量）
    └── api_test.py           ← 04-8 接口冒烟测试
```

---

## 3. 运行顺序（全部在 `src/` 下执行）

```powershell
cd 04_BERT\src

python train.py           # 04-3 训练（CPU 约 18 小时；有 GPU 会快几十倍）
python test_model.py      # 04-4 用最优模型跑测试集
python predictor.py       # 04-5 单条推理自测
python api.py             # 04-6 起接口（5002，保持窗口不关）
python api_test.py        # 04-8 批量接口冒烟（另开窗口）
python app.py             # 04-7 前端页面（再开一个窗口）
```

> 训练脚本会写 `models/best_model/`；已有模型时可直接从第 04-4 步开始，不用重训。

---

## 4. 训练细节

| 项 | 值 | 说明 |
|---|---|---|
| 预训练模型 | `bert-base-chinese`（本地） | 离线加载，不联网 |
| 分类头 | 10 类 | `num_labels=10`，同时传 `id2label`/`label2id` |
| batch_size | 16 | 显存/内存不足就降到 8 |
| max_length | 64 | 标题 P95 只有 25 字，64 足够 |
| 学习率 | 2e-5 | BERT 微调标准量级 |
| weight_decay | 0.01 | — |
| warmup | 10% 总步数 | `get_linear_schedule_with_warmup` |
| 梯度裁剪 | 1.0 | 防梯度爆炸 |
| 早停 | 按 dev macro-F1，patience 4 | 实际第 5 轮没提升就停了 |
| 保存策略 | 只存 dev-F1 最高的那轮 | `save_pretrained` + tokenizer + metadata.json |

**为什么按 F1 而不是 accuracy 选模型**：10 个类别虽然均衡，但"股市 / 金融"、"科技 / 社会"边界模糊，macro-F1 对每一类平权，更能反映薄弱类别。

**训练循环五步**（`train.py` 核心）：
`zero_grad` → `forward` → `loss.backward` → `clip_grad_norm_` → `optimizer.step()` + `scheduler.step()`

---

## 5. 推理与部署

### `predictor.py` — 推理封装

```python
predictor = BertNewsClassifierPredictor()          # 从 best_model 加载模型 + tokenizer
predictor.predict_one("56岁女子退休后考研上岸")
# {'text': ..., 'pred': 'education', 'score': 0.99xx}

predictor.predict_batch(["文本1", "文本2", ...])    # 批量，一次 tokenize + 一次 forward
```

- 类别列表和 `max_length` **从 `metadata.json` 读**，不写死在代码里 —— 换模型不用改代码
- 输出 `score` 是 **softmax 后的概率**（0~1），前端可以直接画进度条
- `predict_batch` 会校验空文本并抛 `ValueError`

### `api.py` — Flask 接口（端口 5002）

| 接口 | 方法 | 请求 | 返回 |
|---|---|---|---|
| `/health` | GET | — | `{"status":"ok"}` |
| `/predict` | POST | `{"text": "..."}` | `{"data": {"text","pred","score"}}` |
| `/predict_batch` | POST | `{"texts": ["...","..."]}` | `{"data": [{...}, ...]}` |

> **端口为什么是 5002**：`03_fastext/src/api.py` 占用了 5001，两个服务可以同时跑。

### `app.py` — Streamlit 前端

- 左侧栏可以改 API 地址、点「检测后端服务」
- **单条预测**页：类别 + 分数 + 置信度进度条 + 原始 JSON
- **批量预测**页：每行一条，最多 200 条，表格展示 + 失败计数

---

## 6. 已知问题

1. **训练在 CPU 上跑了 18.56 小时**：`models/setup/wheels/` 里下了 `torch-2.14.1+cu126` 的轮子但没装成，`training_args.json` 里 `"device": "cpu"` 可证。建议装好 CUDA 版 torch 再训。
2. **`train.py` 只写 `metadata.json`**，不写 `history.json` / `best_metrics.json` / `training_args.json`。仓库里这三个文件是**早期更完整版本**的产物（`training_args.json` 里还留着原路径 `D:\soft\python\NLP\投满分项目\04_bert`），已备份到 `results/` 留档。想让脚本自己产出这些文件，需要把曲线记录逻辑补回 `train.py`。
3. **`api_test.py` 原来跑不通**：端口写 5001（应为 5002）、字段写 `text`（`/predict_batch` 要 `texts`）→ 已修正。
4. **前端「批量预测」不是真批量**：它是前端 for 循环逐条调 `/predict`，200 条会很慢。`api.py` 已经有 `/predict_batch`，前端应该改成一次请求。
5. **`api.py` 参数校验偏弱**：`data.get('text').strip()` 在 `text` 缺失时抛 500（已按 03_fastext 同样方式修成返回 400）。
6. **`metadata.json` 里键名拼错**：`pretrained_mode` 应为 `pretrained_model`（推理不读它，暂不影响）。
7. **模型 390 MB，不入库**：`.gitignore` 已排除 `04_BERT/models/`，克隆后需自行训练或拷贝。

---

## 7. 下一步

- [ ] 装 CUDA 版 torch（`pip install torch --index-url https://download.pytorch.org/whl/cu126`），把 18.56 h 压到 1 h 内
- [ ] 前端批量页改调 `/predict_batch`，一次请求代替 N 次
- [ ] `train.py` 补回 `history.json` / `best_metrics.json` 写出
- [ ] 针对 `stocks` / `science` 两个弱类做错误分析（混淆矩阵 + 抽样看错例）
- [ ] 与 `05_LLM` 的大模型零样本分类做对比（同 dev 口径）
- [ ] 蒸馏 / 量化，把 390 MB 压到 100 MB 以内
