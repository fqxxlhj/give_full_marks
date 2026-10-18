# day04 · BERT 新闻标题分类（进行中）

> 2026-10-18 开始，**目前只完成到"数据管线"**。目标：用 `bert-base-chinese` 微调，和 day03 的 FastText 做同口径对比。

---

## 1. 目标与技术路线

```
原始 txt ──► load_news_data ──► NewsDataset ──► collate_fn(tokenizer) ──► DataLoader
                                                                              │
                                              AutoModelForSequenceClassification(10 类)
                                                                              │
                                    train.py（待写）──► evaluate.py（待写）──► api.py（待写）
```

- 预训练模型：本地 `pretrained/bert-base-chinese`（离线，不联网下载）
- 分类头：10 类
- 关键超参已固化在 `config.py`，不写死在脚本里

---

## 2. 今日进度

| 步骤 | 文件 | 状态 | 说明 |
|---|---|---|---|
| 04-0 | `src/config.py` | ✅ 完成 | 路径、预训练模型目录、训练超参、设备自动选择 |
| 04-1 | `src/data.py` | ✅ 完成 | `load_news_data` / `NewsDataset` / `collate_fn` / `build_dataloader` |
| 04-2 | `src/train.py` | ❌ 未开始 | 训练循环（含 warmup、early stopping、保存 best_model） |
| 04-3 | `src/evaluate.py` | ❌ 未开始 | dev 准确率 + 分类报告 |
| 04-4 | `src/predictor.py` | ❌ 未开始 | 单条文本推理封装 |
| 04-5 | `src/api.py` | ❌ 未开始 | Flask 接口 |
| 04-6 | `src/app.py` | ❌ 未开始 | Streamlit 页面 |

`models/best_model/` 目前是**空目录** —— 还没训练过，没有权重。

---

## 3. 目录结构

```
day04_BERT/
├── README.md
├── requirements.txt
├── data/                     ← 原始数据（10.5 MB，不入库；与 day03 同源）
│   ├── train.txt  dev.txt  test.txt  class.txt
├── pretrained/
│   └── bert-base-chinese/    ← 1.24 GB，不入库（本地离线权重）
│       ├── config.json  vocab.txt  tokenizer.json  tokenizer_config.json
│       └── pytorch_model.bin (411 MB) / tf_model.h5 (478 MB) / flax_model.msgpack (409 MB)
├── models/
│   └── best_model/           ← 空，训练后写入
└── src/
    ├── config.py             ← 04-0 全局配置单例 `config`
    └── data.py               ← 04-1 数据管线
```

> `pretrained/` 里三套框架的权重各存了一份，实际只用 `pytorch_model.bin`；另外两份（tf / flax）共 887 MB 可以删掉。

---

## 4. 数据管线做了什么（`data.py`）

| 函数 | 作用 | 关键点 |
|---|---|---|
| `load_class_list()` | 读 `class.txt` → 10 个类别名 | **行号即标签编号**，顺序不能变 |
| `load_news_data(path)` | 读 `文本<TAB>标签` → `[(text, label_int), ...]` | 跳过空行；`split('\t', 1)` 防止文本里含制表符 |
| `NewsDataset` | PyTorch `Dataset`，`__getitem__` 返回 `(text, label)` | **不在这里分词**，留给 collate_fn 批量做 |
| `build_collect_fn(tokenizer)` | 把 batch 的文本批量 tokenize | `padding=True` + `truncation=True` + `max_length=64`；labels 转 `torch.long` |
| `build_dataloader(...)` | 组装 `DataLoader` | `num_workers=0`（Windows 教学环境）、GPU 时开 `pin_memory` |

**为什么把 tokenize 放在 collate_fn**：同一个 batch 内动态 padding 到该 batch 最长，比全局 padding 到 64 更省算力；`Dataset` 只存原始文本，内存占用小。

---

## 5. 配置项（`config.py`）

| 配置 | 值 | 说明 |
|---|---|---|
| `batch_size` | 16 | 显存不足就降到 8 |
| `max_length` | 64 | 新闻标题 P95 才 25 字，64 足够 |
| `num_epochs` | 5 | 配 early stopping 用 |
| `learning_rate` | 2e-5 | BERT 微调标准量级，别超过 5e-5 |
| `weight_decay` | 0.01 | 防过拟合 |
| `warmup_ratio` | 0.1 | 前 10% 步数线性 warmup |
| `patience` | 2 | dev 连续 2 轮不涨就停 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `num_workers` | 0 | Windows 下多进程容易报错，保持 0 |
| `device` | 自动 | `cuda` 可用就用，否则 `cpu` |

---

## 6. 已知问题（明天接着做时先看这里）

1. **CPU 训练会非常慢**：本机是 `torch 2.14.1+cpu`、无 CUDA。18 万条 × 5 轮，batch 16 → 每轮约 1.1 万步，CPU 上大约要 **3 小时/轮**。
   **建议**：先用前 1~2 万条跑通全流程，确认能出结果，再考虑全量或换 GPU。
2. ~~**`data.py` 的 `__main__` 里 tokenizer 路径是硬编码** `'../pretrained/bert-base-chinese'`~~ → **已在本次整理中修正**，改用 `config.pretrained_model_path`，在任何目录下运行都能找到权重。
3. **配置风格与 day03 不一致**：day03 是 `Config` 类 + 每个脚本 `conf = Config()`；这里是模块级单例 `config = Config()`。两种都能用，但同一仓库里最好统一（建议保留 day04 这种单例，更省事）。
4. **`padding=True` 是动态 padding**：换成 `padding='max_length'` 训练速度更稳定，但浪费算力。当前设置更省。
5. **`pretrained/` 1.24 GB + `data/` 10.5 MB 已列入 `.gitignore`**，克隆仓库后需自行准备（`transformers` 第一次运行可从 HuggingFace 下载同名模型）。

---

## 7. 明天要写的（骨架，按顺序）

**04-2 `train.py`**

```
1. tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
2. train_loader = build_dataloader(config.train_path, tokenizer, shuffle=True)
   dev_loader   = build_dataloader(config.dev_path,   tokenizer, shuffle=False)
3. model = AutoModelForSequenceClassification.from_pretrained(
              config.pretrained_model_path, num_labels=10).to(config.device)
4. optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
   scheduler = get_linear_schedule_with_warmup(total_steps*0.1, total_steps)
5. for epoch in range(num_epochs):
       model.train()
       for batch in train_loader:          # batch 已含 input_ids/attention_mask/labels
           out = model(**batch.to(device))
           loss.backward(); clip_grad_norm_(1.0); optimizer.step(); scheduler.step()
       在 dev 上算准确率 → 涨了就 save_pretrained(best_model_dir) 并记住最佳轮
       连续 patience 轮不涨 → break
```

**04-3 `evaluate.py`**：`AutoModelForSequenceClassification.from_pretrained(best_model_dir)` → dev 集跑 `argmax` → `classification_report`（与 day03 用同一套口径，方便对比）

**04-4~04-6**：照搬 day03 的 `predictor.py` / `api.py` / `app.py`，只把 `FastTextPredictor` 换成 BERT 推理（tokenizer + model），接口端口换成 5002 避免冲突。

---

## 8. 验收标准（做完后怎么算成功）

- [ ] `train.py` 能跑完至少 1 轮并写出 `models/best_model/`（含 `config.json` + `pytorch_model.bin`）
- [ ] `evaluate.py` 打印 dev 准确率，**且高于 FastText 的 0.9012**
- [ ] `api.py` 起服务，`POST /predict` 返回 `{"label": "sports", "probability": 0.9x}`
- [ ] `app.py` 页面能出结果
- [ ] 与 day03 的对比表更新到根目录 README
