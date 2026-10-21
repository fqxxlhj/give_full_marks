# 06 · BERT 模型量化（INT8 动态量化）

> 2026-10-21。用 `torchao` 对 04 训练好的 BERT 做 **INT8 动态量化**：
>
> **体积 390 MB → 146 MB（2.67×）｜精度 0.9447 → 0.9448（不变）｜p50 746 ms → 663 ms（1.13×）**

---

## 1. 结果一览（全部实跑，CPU）

| 指标 | FP32 | INT8 | 变化 |
|---|---|---|---|
| 模型文件 | `model.safetensors` 390.17 MB | `model_state.pt` 145.97 MB | **2.67× 更小** |
| test accuracy | 0.9447 | 0.9448 | ±0.0001（噪声级） |
| test macro-F1 | 0.9448 | 0.9449 | ±0.0001 |
| 延迟 p50（batch=32） | 746.6 ms | 663.4 ms | **1.13× 更快** |
| 延迟 p95（batch=32） | 1201.9 ms | 820.0 ms | **1.47× 更快（尾部改善更明显）** |
| 折算单条延迟 | ≈23.3 ms | ≈20.7 ms | — |

> **延迟口径**：`benchmark.py` 取 `DataLoader` 的一个 batch（`batch_size=32`、`max_length=128`）重复前向 100 次，warmup 20 次。所以 746 ms 是**一批 32 条**的耗时，不是单条。

**一句话结论**：体积砍到 1/2.7、精度一点没掉，但 CPU 上只快 13%。**省显存/省磁盘非常划算，省时间不划算**。

---

## 2. 环境

```powershell
conda activate compression     # D:\AnacondaEnvs\compression
pip install torchao
```

实测：Python 3.11.17 / torch 2.14.1+cpu / **torchao 0.18.0** / transformers 5.20.2

> ⚠️ 这个项目必须用 `compression` 环境，`fasttext_env` / `tensorflow_env` **都没装 torchao**。

---

## 3. 目录结构

```
06_model_quantization/
├── README.md
├── data/
│   └── test.txt                 ← 测试集（0.53 MB，不入库）
├── models/                      ← 模型产物（~536 MB，不入库）
│   ├── bert_fp32/               ← 从 04_BERT 复制的 FP32 模型（390 MB）
│   └── bert_int8/
│       └── model_state.pt       ← 量化后的 state_dict（146 MB）
├── reports/                     ← 结果留档（入库）
│   ├── quantization_report.md
│   └── quantization_report.json
├── src/
│   ├── config.py                ← 06-0 配置（@dataclass）
│   ├── quantize.py              ← 06-1 执行 INT8 动态量化
│   ├── dataset.py               ← 06-2 数据管线（Dataset + collate_fn + DataLoader）
│   ├── evaluate.py              ← 06-3 FP32 vs INT8 精度对比
│   ├── benchmark.py             ← 06-4 延迟基准（p50/p95 + 加速比）
│   └── model_compare.py         ← 06-5 体积对比
└── test/
    └── 01_快速演示模型量化.py      ← 最小可跑的量化 demo（玩具模型，3 分钟看懂）
```

---

## 4. 运行顺序（全部在 `src/` 下执行）

```powershell
conda activate compression
cd 06_model_quantization\src

python quantize.py        # 06-1 量化并保存 model_state.pt（约 1 分钟）
python model_compare.py   # 06-5 体积对比
python evaluate.py        # 06-3 精度对比（两个模型各跑 1 万条，约 8 分钟）
python benchmark.py       # 06-4 延迟基准（约 3 分钟）
```

先看原理再看代码，建议顺序：`test/01_快速演示模型量化.py`（玩具模型） → `src/quantize.py`（真实模型）。

---

## 5. 量化到底做了什么

```python
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

int8_model = copy.deepcopy(fp32_model)          # 复制，不动原模型
quantize_(int8_model, Int8DynamicActivationInt8WeightConfig())
torch.save(int8_model.state_dict(), "model_state.pt")
```

| 名词 | 含义 |
|---|---|
| **INT8 权重** | 原来的 float32 权重（4 字节）压成 int8（1 字节），**这是 2.67× 体积的来源**（理论上限 4×，其余是 fp32 的 embedding / LayerNorm / 量化 scale 没被压） |
| **动态激活** | 每一层在前向时**现场统计激活值的范围**再量化，不需要校准数据集 |
| **只量化 Linear** | BERT 里的 QKV 投影、attention 输出投影、FFN 两层都是 `nn.Linear`，是主要算力，量化收益最高 |
| **不量化** | LayerNorm、softmax、GELU、embedding 查表 —— 这些对数值敏感或本来就不是矩阵乘，保持 fp32 |

**代价**：动态量化每层要多算一次 min/max（计算 scale/zero_point），这部分开销会吃掉一部分收益。

---

## 6. 为什么只快了 1.13×？（这是本节最值得记的一句话）

1. **序列太短**：`max_length=128`，而 BERT 的计算量随序列长度平方增长；短序列下矩阵乘占比没那么高，量化收益被稀释。
2. **算子没全覆盖**：上述 LayerNorm/softmax/GELU 仍是 fp32，且这些"非矩阵乘"在小 batch 下占比更大。
3. **动态量化的额外开销**：每层现场算激活范围。
4. **CPU 单机多线程的内存带宽瓶颈**：int8 的算力优势在 CPU 上体现得不如 GPU 明显，尤其 `torch 2.14+cpu` 没走 oneDNN 的最佳 kernel 时。

**想更快可以走**：
- 静态量化（`Int8StaticActivationInt8WeightConfig` + 校准集）→ 省掉动态统计开销
- 序列更长 / batch 更大时量化收益会变大
- 上 GPU：INT8 tensor core 才是量化的主战场

---

## 7. 已知问题

1. **加载方式很绕**：`model_state.pt` 只是 `state_dict`，**不能直接 `from_pretrained`**。必须先加载 FP32 结构 → `quantize_` 转成量化结构 → `load_state_dict`。见 `evaluate.py::load_int8_model`。想简化应该保存成完整模型目录（含 config/tokenizer）。
2. **只做了动态量化**，没做静态量化对比。
3. **`06/data/test.txt` 与 `04_BERT/data/test.txt` 是同一份数据**（0.53 MB），项目间各存一份，有点冗余。
4. **`reports/` 原来是空的**：本次已把实测结果写进 `reports/quantization_report.{md,json}`。
5. **本机 FP32 精度 0.9447 与 04 的 0.9441 略有差异**：因为这里 `max_length=128`（04 是 64），两者不是同一配置，**不要直接把 0.9447/0.9448 和 04 的 0.9441 对比**；同一配置下的结论是"INT8 与 FP32 无差异"。

---

## 8. 下一步

- [ ] 静态量化对比（需要 200~500 条校准集）
- [ ] 把 INT8 保存成完整模型目录（`config.json` + `tokenizer` + 量化 state），加载不再绕
- [ ] batch 1（真实在线推理形态）下的延迟对比
- [ ] GPU 上做 INT8，看加速比能到多少
- [ ] 和 `03_fastext` 的 100 MB 模型对比"同体积下的精度"（BERT_INT8 146 MB / 0.9448 vs FastText 100 MB / 0.9012）
