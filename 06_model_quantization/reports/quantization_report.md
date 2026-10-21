# BERT INT8 量化报告（2026-10-21 实测）

| 指标 | FP32 | INT8 | 变化 |
|---|---|---|---|
| 体积 | 390.17 MB | 145.97 MB | **2.67× 更小** |
| test accuracy（1 万条） | 0.9447 | 0.9448 | ±0.0001 |
| test macro-F1 | 0.9448 | 0.9449 | ±0.0001 |
| p50（batch=32） | 746.6 ms | 663.4 ms | 1.13× |
| p95（batch=32） | 1201.9 ms | 820.0 ms | 1.47× |

- 方法：`torchao` `Int8DynamicActivationInt8WeightConfig`（动态量化，只量化 Linear）
- 环境：Python 3.11.17 / torch 2.14.1+cpu / torchao 0.18.0 / transformers 5.20.2
- 基准配置：batch_size=32、max_length=128、warmup 20 步、重复 100 次
- 说明：延迟是**一批 32 条**的耗时（折算单条约 23.3 ms → 20.7 ms）
