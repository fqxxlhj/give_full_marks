# 步骤 06-4｜延迟基准：warmup 20 + 重复 100 次，输出 p50/p95 与加速比

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import time
import statistics  # 用于计算中位数
import torch
from dataset import build_dataloader
from evaluate import load_int8_model,load_fp32_model
from config import config
from transformers import AutoTokenizer




# ... existing code ...
# 性能测试的函数
def benchmark_model(model,batch,warmup_steps=20,repeat_step=100):
    # 进入推理模式
    model.eval()

    with torch.no_grad():
        # 预热：只是为了触发 kernel 初始化/内存分配，不计时
        for _ in range(warmup_steps):
            model(**batch)

    latencies = []
    with torch.no_grad():
        for _ in range(repeat_step):
            start_time = time.perf_counter()
            outputs = model(**batch)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time)*1000
            latencies.append(latency_ms)

    # 计算p50
    p50=statistics.median(latencies)
    # 计算p95
    sort_latencies = sorted(latencies)
    p95_index = max(0, min(len(sort_latencies) - 1, int(len(sort_latencies) * 0.95) - 1))
    p95 = sort_latencies[p95_index]
    return {
        'p50_ms': p50,
        'p95_ms': p95,
        'all_ms': latencies
    }

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(config.fp32_model_dir)
    dataLoader = build_dataloader(config.test_path, tokenizer, False)
    # 获取dataLoader中的一个批次数据
    batch = next(iter(dataLoader))
    fp32_model = load_fp32_model()
    fp32_result = benchmark_model(fp32_model, batch, config.warmup_steps, config.benchmark_steps)
    print(fp32_result)

    print('、、、、、、、、、')
    int8_model = load_int8_model()
    int8_result = benchmark_model(int8_model, batch, config.warmup_steps, config.benchmark_steps)
    print(int8_result)

    # 加速比
    print(f"加速比: {fp32_result['p50_ms'] / int8_result['p50_ms']:.2f}x")
# ... existing code ...