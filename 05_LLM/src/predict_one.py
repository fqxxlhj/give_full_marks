# 步骤 05-4｜批量预测：500 条抽样、断点续跑、逐条落盘（失败也记录一行）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import time

import pandas as pd

from classifier import classify_news
from config import config

# #单条测试
# news='国家数据集团成立'
# result=classify_news(news)
# print(result)

# 定义预测输出的文件
output_dir = config.output_dir / 'dev_predictions.csv'


def load_data(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['news', 'y_true'], encoding='utf-8')
    # 给每条数据增加rowid
    df.insert(0, 'row_id', range(len(df)))
    return df


# 获取已经完成预测的新闻编号
def load_finish_ids():
    # 如果结果文件不存在
    if not output_dir.exists():
        return set()
    # 读取已有的结果
    old_df = pd.read_csv(output_dir)
    # 只把success当作完成
    success_df = old_df[old_df['status'] == 'success']
    return set(success_df['row_id'].astype(int).tolist())


# 拼接结果
def append_result(result):
    # 将单条的结果转换成DataFrame
    result_df = pd.DataFrame([result])
    # 判断文件是否存在
    file_exists = output_dir.exists()
    # 追加写入
    result_df.to_csv(
        output_dir,
        mode='a',
        header=not file_exists,  # 如果表头存在，就不需要重新
        index=False,  # 不保存索引
        encoding='utf-8'
    )

# 批量预测数据
def run_batch():
        # 读取数据
        df = load_data(config.dev_path)
        # 测试阶段随机取500条数据进行演示
        df = df.sample(n=500, random_state=42)
        # 获取已经完成数据的id
        finish_ids = load_finish_ids()
        # 逐条处理数据
        for row in df.itertuples(index=False):
            if row.row_id in finish_ids:
                continue
            start_time = time.perf_counter()  # 获取当前时间
            try:
                # 调用模型预测
                pred = classify_news(row.news)
                # 计算请求耗时
                latency_ms = (time.perf_counter() - start_time) * 1000
                # 构造成功结果
                result = {
                    'row_id': row.row_id,
                    'title': row.news,
                    'true_label': row.y_true,
                    'pred_label': pred['category'],
                    'reason': pred['reason'],
                    'status': 'success',
                    'latency_ms': latency_ms,
                    'prompt_tokens': pred['prompt_tokens'],
                    'completion_tokens': pred['completion_tokens'],
                    'model': config.model_name,
                    'error': ''
                }
            except Exception as error:
                result = {
                    'row_id': row.row_id,
                    'title': row.news,
                    'true_label': row.y_true,
                    'pred_label': None,
                    'reason': None,
                    'status': 'fail',
                    'latency_ms': None,
                    'prompt_tokens': None,
                    'completion_tokens': None,
                    'model': config.model_name,
                    'error': str(error)
                }
            append_result(result)


if __name__ == '__main__':
    run_batch()

