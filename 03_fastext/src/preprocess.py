# 步骤 03-1｜数据预处理：文本 -> __label__类别名 分词串（word= jieba 分词 / char= 逐字）

import sys  # 控制台输出编码容错：Windows 的 GBK 控制台打印乱码字符会直接报错

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import jieba
from config import Config

#创建配置对象
conf=Config()

# 加载类别映射
def load_label(class_path):
    # 创建一部字典用来保存映射关系
    id_to_name={}
    # 读取类别文件
    with open(class_path,'r',encoding='utf-8') as f:
        for idx,line in enumerate(f):
            id_to_name[idx]=line.strip()

    return id_to_name

# 编写函数统一进行文本处理
def preprocess_text(text,mode='word'):
    # 去掉首位空格
    text=text.strip()
    # 字符集的处理
    if mode=='char':
        tokens=list(text)
    elif mode=='word':
        tokens=jieba.lcut(text)
    else:
        raise ValueError("Invalid mode.Supported modes:'char','word'")
    # 去除tokens的空格
    tokens=[token for token in tokens if token.strip()]
    # 将列表转换成字符串
    process_text = ' '.join(tokens)
    return process_text

# 构建Fasttext的数据函数   __label__class_name
def build_fasttext_data(input_path,output_path,label_map,mode='word'):
    with open(input_path,'r',encoding='utf-8') as input_file,open(output_path,'w',encoding='utf-8') as output_file:
        count=0
        for line in input_file:
            # 去除首位空格
            line=line.strip()
            # 跳过空行
            if not line:
                continue
            # 分离类别和文本
            text,label=line.split('\t')
            # 标签转化成数字
            label_id=int(label)
            # 根据标签找到对应的class_name
            class_name=label_map[label_id]
            # 构造Fast text的标签
            fasttext_label=f"__label__{class_name}"
            # 文本统一预处理
            process_text=preprocess_text(text,mode)
            # 拼接Fast text的数据
            fasttext_line=f"{fasttext_label} {process_text}"
            # 写入文件
            output_file.write(fasttext_line+'\n')
            count+=1
    print(f"处理完成，共处理{count}行数据")



if __name__ == '__main__':
    print(load_label(conf.class_path))
    print(preprocess_text("这是一个测试", ))
    build_fasttext_data(conf.train_path, conf.processed_data_dir / 'train_word.txt', load_label(conf.class_path), 'word')
    build_fasttext_data(conf.train_path, conf.processed_data_dir / 'train_char.txt', load_label(conf.class_path), 'char')

    build_fasttext_data(conf.dev_path, conf.processed_data_dir / 'dev_word.txt', load_label(conf.class_path),
                        'word')
    build_fasttext_data(conf.dev_path, conf.processed_data_dir / 'dev_char.txt', load_label(conf.class_path),
                        'char')