"""
milvus建表模版
"""

from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(
    "http://172.23.39.215:19530",
    "root",
    "Milvus"
)

# 创建schema
schema = client.create_schema()


# 添加主键字段，指定auto_id，int型则自增，varchar型则随机生成字符串
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
# 添加标量字段，用于存放原始文本内容，需要全文检索，则需要指定分词器
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=1024,
    enable_analyzer=True,
    analyzer_params={
        "tokenizer": "jieba",  # 指定分词器，中文指定jieba，英文为standard
        "filter": ["cnalphanumonly"]  # 指定过滤器（中文），该过滤器删除任何非汉字的标记
        # "filter": [  # 指定过滤器（英文）
        #     "lowercase",  # 将所有标记转化为小写，从而实现不区分大小写的搜索
        #     {
        #         "type": "stemmer",  # 将单词还原成词根的形式，以支持更广泛的匹配
        #         "language": "english"
        #     },{
        #         "type": "stop",  # 删除常见的英文停止词，以便集中搜索文本中的关键词语
        #         "stop_words": "_english_"
        #     }
        # ]
    }
)
# 添加标量字段
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=1024, nullable=True)
schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=1024, nullable=True)
schema.add_field(field_name="filetype", datatype=DataType.VARCHAR, max_length=1024, nullable=True)
schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024, nullable=True)

# 添加稀疏向量字段
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

# 添加密集向量字段
schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)


# 添加方法，将文本转化为稀疏向量
bm25_function = Function(
    name="text_bm25_emb",
    input_field_names=["text"],  # 文本字段
    output_field_names=["sparse"],  # 稀疏向量字段
    function_type=FunctionType.BM25
)
schema.add_function(bm25_function)


# 创建索引
index_params = client.prepare_index_params()

# 稀疏向量 索引
index_params.add_index(
    field_name="sparse",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",
        "bm25_k1": 1.2,  # 1.2~2.0 词频（TF）的饱和度：高频词的贡献越大，词频影响越线性，饱和度增长越慢(控制一个词出现多少次才算“多”)
        "bm25_b": 0.75  # 0.0~1.0 文档长度归一化强度：文档长度的影响越大，对长文档的惩罚越强(控制“长篇大论”相对于“言简意赅”的劣势有多大，避免长文档仅仅因为包含更多词汇而在相似度计算中占据不公平的优势)
    }
)

# 密集向量 索引
index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="AUTOINDEX",
    metric_type="IP"  # 向量内积
)


# 创建collection
client.create_collection(
    collection_name="doc_collection",
    schema=schema,
    index_params=index_params
)


# 查看collection信息
desc = client.describe_collection(collection_name="doc_collection")
print(desc)