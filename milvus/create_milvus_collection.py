"""
创建:
知识库存储表：t_doc_collection
长期记忆存储表：t_context_collection
"""
import os
import sys
from pymilvus import MilvusClient, DataType, Function, FunctionType

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.log_utils import log

DOC_COLLECTION_NAME = "t_doc_collection"
CONTEXT_COLLECTION_NAME = "t_context_collection"

client = MilvusClient(
    "http://localhost:19530",
    "root",
    "Milvus"
)


def create_doc_collection():
    schema = client.create_schema()
    # 字段
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                     analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
    schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='image_path', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=2048)
    # 索引
    bm25_function = Function(
        name="text_bm25_emb",  # Function name
        input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,  # Set to `BM25`
    )
    schema.add_function(bm25_function)
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
        metric_type="BM25",  # 倒排索引
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        },
    )
    index_params.add_index(
        field_name="dense",
        index_name="dense_inverted_index",
        index_type="AUTOINDEX",
        metric_type="IP"
    )

    client.create_collection(
        collection_name=DOC_COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )


def create_context_collection():
    schema = client.create_schema()
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
    # 某一条聊天记录的文本内容
    schema.add_field(field_name='context_text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                     analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
    # 用户名
    schema.add_field(field_name='user', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='timestamp', datatype=DataType.INT64, nullable=True)
    schema.add_field(field_name='message_type', datatype=DataType.VARCHAR, max_length=100, nullable=True)
    schema.add_field(field_name='context_sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name='context_dense', datatype=DataType.FLOAT_VECTOR, dim=2048)

    bm25_function = Function(
        name="text_bm25_emb",  # Function name
        input_field_names=["context_text"],  # Name of the VARCHAR field containing raw text data
        output_field_names=["context_sparse"],
        function_type=FunctionType.BM25,  # Set to `BM25`
    )
    schema.add_function(bm25_function)
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="context_sparse",
        index_name="context_sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        },
    )
    index_params.add_index(
        field_name="context_dense",
        index_name="context_dense_inverted_index",
        index_type="AUTOINDEX",
        metric_type="IP"
    )

    client.create_collection(
        collection_name=CONTEXT_COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )


if __name__ == '__main__':
    collections = client.list_collections()
    if DOC_COLLECTION_NAME not in collections:
        create_doc_collection()
        log.info(f"{DOC_COLLECTION_NAME}表创建成功")
    else:
        log.info(f"{DOC_COLLECTION_NAME}表已存在，跳过创建流程")

    if CONTEXT_COLLECTION_NAME not in collections:
        create_context_collection()
        log.info(f"{CONTEXT_COLLECTION_NAME}表创建成功")
    else:
        log.info(f"{CONTEXT_COLLECTION_NAME}表已存在，跳过创建流程")

    print(client.describe_collection(collection_name="t_doc_collection"))
    print(client.describe_collection(collection_name="t_context_collection"))