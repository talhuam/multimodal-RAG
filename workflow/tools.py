import os
import sys
from langchain_core.tools import tool
from pymilvus import AnnSearchRequest, WeightedRanker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.embedding_utils import vl_embed
from milvus.create_milvus_collection import client, CONTEXT_COLLECTION_NAME
from workflow.evaluate_node import rag_evaluator
from utils.log_utils import log


@tool("search_context", parse_docstring=True)
async def search_context(query: str = None, user_name: str = None) -> str:
    """
    根据用户的输入，检索与查询相关的历史上下文信息，然后给出正确的回答。

    Args:
        query: (可选)用户刚刚输入的文本内容。
        user_name: (可选)当前的用户名。

    Returns:
        从历史上下文中检索到的结果。
    """
    query_embedding = vl_embed(text=query)
    filter_expr = None
    if user_name:
        # 如果user_name不为空，则检索属于该用户的历史上下文，否则检索所有用户的上下文
        filter_expr = f"user == '{user_name}'"

    # 构建milvus查询，混合检索
    dense_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    dense_req = AnnSearchRequest(
        [query_embedding], "context_dense", dense_search_params, limit=3, expr=filter_expr
    )
    sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
    sparse_req = AnnSearchRequest(
        [query], "context_sparse", sparse_search_params, limit=3, expr=filter_expr
    )

    rerank = WeightedRanker(1.0, 1.0)
    res = client.hybrid_search(
        collection_name=CONTEXT_COLLECTION_NAME,
        reqs=[sparse_req, dense_req],
        ranker=rerank,  # 重排算法
        limit=3,
        output_fields=["context_text"]
    )
    context_pieces = []
    for hit in res[0]:
        context_pieces.append(f"{hit.get('context_text')}")
    # 调用评估模块评估
    score = await rag_evaluator.evaluate_context(query, context_pieces)
    log.info(f"上下文检索后，评估分数为: {score}")
    if score < 1.0:  # 评估分数小于1.0，则返回空
        context_pieces = []

    return "\n".join(context_pieces) if context_pieces else "没有找到相关的历史上下文信息。"