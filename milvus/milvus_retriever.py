"""
chunk检索，支持：
1、稀疏索引检索
2、密集索引检索
3、混合检索
"""
import json
import os
import sys
from typing import Dict, Any
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import embedding_utils


class MilvusRetriever:
    def __init__(self, collection_name: str, milvus_client: MilvusClient, top_k: int = 10):
        self.collection_name = collection_name
        self.milvus_client = milvus_client
        self.top_k = top_k

    def dense_search(self, query_dense_embedding, limit=10):
        # IP：采用内积度量
        # nprobe：IVF 索引的原理是，先通过聚类算法把所有向量分成许多个“桶”，搜索时，理论上只需要在与查询向量最相近的几个桶里找，而不是大海捞针般地搜索全部向量
        # nprobe = 10 意味着，系统会选出离查询向量最近的 10 个聚类中心，然后只在这 10 个桶里进行详细搜索
        # - 值越大：意味着搜索的范围越广，召回率越高（结果更准），但消耗的时间和资源也越多，速度越慢
        # - 值越小：搜索范围小，速度越快，但可能会错过藏在其他桶里的正确结果，精度会下降
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_dense_embedding],
            anns_field="dense",
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params=search_params
        )
        return res[0]

    def sparse_search(self, query, limit=10):
        """
        全文检索
        """
        return self.milvus_client.search(
            collection_name=self.collection_name,
            data=query,
            anns_field="sparse",
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
            search_params={"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}}  # 搜索时要忽略的低重要性词语的比例
        )[0]

    def hybrid_search(
            self,
            query_dense_embedding,
            query,
            sparse_weight=1.0,
            dense_weight=1.0,
            limit=10
    ):
        filter_expr = None
        # filter_expr = "category == 'text'"
        # dense query
        dense_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        dense_req = AnnSearchRequest(
            data=[query_dense_embedding],
            anns_field="dense",
            param=dense_search_params,
            limit=limit,
            expr=filter_expr
        )
        # sparse query
        sparse_search_params = {"metric_type": "BM25", 'params': {'drop_ratio_search': 0.2}}
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param=sparse_search_params,
            limit=limit,
            expr=filter_expr
        )
        # 重排
        rerank = WeightedRanker(sparse_weight, dense_weight)
        return self.milvus_client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[sparse_req, dense_req],
            ranker=rerank,  # 重排算法
            limit=limit,
            output_fields=["text", 'category', 'filename', 'image_path', 'title'],
        )[0]

    def retrieve(self, query: str) -> list[Dict[str, Any]]:
        if os.path.isfile(query):
            # 输入为图片
            embedding = embedding_utils.vl_embed(image=query)
            # 应用dense_search
            search_result = self.dense_search(query_dense_embedding=embedding, limit=self.top_k)
        else:
            # 文本
            embedding = embedding_utils.vl_embed(text=query)
            search_result = self.hybrid_search(
                query_dense_embedding=embedding,
                query=query,
                limit=self.top_k
            )
        print(json.dumps(search_result))
        docs = []
        for hit in search_result:
            docs.append({"text": hit.get("text"), "category": hit.get("category"), "image_path": hit.get("image_path"),
                         "filename": hit.get("filename")})
        return docs


# test
if __name__ == '__main__':
    from create_milvus_collection import client, DOC_COLLECTION_NAME
    retriever = MilvusRetriever(
        collection_name=DOC_COLLECTION_NAME,
        milvus_client=client,
        top_k=10
    )
    # results = retriever.retrieve(query="有界流和无界流")
    results = retriever.retrieve(query=r"D:\python_workspace\multimodal-RAG\data\images\f974ab743c58ca5599dff13fc4efcbc9.png")
    for ele in results:
        print(ele)
