import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from workflow.workflow_state import MultiModalRAGState
from utils.log_utils import log
from utils.embedding_utils import vl_embed
from milvus.create_milvus_collection import client, DOC_COLLECTION_NAME
from milvus.milvus_retriever import MilvusRetriever


m_re = MilvusRetriever(DOC_COLLECTION_NAME, client)


def retriever_node(state: MultiModalRAGState):
    """检索知识库并返回"""
    if state.get('input_type') == 'only_image':
        log.info(f"开始从知识库中检索图片：{state.get('input_image')}")
        embedding = vl_embed(image=state.get('input_image'))
        results = m_re.dense_search(embedding, limit=3)

    else:
        embedding = vl_embed(text=state.get('input_text'))
        results = m_re.hybrid_search(embedding, state.get('input_text'), limit=3)
    log.info(f"从知识库中检索到的结果：{results}")

    # 返回文档内容
    images = []  # 图片路径列表
    docs = []
    print(results)
    for hit in results:
        if hit.get('category') == 'image':
            images.append(hit.get('image_path'))
        docs.append({"text": hit.get("text"), "category": hit.get("category"), "image_path": hit.get("image_path"),
                     "filename": hit.get("filename"), })

    # 返回并更改状态
    return {'context_retrieved': docs, 'images_retrieved': images}